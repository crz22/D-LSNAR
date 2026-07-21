import tifffile
import numpy as np
import os
import subprocess
from repaire.rep_utils import resample_swc_

def read_image(tif_path):
    return tifffile.imread(tif_path)

def load_swc(filepath):
    """
    加载 swc 文件，返回 N×7 的 numpy 数组
    跳过注释行和无效行
    """
    if not os.path.exists(filepath):
        print(f"[load_swc] file not found: {filepath}")
        return None

    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # 跳过注释行和空行
            if not line or line.startswith('#'):
                continue

            # 分割并过滤空字符串
            cells = [c for c in line.split() if c]

            # 只保留前 7 列（允许多余列）
            if len(cells) < 7:
                continue

            try:
                row = [float(c) for c in cells[:7]]
                rows.append(row)
            except ValueError:
                continue

    if not rows:
        return None

    return np.array(rows, dtype=np.float64)

# ------------------------------------------------------------------ #
#  resample_vaa3d
def resample_vaa3d(config,swc_path,output_path,stepLen='5'):
    V3DPath = config['Vaa3d_path']
    pluginName = config['Vaa3d_resample_plugin_path']
    funcName = "resample_swc"

    cmd = [V3DPath, '/x', pluginName, '/f', funcName, '/i', swc_path, '/o', output_path, '/p', stepLen]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
    except Exception as e:
        print(f"[resample_vaa3d] failed: {e}")

def prune_branch(swc, min_branch_lenth=3, bmin=13, bmax=244):
    """
    修剪短分支
    优化：预建父子关系字典，将内层 np.where 从 O(N) 降至 O(1)
    """
    swc1 = swc.copy()

    while True:
        n = swc1.shape[0]

        # 预建索引：id -> row，parent_id -> [child_rows]
        id_to_row = {int(swc1[i, 0]): i for i in range(n)}
        parent_to_children = {}
        for i in range(n):
            pid = int(swc1[i, -1])
            if pid not in parent_to_children:
                parent_to_children[pid] = []
            parent_to_children[pid].append(i)

        node2del = set()

        for i in range(n):
            cur_id = int(swc1[i, 0])

            # 只处理叶子节点（没有子节点）
            children = parent_to_children.get(cur_id, [])
            if len(children) > 0:
                continue

            # 沿着父链向上追踪
            cur_branch_nodes = [i]
            cur_row = i

            while True:
                pid = int(swc1[cur_row, -1])
                p_row = id_to_row.get(pid, None)
                if p_row is None:
                    break

                # 父节点的子节点数 > 1，说明到了分叉点
                p_id_val = int(swc1[p_row, 0])
                p_children = parent_to_children.get(p_id_val, [])
                if len(p_children) > 1:
                    break

                cur_branch_nodes.append(p_row)
                if len(cur_branch_nodes) > min_branch_lenth:
                    break
                cur_row = p_row

            # 边界上的分支：如果长度 > 1，允许保留
            if np.any(swc1[i, 2:5] <= bmin) or np.any(swc1[i, 2:5] >= bmax):
                if len(cur_branch_nodes) > 1:
                    continue

            # 短分支：标记删除
            if len(cur_branch_nodes) <= min_branch_lenth:
                node2del.update(cur_branch_nodes)

        if not node2del:
            break

        keep_mask = np.ones(n, dtype=bool)
        for idx in node2del:
            keep_mask[idx] = False
        swc1 = swc1[keep_mask]

        print("prune: ", swc1.shape[0])

    swc1 = resample_swc_(swc1)
    return swc1



