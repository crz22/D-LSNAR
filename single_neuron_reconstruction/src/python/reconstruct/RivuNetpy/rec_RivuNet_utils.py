import numpy as np
from collections import deque
import os

class Node(object):
   def __init__(self, position, branch_index=0, radius=0, node_type=3):
       self.position = position
       self.branch_index = branch_index
       self.radius = radius
       self.node_type = node_type
       self.nbr = []

def swc2Nodelist(swc):
    N = swc.shape[0]
    Nodelist = []

    id2idx = {swc[i, 0]: i for i in range(N)}
    parent_dict = {}

    # build parent->children map
    for i in range(N):
        pid = swc[i, -1]
        if pid not in parent_dict:
            parent_dict[pid] = []
        parent_dict[pid].append(i)

    for i in range(N):
        coord = np.array([swc[i,3],swc[i,2],swc[i,4]])
        node = Node(
            position=coord,
            radius=swc[i, 5],
            node_type=swc[i, 1]
        )

        node_id = swc[i, 0]
        parent_id = swc[i, -1]

        # parent
        if parent_id in id2idx:
            node.nbr.append(id2idx[parent_id])

        # children
        if node_id in parent_dict:
            for c in parent_dict[node_id]:
                if c != i:
                    node.nbr.append(c)

        Nodelist.append(node)

    return Nodelist

def get_soma_node_approx(soma_mask):
    mask = soma_mask > 0

    if not mask.any():
        print("warning: no target soma found")
        return np.array([0, 0, 0], dtype=np.float32), 1.0

    z, x, y = np.where(mask)

    soma_coord = np.array(
        [x.mean(), y.mean(), z.mean()],
        dtype=np.float32
    )

    volume = len(z)

    # 根据球体体积估算半径
    soma_rad = float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))

    print("soma approx:", soma_coord, soma_rad)

    return soma_coord, soma_rad

def compute_trees(nodelist, del_node=None, soma_mark=None):
    node_num = len(nodelist)
    treecnt = 0
    q = deque()
    n_tree = []


    visited = np.zeros(node_num, dtype=bool)
    nmap = np.full(node_num, -1, dtype=np.int32) # index in output tree n1
    parent = np.full(node_num, -1, dtype=np.int32) # parent index in current tree n0

    if del_node is not None:
        valid_del = [i for i in del_node if 0 <= i < node_num]
        visited[valid_del] = True

    print('compute_trees: start from Soma')
    # =========================
    # step 1. 从 soma 节点开始 BFS
    # =========================
    print("[compute tree] soma_mask: ",soma_mark.max(),soma_mark.min())
    PointsInSoma_set = set()
    if soma_mark is not None and soma_mark.max() > 0:
        # soma_coord, soma_rad = get_soma_node(soma_mark)
        soma_coord, soma_rad = get_soma_node_approx(soma_mark)

        n = Node(position=soma_coord, radius = soma_rad, node_type=1)
        n_tree.append(n)

        soma_rad_sq = soma_rad ** 2  # 用平方比较，省去求开方的计算量

        for i, node in enumerate(nodelist):
            if visited[i]:
                continue

            dist_sq = np.sum((node.position - soma_coord) ** 2)
            if dist_sq <= 4*soma_rad_sq:
                PointsInSoma_set.add(i)
                q.append(i)
                visited[i] = True

        # BFS
        while len(q) > 0:
            curr = q.popleft()
            cur_node = nodelist[curr]

            n = Node(position=cur_node.position, radius= cur_node.radius, node_type=treecnt + 2)

            if parent[curr] >= 0:
                n.nbr.append(nmap[parent[curr]])
            elif curr in PointsInSoma_set:
                n.nbr.append(np.array([1]))
                n.node_type = 1

            n_tree.append(n)
            nmap[curr] = len(n_tree)

            for adj in cur_node.nbr:
                if adj < 0 or adj >= node_num:
                    continue

                if not visited[adj]:
                    visited[adj] = True
                    parent[adj] = curr
                    q.append(adj)

    # =========================
    # step 2.过滤掉与其他soma相连的分支
    # =========================
    if soma_mark is not None and soma_mark.min()<0:
        mark_shape = soma_mark.shape
        for i, node in enumerate(nodelist):
            if visited[i]:
                continue

            # node.position 原本是 [y, x, z]
            p_y, p_x, p_z = node.position

            # 四舍五入并转为整型
            pz = int(round(float(p_z)))
            py = int(round(float(p_y)))
            px = int(round(float(p_x)))

            # 加入边界保护，防止坐标溢出 soma_mark 报错
            if (0 <= pz < mark_shape[0] and
                    0 <= py < mark_shape[1] and
                    0 <= px < mark_shape[2]):

                # 注意这里：直接用 pz, py, px 索引，千万不要套方括号 []
                if soma_mark[pz, py, px] == -1:
                    q.append(i)
                    visited[i] = True

        num = 0
        while q:
            curr = q.popleft()
            cur_node = nodelist[curr]
            num += 1

            for adj in cur_node.nbr:
                if 0 <= adj < node_num and not visited[adj]:
                    visited[adj] = True
                    # parent[adj] = curr  # 不需要存 parent，因为这些节点被丢弃
                    q.append(adj)

        print("points connect with other soma: ",num)

    # =========================
    # step 3. 处理未连接到 soma 的其他连通分量
    # =========================
    # print(node_num)
    for seed in range(node_num):
        if visited[seed]:
            continue

        treecnt += 1

        visited[seed] = True
        # dist[seed] = 0
        q.append(seed)

        while q:
            curr = q.popleft()  # 统一使用 BFS
            cur_node = nodelist[curr]

            n = Node(position=cur_node.position, radius=cur_node.radius, node_type=treecnt + 2)

            if parent[curr] >= 0:
                n.nbr.append(nmap[parent[curr]])

            n_tree.append(n)
            nmap[curr] = len(n_tree)

            for adj in cur_node.nbr:
                if adj < 0 or adj >= node_num:
                    continue

                if not visited[adj]:
                    visited[adj] = True
                    parent[adj] = curr
                    q.append(adj)

    return n_tree

##################################################################
def nodelist2swc(tree):
    data = []
    cnt_recnodes = 0

    for node in tree:
        if len(node.nbr) == 0:
            cnt_recnodes += 1
            pid = -1

            new_node = [
                cnt_recnodes,
                node.node_type,
                node.position[1],
                node.position[0],
                node.position[2],
                node.radius,
                pid
            ]

            data.append(new_node)

        else:
            for nbr in node.nbr:
                cnt_recnodes += 1

                pid = np.asarray(nbr).squeeze()

                new_node = [
                    cnt_recnodes,
                    node.node_type,
                    node.position[1],
                    node.position[0],
                    node.position[2],
                    node.radius,
                    pid
                ]

                data.append(new_node)

    if len(data) == 0:
        return np.empty((0, 7), dtype=np.float32)

    return np.asarray(data, dtype=np.float32)

##################################################################
import tifffile as tiff
def tif2imagej_tif(image: np.ndarray, fixed_img_path: str):
    """
    将 3D numpy 数组保存为 RivuNet 要求的 ImageJ hyperstack TIFF
    """
    assert image.ndim == 3, f"期望3D输入 (Z,Y,X)，实际 shape={image.shape}"
    z, y, x = image.shape

    print(f'[tif2imagej] Input shape (Z,Y,X): ({z},{y},{x})')
    # -------------------  Step 1: 归一化到 0~255 -------------------
    img = image.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        raise ValueError(f"Image min==max=={img_min}")
    img = img.astype(np.uint8)

    # # ------------------- Step 2: 构造 ImageJ description -------------------
    imagej_description = (
            "ImageJ=1.53t\n"
            f"images={z}\n"
            "channels=1\n"
            f"slices={z}\n"
            "frames=1\n"
            "hyperstack=true\n"
            "mode=grayscale\n"
            "loop=false\n"
            "finterval=1.0\n"
            "tunit=ms\n"
        )

    # # ------------------- Step 3: save -------------------
    tiff.imwrite(
        fixed_img_path,
        img.astype(np.uint8),
        description=imagej_description,
        resolution=(1.0, 1.0),
        metadata=None
    )
    print(f'[tif2imagej] save fininshed: {fixed_img_path}')

##################################################################
def _swc_to_array(swc) -> np.ndarray:
    """
    把各种格式的 swc 统一转成 np.ndarray shape=(N,7)
    支持: ndarray / 文件路径 / 含 data/_data/array 属性的对象
    """
    if swc is None:
        return np.zeros((0, 7), dtype=np.float64)

    # numpy array
    if isinstance(swc, np.ndarray):
        if swc.ndim == 2 and swc.shape[1] >= 7:
            return swc[:, :7].astype(np.float64)
        return np.zeros((0, 7), dtype=np.float64)

    # 尝试直接转
    try:
        arr = np.asarray(swc, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 7:
            return arr[:, :7]
    except Exception:
        pass

    # 尝试常见属性
    for attr in ("data", "_data", "array", "swc", "nodes"):
        val = getattr(swc, attr, None)
        if val is not None:
            try:
                arr = np.asarray(val, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] >= 7:
                    return arr[:, :7]
            except Exception:
                pass

    raise TypeError(f"无法把 {type(swc)} 转换为 SWC 数组")

def merge_swcs(swc_list):
    """
    将多个 SWC 文件合并为一个 SWC 文件。
    会自动重新编号 node id，并更新 parent id。
    每个神经元的 soma/root 仍然保持 parent=-1。
    """
    arrays = []
    for i, swc in enumerate(swc_list):
        arr = np.array(swc)
        if arr is None or arr.size == 0:
            print(f"[WARN] swc_list[{i}] 为空，跳过")
            continue
        arrays.append(arr)

    if len(arrays) == 0:
        raise ValueError("swc_list 中没有有效的 SWC")

    merged_parts = []
    next_id = 1

    for idx, arr in enumerate(arrays):
        arr = np.asarray(arr, dtype=np.float64)

        # 只保留前7列
        if arr.shape[1] > 7:
            arr = arr[:, :7]

        # 过滤非法行
        valid_mask = np.isfinite(arr).all(axis=1)
        arr = arr[valid_mask]

        if len(arr) == 0:
            print(f"[WARN] 第 {idx} 个 swc 全是非法行，跳过")
            continue

        old_ids = np.rint(arr[:, 0]).astype(int)
        old_types = np.rint(arr[:, 1]).astype(int)
        old_parents = np.rint(arr[:, 6]).astype(int)

        # old id -> new id 映射
        id_map = {}
        for k, old_id in enumerate(old_ids):
            id_map[old_id] = next_id + k

        new_arr = np.zeros((len(arr), 7), dtype=np.float64)

        # 新 id
        new_arr[:, 0] = [id_map[oid] for oid in old_ids]
        # type
        new_arr[:, 1] = old_types
        # x,y,z,r
        new_arr[:, 2:6] = arr[:, 2:6]

        old_id_set = set(old_ids.tolist())

        # parent 重映射
        new_parents = []
        for p in old_parents:
            if p < 0 or p not in old_id_set:
                new_parents.append(-1)
            else:
                new_parents.append(id_map[p])

        new_arr[:, 6] = new_parents

        merged_parts.append(new_arr)
        next_id += len(arr)

    if len(merged_parts) == 0:
        raise ValueError("All SWCs are invalid and cannot be merged")

    merged = np.vstack(merged_parts)
    merged = merged[np.argsort(merged[:, 0])]
    return merged