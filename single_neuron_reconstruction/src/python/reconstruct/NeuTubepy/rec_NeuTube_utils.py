import numpy as np
from collections import deque

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
            if dist_sq <= soma_rad_sq:
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
