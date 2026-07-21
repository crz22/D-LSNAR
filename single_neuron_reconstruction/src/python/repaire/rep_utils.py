import numpy as np
from collections import deque

def coord_transformation (swc,swc_name, mode='to_255'):
    xyz = swc_name.split('_')
    x_cord = float(xyz[0].replace('x', ''))
    y_cord = float(xyz[1].replace('y', ''))
    z_cord = float(xyz[2].replace('z', ''))
    # print(swc_name)
    if mode == 'to_255':
        if np.any(swc[:, 2:5] >= 256):
            swc[:, 2] = swc[:, 2] - x_cord
            swc[:, 3] = swc[:, 3] - y_cord
            swc[:, 4] = swc[:, 4] - z_cord
    elif mode == 'to_origin':
        if np.all(swc[:, 2:5] <= 256):
            swc[:, 2] = swc[:, 2] + x_cord
            swc[:, 3] = swc[:, 3] + y_cord
            swc[:, 4] = swc[:, 4] + z_cord
    return swc

# ------------------------------------------------------------------ #
#  generate_sphere with cache
_sphere_cache = {}

def generate_sphere(Ma, Mp):
    key = (Ma, Mp)
    if key in _sphere_cache:
        return _sphere_cache[key]

    m1 = np.arange(1, Ma + 1, 1).reshape(-1, Ma)
    m2 = np.arange(1, Mp + 1, 1).reshape(-1, Mp)
    alpha = 2 * np.pi * m1 / Ma
    phi = -(np.arccos(2 * m2 / (Mp + 1) - 1) - np.pi)

    xm = (np.cos(alpha).reshape(Ma, 1)) * np.sin(phi)
    ym = (np.sin(alpha).reshape(Ma, 1)) * np.sin(phi)
    zm = np.cos(phi)
    zm = np.tile(zm, (Mp, 1))

    sphere_core = np.ascontiguousarray(
        np.concatenate([xm.reshape(-1, 1), ym.reshape(-1, 1), zm.reshape(-1, 1)], axis=1),
        dtype=np.float32
    )
    _sphere_cache[key] = sphere_core
    return sphere_core

def Spherical_Patches_Extraction(img2, position, SP_N, SP_core, SP_step=1):
    x = position[0]
    y = position[1]
    z = position[2]
    radius = 1
    j = np.arange(radius, SP_N * SP_step + radius, SP_step).reshape(-1, SP_N)
    ray_x = x + (SP_core[:, 0].reshape(-1, 1)) * j
    ray_y = y + (SP_core[:, 1].reshape(-1, 1)) * j
    ray_z = z + (SP_core[:, 2].reshape(-1, 1)) * j

    bmax = int(img2.shape[0]-1)
    Rray_x = np.clip(np.rint(ray_x).astype(int), 0, bmax)
    Rray_y = np.clip(np.rint(ray_y).astype(int), 0, bmax)
    Rray_z = np.clip(np.rint(ray_z).astype(int), 0, bmax)

    Spherical_patch_temp = img2[Rray_z, Rray_x, Rray_y]
    Spherical_patch = Spherical_patch_temp[:, 1:SP_N]

    SP = np.asarray(Spherical_patch)
    return SP

def create_SP_feature(swc, image, Ma=16, Mp=16, SP_N=10, SP_step=1):
    SP_core = generate_sphere(Ma, Mp)
    image_norm = image / image.max()
    SP_featuer_list = []
    for i in range(swc.shape[0]):
        xyz = swc[i, 2:5]
        yxz = xyz[[1, 0, 2]]
        Spherical_patch = Spherical_Patches_Extraction(image_norm, yxz, SP_N, SP_core, SP_step)
        SP_featuer = Spherical_patch.reshape([Ma, Mp, SP_N - 1]).transpose([2, 0, 1])
        SP_featuer = SP_featuer/(SP_featuer.max()+1e-5)
        SP_featuer_list.append(SP_featuer)
    SP_featuer_list = np.array(SP_featuer_list).astype(np.float32)
    return SP_featuer_list

# ------------------------------------------------------------------ #
def generate_Ntree(swc):
    n = swc.shape[0]

    # 预建 id -> row 映射，消除 np.where 循环
    id_to_row = {int(swc[i, 0]): i for i in range(n)}

    # 预计算每个节点的子节点列表
    children = {i: [] for i in range(n)}
    for i in range(n):
        pid_val = int(swc[i, -1])
        if pid_val != -1:
            p_row = id_to_row.get(pid_val, None)
            if p_row is not None:
                children[p_row].append(i)

    Node_list = []
    for i in range(n):
        node = Node1(idx=i, position=swc[i, 2:5], radius=swc[i, 5], node_type=swc[i, 1])

        # 父节点
        pid_val = int(swc[i, -1])
        if pid_val != -1:
            p_row = id_to_row.get(pid_val, None)
            if p_row is not None:
                node.nbr.append(p_row)

        # 子节点
        node.nbr.extend(children[i])
        Node_list.append(node)

    return Node_list

def bulid_swc_feat(subtree):
    subtree_len = len(subtree)
    adj_raw = np.zeros([subtree_len, subtree_len], dtype=np.float32)
    swc_feat = np.zeros([subtree_len, 3], dtype=np.float32)
    idx_list = [subtree[i].idx for i in range(subtree_len)]
    idx_to_local = {idx: j for j, idx in enumerate(idx_list)}

    for i in range(subtree_len):
        #swc_feat[i] = subtree[i].position / 255.0
        adj_raw[i, i] = 1
        for nbr in subtree[i].nbr:
            j = idx_to_local.get(nbr, None)
            if j is None:
                continue
            adj_raw[j, i] = 1
            adj_raw[i, j] = 1

    D = np.power(np.sum(adj_raw, axis=-1), -0.5)
    D[np.isinf(D)] = 0.0
    D = np.diag(D)
    A = adj_raw.dot(D).T.dot(D)

    swc_feat -= swc_feat.min(axis=0)
    swc_feat_range = swc_feat.max(axis=0) - swc_feat.min(axis=0) + 1e-4
    swc_feat /= swc_feat_range

    return [swc_feat, A, idx_list]

def bulid_subgraphs(swc, deeplayer=4):
    Tree = generate_Ntree(swc)
    subtree_list = []

    for i in range(len(Tree)):
        cur_stree = []
        cur_set = set()
        cur_node_list = [i]
        next_node_list = []

        for j in range(deeplayer):
            while cur_node_list:
                cur_node = cur_node_list.pop()
                cur_stree.append(cur_node)
                cur_set.add(cur_node)
                for nbr_node_id in Tree[cur_node].nbr:
                    if nbr_node_id not in cur_set and swc[nbr_node_id][1] != -1:
                        next_node_list.append(nbr_node_id)
            cur_node_list = next_node_list.copy()
            next_node_list.clear()

        if len(cur_stree) > 16:
            cur_stree = cur_stree[:16]

        cur_stree1 = [Tree[idx] for idx in cur_stree]
        subtree_list.append(bulid_swc_feat(cur_stree1))

    return subtree_list

# ------------------------------------------------------------------ #
#  resample_swc_
def resample_swc_(swc):
    swc = swc.copy()

    if swc.shape[1] < 8:
        swc = np.concatenate([swc, np.zeros((swc.shape[0], 1))], axis=1)
    else:
        swc[:, 7] = 0

    old_ids = swc[:, 0].copy().astype(np.int64)
    old_pids = swc[:, 6].copy().astype(np.int64)

    # 预建旧 id -> 新 row (1-based) 映射
    old_id_to_new = {int(old_ids[i]): i + 1 for i in range(swc.shape[0])}

    for i in range(swc.shape[0]):
        swc[i, 0] = i + 1

        pid_old = int(old_pids[i])
        if pid_old == -1:
            swc[i, 6] = -1
        else:
            new_pid = old_id_to_new.get(pid_old, None)
            swc[i, 6] = new_pid if new_pid is not None else -1

    return swc[:, :7]

# ------------------------------------------------------------------ #
def get_Adjacency_matrix(swc):
    n = swc.shape[0]
    swc_adj = np.zeros((n, n), dtype=np.float32)

    # id -> row 映射
    id_to_row = {int(swc[i, 0]): i for i in range(n)}
    par_ids = swc[:, -1].astype(np.int32)

    for i in range(n):
        swc_adj[i, i] = 1
        par_row = id_to_row.get(int(par_ids[i]), None)
        if par_row is not None:
            swc_adj[i, par_row] = 1
            swc_adj[par_row, i] = 1

    D = np.power(np.sum(swc_adj, axis=-1), -0.5)
    D[np.isinf(D)] = 0
    D = np.diag(D)
    adj_org = swc_adj.copy()
    A = swc_adj.dot(D).T.dot(D)
    return adj_org, A

# ------------------------------------------------------------------ #
def calculate_angle_cos(A, B, C):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    AB = B - A
    AC = C - A
    mag_AB = np.linalg.norm(AB)
    mag_AC = np.linalg.norm(AC)
    if mag_AB == 0 or mag_AC == 0:
        return 0
    cos_theta = np.dot(AB, AC) / (mag_AB * mag_AC)
    return float(np.clip(cos_theta, -1.0, 1.0))

def cal_ang_weight(idx1, idx2, swc):
    pointA = swc[idx1, 2:5]
    pointB = swc[idx2, 2:5]
    parent_A = np.where(swc[:,0] == swc[idx1, 6])[0]
    child_A = np.where(swc[:,6] == swc[idx1, 0])[0]
    cos1_1 = 0
    cos1_2 = 0
    if len(parent_A) == 1 and parent_A[0] != -1:
        pointC = swc[parent_A[0], 2:5]
        cos1_1 = (1 - calculate_angle_cos(pointA, pointB, pointC)) / 2
        # print('cos1_1 = {}'.format(cos1_1))
    if len(child_A) == 1:
        pointC = swc[child_A[0], 2:5]
        cos1_2 = (1 - calculate_angle_cos(pointA, pointB, pointC)) / 2
        # print('cos1_2 = {}'.format(cos1_2))

    cos1 = max(cos1_1, cos1_2)
    # print('cos1 = {}'.format(cos1))

    parent_B = np.where(swc[:,0] == swc[idx2, 6])[0]
    child_B = np.where(swc[:,6] == swc[idx2, 0])[0]
    cos2_1 = 0
    cos2_2 = 0
    if len(parent_B) == 1 and parent_B[0] != -1:
        pointD = swc[parent_B[0], 2:5]
        cos2_1 = (1 - calculate_angle_cos(pointB, pointA, pointD)) / 2
        # print('cos2_1 = {}'.format(cos2_1))
    if len(child_B) == 1 or (len(child_B) == 2 and len(parent_B) == 0):
        pointD = swc[child_B[0], 2:5]
        cos2_2 = (1 - calculate_angle_cos(pointB, pointA, pointD)) / 2
        # print('cos2_2 = {}'.format(cos2_2))
    cos2 = max(cos2_1, cos2_2)
    return 2 * cos1 * cos2 / (cos1 + cos2+1e-5)

def Joint_judgment(predict_adj, raw_adj, swc, Max_distance=20, lamd_dist=0.5,t=0.4,bmin = 12, bmax=244):
    repaired_adj = raw_adj.copy().astype(int)
    predict_adj1 = predict_adj.copy()

    #Select all termination points as candidate nodes to be connected
    candidate_nodes = np.where(np.sum(repaired_adj, axis=1) == 2)[0]  # edge=2: itself + 1 other nodes
    # print("candidate_connect_nodes1:",candidate_nodes.shape)
    # Ignore nodes on the boundary

    if len(candidate_nodes) > 0:
        nodes_in_range = np.all((swc[candidate_nodes, 2:5] > bmin) & (swc[candidate_nodes, 2:5] < bmax), axis=1)
        candidate_nodes = candidate_nodes[nodes_in_range]
        # print("candidate_connect_nodes2:", candidate_nodes.shape)

    if len(candidate_nodes) == 0:
        print("There are no points that need to be connected.")
        return repaired_adj

    # Used to record nodes that do not meet the connection criteria
    mask = np.ones_like(predict_adj1)

    # Distance and angle constraints
    # Calculate distance weight
    distances = np.linalg.norm((swc[:,2:5]-swc[candidate_nodes,2:5][:, np.newaxis, :])* (1, 1, 2),axis=2)
    dis_w = 1 / (1 + np.exp(distances / Max_distance / lamd_dist - 1))

    # Nodes that are greater than the Max_distance do not have a connection relationship
    mask[candidate_nodes] = (distances <= Max_distance)
    predict_adj1 = predict_adj1 * mask
    # print("dis_w: ",dis_w.shape)

    # Angle weight
    ang_w = np.zeros_like(predict_adj1)
    id_to_row = {int(swc[i, 0]): i for i in range(swc.shape[0])}

    for idx_x in candidate_nodes:
        idx_y_list = np.where(predict_adj1[idx_x] > 0)[0]
        for idx_y in idx_y_list:
            # Remove nodes that has already connected candidate nodes
            if find_link(idx_x, idx_y, swc, id_to_row=id_to_row):
                mask[idx_x, idx_y] = 0
            else:
                # Calculate angle weight
                ang_w[idx_x, idx_y] = cal_ang_weight(idx_x, idx_y, swc)


    # Only consider terminal to terminal connections
    mask_terminal_node = np.zeros_like(predict_adj1)
    mask_terminal_node[:, candidate_nodes] = 1
    mask = mask*mask_terminal_node

    overall_score = np.zeros_like(predict_adj1)
    l_pre,l_dis,l_ang = 0.4,0.3,0.3
    overall_score[candidate_nodes] = (predict_adj[candidate_nodes]*l_pre + dis_w*l_dis +ang_w[candidate_nodes]*l_ang) \
                                        * mask[candidate_nodes, :]

    # Select the k nodes with the highest scores as candidate connection nodes
    k = 3
    idx_topk = np.argsort(overall_score[candidate_nodes])[:, -k:][:, ::-1]
    values_topk = np.take_along_axis(overall_score[candidate_nodes], idx_topk, axis=1)
    # print('idx_topk: ',idx_topk)
    # print('values_topk: ',values_topk)

    # Connect from the node with the highest score
    remain_nodes = list(candidate_nodes.copy())
    while values_topk.max() > 0:
        idx_x, idx_y = np.unravel_index(np.argmax(values_topk), values_topk.shape)
        # print(idx_x,idx_y)
        node_x = candidate_nodes[idx_x]
        node_y = idx_topk[idx_x, idx_y]
        # print('node: ', node_x, node_y)

        # If the score changes, update it to values_topk for the next selection
        if overall_score[node_x, node_y] != values_topk[idx_x, idx_y]:
            # print("new: ",values_topk[idx_x, idx_y],overall_score[node_x,node_y])
            values_topk[idx_x, idx_y] = overall_score[node_x, node_y]
            continue

        node_value = overall_score[node_x, node_y]

        if repaired_adj[node_x].sum() <= 2:
            # The score is less than the t, and all remaining points do not meet the connection conditions
            if node_value <= t:
                # print('connect pre < th')
                break
            # The selected node_y already has 3 edges
            if repaired_adj[node_y].sum() >= 4:
                # print(node_y, ': eage > 4')
                overall_score[node_x, node_y] = 0
                values_topk[idx_x, idx_y] = 0
                continue

            # print("connet: ", node_x, node_y)
            #connect two node
            repaired_adj[node_x, node_y] = 1
            repaired_adj[node_y, node_x] = 1

            remain_nodes.remove(node_x)
            overall_score[node_x] = 0
            values_topk[idx_x] = 0

            overall_score[:, node_x] /= 2  # defult: 2
            overall_score[:, node_y] /= 2  # defult: 2

            if node_y in remain_nodes:
                remain_nodes.remove(node_y)
                overall_score[node_y] = 0
                values_topk[np.where(candidate_nodes == node_y)[0][0]] = 0

    return repaired_adj

def find_link(point1, point2, swc, id_to_row=None):
    if point1 == point2:
        return True

    # 允许外部传入 id_to_row 缓存，避免重复构建
    if id_to_row is None:
        id_to_row = {int(swc[i, 0]): i for i in range(swc.shape[0])}

    def ancestors(start):
        visited = set()
        cur = start
        while True:
            pid_val = int(swc[cur, 6])
            if pid_val == -1:
                break
            p_row = id_to_row.get(pid_val, None)
            if p_row is None or p_row in visited:
                break
            visited.add(p_row)
            cur = p_row
        return visited

    anc1 = ancestors(point1)
    anc1.add(point1)

    if point2 in anc1:
        return True

    # 从 point2 往上找，看有没有 anc1 里的
    cur = point2
    while True:
        pid_val = int(swc[cur, 6])
        if pid_val == -1:
            break
        p_row = id_to_row.get(pid_val, None)
        if p_row is None:
            break
        if p_row in anc1:
            return True
        cur = p_row

    return False

# ---------------------------------------------------------------------- #
class Node(object):
    def __init__(self, position, radius, node_type=3):
        self.position = position
        self.radius = radius
        self.node_type = node_type
        self.nbr = []

class Node1(object):
    def __init__(self, idx, position, radius, node_type=3):
        self.position = position
        self.radius = radius
        self.node_type = node_type
        self.nbr = []
        self.idx = idx

def adj2swc(swc, recover_adj):
    np.fill_diagonal(recover_adj, 0)
    connect_idx = np.where(recover_adj == 1)

    nodelist = []
    for i in range(swc.shape[0]):
        node = Node(position=swc[i, 2:5], radius=swc[i, 5], node_type=swc[i, 1])
        index = np.where(connect_idx[0] == i)[0]
        for nbr_id in index:
            node.nbr.append(connect_idx[1][nbr_id])
        nodelist.append(node)

    neuron_tree = compute_trees(nodelist)
    out_swc = nodelist2swc(neuron_tree)
    return out_swc

def get_undiscover(dist):
    for i in range(dist.shape[0]):
        if dist[i] == 100000:
            return i
    return -1

def compute_trees(nodelist, del_node=None):
    if del_node is None:
        del_node = []

    node_num = len(nodelist)
    treecnt = 0
    del_node_set = set(del_node)

    n_tree = []
    dist = np.full(node_num, 100000, dtype=np.int32)
    nmap = np.full(node_num, -1, dtype=np.int32)
    parent = np.full(node_num, -1, dtype=np.int32)

    def _bfs(seeds, use_deque=True):
        q = deque(seeds)
        while q:
            curr = q.popleft() if use_deque else q.pop()
            if curr in del_node_set:
                continue
            n = Node(
                nodelist[curr].position,
                radius=nodelist[curr].radius,
                node_type=treecnt + 2
            )
            if parent[curr] >= 0:
                n.nbr.append(nmap[parent[curr]])
            n_tree.append(n)
            nmap[curr] = len(n_tree)
            for adj in nodelist[curr].nbr:
                if dist[adj] == 100000:
                    dist[adj] = dist[curr] + 1
                    parent[adj] = curr
                    q.append(adj)

    # 从 soma 节点出发（如果 node_type==1）
    if node_num > 0 and nodelist[0].node_type == 1:
        dist[0] = 0
        _bfs([0], use_deque=True)

    # 处理剩余未连接分量
    unjoined = np.where(dist == 100000)[0]
    while len(unjoined) > 0:
        treecnt += 1
        seed = int(unjoined[0])
        dist[seed] = 0
        _bfs([seed], use_deque=False)
        unjoined = np.where(dist == 100000)[0]

    return n_tree

def nodelist2swc(tree):
    rows = []
    cnt = 0
    for node in tree:
        if len(node.nbr) == 0:
            cnt += 1
            rows.append([cnt, node.node_type,
                         node.position[0], node.position[1], node.position[2],
                         node.radius, -1])
        else:
            for pid in node.nbr:
                cnt += 1
                rows.append([cnt, node.node_type,
                              node.position[0], node.position[1], node.position[2],
                              node.radius, int(np.squeeze(pid))])

    if rows:
        return np.array(rows, dtype=np.float64)
    else:
        return np.zeros((0, 7), dtype=np.float64)







