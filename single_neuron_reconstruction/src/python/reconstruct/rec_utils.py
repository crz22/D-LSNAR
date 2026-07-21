import numpy as np
from scipy.ndimage import distance_transform_edt
from collections import deque

class Node(object):
   def __init__(self, position, branch_index=0, radius=0, node_type=3):
       self.position = position
       self.branch_index = branch_index
       self.radius = radius
       self.node_type = node_type
       self.nbr = []

def out_boundary(points,bmin,bmax):
    return bool(np.any((points < bmin) | (points >= bmax)))

# ------------------------------------------------------------------------#
def Spherical_Patches_Extraction(img2, position, SP_N, SP_core, SP_step=1):
    x = position[0]
    y = position[1]
    z = position[2]
    radius = 1
    j = np.arange(radius, SP_N*SP_step + radius, SP_step).reshape(-1, SP_N)

    ray_x = x + (SP_core[:, 0].reshape(-1, 1)) * j
    ray_y = y + (SP_core[:, 1].reshape(-1, 1)) * j
    ray_z = z + (SP_core[:, 2].reshape(-1, 1)) * j

    Rray_x = np.rint(ray_x).astype(int)
    Rray_y = np.rint(ray_y).astype(int)
    Rray_z = np.rint(ray_z).astype(int)

    Spherical_patch_temp = img2[Rray_z, Rray_x, Rray_y]
    Spherical_patch = Spherical_patch_temp[:, 1:SP_N]

    SP = np.asarray(Spherical_patch)
    SP = (SP-SP.min()) / (SP.max()-SP.min()+1e-5)
    return SP

_sphere_cache = {}
def generate_sphere(Ma, Mp):
    key = (Ma, Mp)
    if key in _sphere_cache:
        return _sphere_cache[key].copy()

    m1 = np.arange(1, Ma + 1, 1).reshape(-1, Ma)
    m2 = np.arange(1, Mp + 1, 1).reshape(-1, Mp)
    alpha = 2 * np.pi * m1 / Ma
    phi = -(np.arccos(2 * m2 / (Mp + 1) - 1) - np.pi)

    xm = (np.cos(alpha).reshape(Ma, 1)) * np.sin(phi)
    ym = (np.sin(alpha).reshape(Ma, 1)) * np.sin(phi)
    zm = np.cos(phi)
    zm = np.tile(zm, (Mp, 1))

    sphere_core = np.ascontiguousarray(
        np.concatenate(
            [xm.reshape(-1, 1), ym.reshape(-1, 1), zm.reshape(-1, 1)],
            axis=1
        ),
        dtype=np.float32
    )
    _sphere_cache[key] = sphere_core
    return sphere_core.copy()

def mask_sphere(central_point,rad,bmin,bmax):
    rad_int = int(round(rad))
    cx, cy, cz = central_point
    rad_sq = rad * rad

    # 直接生成裁剪后的坐标范围
    x_range = np.arange(max(cx - rad_int, bmin[0]), min(cx + rad_int + 1, bmax[0]))
    y_range = np.arange(max(cy - rad_int, bmin[1]), min(cy + rad_int + 1, bmax[1]))
    z_range = np.arange(max(cz - rad_int, bmin[2]), min(cz + rad_int + 1, bmax[2]))

    if len(x_range) == 0 or len(y_range) == 0 or len(z_range) == 0:
        return np.empty((0, 3), dtype=int)

    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    # 向量化距离过滤
    mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= rad_sq
    return np.column_stack((X[mask], Y[mask], Z[mask]))

    # rad_int = round(rad)
    # position_x, position_y, position_z = central_point
    # print(rad)
    # x_point = np.linspace(position_x - rad_int, position_x + rad_int, 2 * rad_int + 1)
    # y_point = np.linspace(position_y - rad_int, position_y + rad_int, 2 * rad_int + 1)
    # z_point = np.linspace(position_z - rad_int, position_z + rad_int, 2 * rad_int + 1)
    # X_P,Y_P,Z_P = np.meshgrid(x_point,y_point,z_point)
    # point_jh = np.concatenate((X_P.reshape(-1,1),Y_P.reshape(-1,1),Z_P.reshape(-1,1)),axis=1).astype(int)
    # # print("1: ",point_jh)
    # #delete out of boundary
    # for i in range(len(bmax)):
    #     index_del = np.where((point_jh[:, i] > bmax[i]))
    #     point_jh = np.delete(point_jh, index_del, 0)
    #
    #     index_del = np.where(point_jh[:,i] < bmin[i])
    #     point_jh = np.delete(point_jh, index_del, 0)
    # # print("2: ", point_jh)
    # #delete out of spe point
    # dis = np.linalg.norm((point_jh-central_point),axis=1)
    # index_del = np.where(dis>rad)
    # point_jh = np.delete(point_jh,index_del,0)
    # # print("3: ", point_jh)
    # return point_jh

def mask_cube(central_point,rad,bmin,bmax):
    rad_int = int(round(rad))
    cx, cy, cz = central_point

    x_range = np.arange(max(cx - rad_int, bmin[0]), min(cx + rad_int + 1, bmax[0]))
    y_range = np.arange(max(cy - rad_int, bmin[1]), min(cy + rad_int + 1, bmax[1]))
    z_range = np.arange(max(cz - rad_int, bmin[2]), min(cz + rad_int + 1, bmax[2]))

    if len(x_range) == 0 or len(y_range) == 0 or len(z_range) == 0:
        return np.empty((0, 3), dtype=int)

    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    # # rad_int = round(rad+0.5)
    # rad_int = round(rad)
    # position_x, position_y, position_z = central_point
    # #print(rad)
    # x_point = np.linspace(position_x - rad_int, position_x + rad_int, 2 * rad_int + 1)
    # y_point = np.linspace(position_y - rad_int, position_y + rad_int, 2 * rad_int + 1)
    # z_point = np.linspace(position_z - rad_int, position_z + rad_int, 2 * rad_int + 1)
    # X_P, Y_P, Z_P = np.meshgrid(x_point, y_point, z_point)
    # point_jh = np.concatenate((X_P.reshape(-1, 1), Y_P.reshape(-1, 1), Z_P.reshape(-1, 1)), axis=1).astype(int)
    # #delete out of boundary
    # for i in range(len(bmax)):
    #     index_del = np.where((point_jh[:, i] > bmax[i]))
    #     point_jh = np.delete(point_jh, index_del, 0)
    #
    #     index_del = np.where(point_jh[:,i] < bmin[i])
    #     point_jh = np.delete(point_jh, index_del, 0)
    # # print(point_jh)
    # return point_jh

# --------------------------------------------------------------------------------------- #
def get_soma_node(soma_mask):
    soma_mask[soma_mask < 0] = 0
    distance_transform = distance_transform_edt(soma_mask)
    soma_mask_shape = np.array(soma_mask.shape)
    mask_center = soma_mask_shape//2
    soma_rad = np.max(distance_transform)

    candidate_soma_coord = np.argwhere(distance_transform==soma_rad)
    dist_soma2center = np.sum(np.square(candidate_soma_coord-mask_center),axis=1)
    soma_index = np.argmin(dist_soma2center)
    soma_coord = candidate_soma_coord[soma_index] #[z,x,y]
    soma_coord = np.array([soma_coord[1],soma_coord[2],soma_coord[0]]) #[x,y,z]
    print("soma: ", soma_coord, soma_rad)
    return soma_coord,soma_rad

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

def compute_trees(nodelist, del_node=None, soma_mark=None, distance_transform=None):
    if del_node is None:
        del_node = []

    node_num = len(nodelist)
    treecnt = 0
    q = deque()
    n_tree = []

    del_node_set = set(del_node)

    visited = np.zeros(node_num, dtype=bool)
    nmap = np.full(node_num, -1, dtype=np.int32)  # index in output tree (1-based)
    parent = np.full(node_num, -1, dtype=np.int32)

    if len(del_node_set) > 0:
        valid_del = [i for i in del_node_set if 0 <= i < node_num]
        visited[valid_del] = True

    print('[compute_trees] : start from Soma')
    # step 1: Start BFS from the soma node
    if soma_mark is not None and soma_mark.max() > 0:
        # soma_coord, soma_rad = get_soma_node(soma_mark)
        soma_coord, soma_rad = get_soma_node_approx(soma_mark)

        n = Node(soma_coord,radius=soma_rad,node_type=1)
        n_tree.append(n)

        points_in_soma = []

        for i, node in enumerate(nodelist):
            if visited[i]:
                continue

            if node.node_type == 1:
                points_in_soma.append(i)
                q.append(i)
                visited[i] = True

        points_in_soma_set = set(points_in_soma)
        # BFS
        while q:
            curr = q.popleft()

            cur_node = nodelist[curr]
            n = Node(cur_node.position, radius=cur_node.radius, node_type=treecnt + 2)

            if parent[curr] >= 0:
                n.nbr.append(nmap[parent[curr]])
            elif curr in points_in_soma_set:
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

    # step 2. Filter out branches connected to other soma
    for i, node in enumerate(nodelist):
        if visited[i]:
            continue
        if node.node_type == -1:
            q.append(i)
            visited[i] = True

    num = 0
    while q:
        curr = q.pop()
        cur_node = nodelist[curr]
        num += 1

        for adj in cur_node.nbr:
            if adj < 0 or adj >= node_num:
                continue
            if not visited[adj]:
                visited[adj] = True
                parent[adj] = curr
                q.append(adj)

    print("[compute_trees] points connect with other soma: ", num)

    # step 3. Process other connected components that are not connected to soma
    for seed in range(node_num):
        if visited[seed]:
            continue

        treecnt += 1
        visited[seed] = True
        parent[seed] = -1
        q.append(seed)

        while q:
            curr = q.pop()
            cur_node = nodelist[curr]

            n = Node(
                cur_node.position,
                cur_node.branch_index,
                cur_node.radius,
                treecnt + 2
            )

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

# -------------------------------------------------------------------------------- #
def nodelist2swc(tree):
    rows = []
    cnt_recnodes = 0
    for node in tree:
        if len(node.nbr) == 0:
            cnt_recnodes += 1
            pid = -1
            row = [cnt_recnodes, node.node_type,
                   node.position[1], node.position[0], node.position[2],
                   node.radius, pid]
            rows.append(row)
        else:
            for pid in node.nbr:
                cnt_recnodes += 1
                row = [cnt_recnodes, node.node_type,
                       node.position[1], node.position[0], node.position[2],
                       node.radius, int(pid.squeeze())]
                rows.append(row)
    if rows:
        _data = np.array(rows)
    else:
        _data = np.zeros((0, 7))
    return _data


