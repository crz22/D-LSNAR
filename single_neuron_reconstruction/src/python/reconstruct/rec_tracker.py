import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from reconstruct.rec_utils import *

class Tracker():
    def __init__(self,config,image,soma_mask,seed_points,model,p_size,device):
        #load sample and model
        self.image = image
        self.soma_mask = soma_mask
        self.seed_points = seed_points
        self.model = model
        self.device = device

        self.img_shape = np.array(image.shape)
        ### boudary of image
        self.p_size = p_size
        self.bmin = [self.p_size, self.p_size, self.p_size]
        self.bmax = np.array([self.img_shape[1], self.img_shape[2], self.img_shape[0]]) - self.p_size-1

        # set param
        self.SPE_NUM = config['SPE_NUM']
        self.SPE_STEP = config['SPE_STEP']
        self.Ma = config['Ma']
        self.Mp = config['Mp']
        self.max_branch_lenth = config['max_branch_lenth']
        self.track_step = config['track_step']
        self.Angle_T = config['Angle_T']
        self.lamd = config['lamd']
        self.show_progress = config.get('rec_show_progress', True)

        self.SPE_core = generate_sphere(self.Ma, self.Mp)  #//
        self.SPE_core_label = generate_sphere(32, 32)      #//
        self.cos_angle_th = np.cos(self.Angle_T)

        # label the traced area
        self.traced_area = np.zeros_like(image,dtype=np.int64)

        self.nodelist = []
        self.overlap_nodelist = []
        self.cur_node_index = 0
        self.cur_branch_index = 0

        if self.seed_points is not None and self.seed_points.shape[0] > 0:
            self.seed_coords = self.seed_points[:, :3]
            self.seed_scores = self.seed_points[:, 3]
            self.seed_tree = cKDTree(self.seed_coords)
        else:
            self.seed_coords = np.empty((0, 3), dtype=np.float32)
            self.seed_scores = np.empty((0,), dtype=np.float32)
            self.seed_tree = None

    def tracker(self):
        seed_num = self.seed_points.shape[0]
        print("[tracker] seed_num: ",seed_num)

        seed_iter = range(seed_num)
        if self.show_progress:
            from tqdm import tqdm
            seed_iter = tqdm(seed_iter)

        for i in seed_iter:
            cur_node_coord = self.seed_points[i,:3] #[x,y,z]
            cur_node_coord = np.round(cur_node_coord).astype(int)

            # check if the current point has been tracked
            if self.traced_area[cur_node_coord[2],cur_node_coord[0],cur_node_coord[1]]>0:
                continue
            # Check if the point is within the boundary
            if out_boundary(cur_node_coord,self.bmin,self.bmax):
                # print("out_boundary: ",cur_node_coord)
                continue

            # Use spe_dnr to predict direction and radii
            predict_result = self.model_predict(cur_node_coord)
            if predict_result is None:
                continue

            pre_dir,pre_rad = predict_result

            # determine two initial direction
            # first direction
            max_id1 = int(np.argmax(pre_dir))
            direction1 = self.SPE_core_label[max_id1, :]

            # second direction
            cos_angle = self.SPE_core_label @ direction1
            pre_dir2 = pre_dir.copy()
            pre_dir2[cos_angle >= 0] = 0
            max_id2 = int(np.argmax(pre_dir2))
            direction2 = self.SPE_core_label[max_id2, :]
            # print("d1: ",direction1,confidence1)
            # print("d2: ",direction2,confidence2)

            self.cur_branch_index += 1
            self.cur_branch_nodes = []

            # Determine if cur nodes has reached the soma area
            #reached trarget neuron soma
            if self.soma_mask[cur_node_coord[2],cur_node_coord[0],cur_node_coord[1]] == 1:
                cur_node = Node(cur_node_coord,self.cur_branch_index,pre_rad,node_type=1)
                self.cur_node_index += 1
                self.nodelist.append(cur_node)
                # continue

            #reached other neuron soma
            elif self.soma_mask[cur_node_coord[2],cur_node_coord[0],cur_node_coord[1]] == -1:
                cur_node = Node(cur_node_coord, self.cur_branch_index, pre_rad, node_type=-1)
                self.cur_node_index += 1
                self.nodelist.append(cur_node)
                continue

            # add cur node in nodelist
            else:
                cur_node = Node(cur_node_coord, self.cur_branch_index, pre_rad)
                self.cur_node_index += 1
                self.nodelist.append(cur_node)

            self.root_index = self.cur_node_index
            self.cur_branch_nodes.append([cur_node_coord,pre_rad])

            # trace towards direction1
            track_neg = False
            self._Track_Pos(cur_node_coord, direction1, track_neg, pre_rad)
            # print("d1 lenth: ", self.cur_node_index)

            # trace towards direction2
            track_neg = True
            self._Track_Pos(cur_node_coord, direction2, track_neg, pre_rad)
            # print("d2 lenth: ", self.cur_node_index)

            # label the traced branches
            self._mask_point(cur_node_coord, pre_rad, self.root_index)
            branch_lenth = self.cur_node_index - self.root_index  # length of new added branches
            # print("len_branch: ",branch_lenth,self.cur_node_index,self.root_index)

            for j in range(branch_lenth):
                cur_node_coord1 = self.nodelist[self.root_index + j].position
                self._mask_point(cur_node_coord1,self.nodelist[self.root_index + j].radius,self.root_index+j+1)


    def model_predict(self,node_coord):
        SP = Spherical_Patches_Extraction(self.image, node_coord, self.SPE_NUM, self.SPE_core, self.SPE_STEP)
        SP = SP.reshape([1, self.Ma, self.Mp, self.SPE_NUM - 1]).transpose([0, 3, 1, 2])
        # print("SP: ",SP.shape,SP.max(),SP.min(),SP.mean())

        inputs = torch.from_numpy(SP).float().to(self.device)
        predict_dir,predict_dis = self.model(inputs)

        #get node radii and determine whether the tracking has stopped
        predict_rad = torch.relu(predict_dis[:, -1, :, :]).reshape(-1)[0].item() + 1

        predict_stop = int(torch.argmax(predict_dis[:, :2, :, :].reshape(-1)).item())
        if predict_stop == 1:
            return None

        #get the direction of next node
        predict_dir = F.softmax(predict_dir, dim=1).reshape(-1).detach().cpu().numpy()
        # print("predict direction: ", predict_dir.shape, predict_dir.max(), predict_dir.min())
        return predict_dir,predict_rad

    def overlap_with_cur_branch(self,cur_node_rad,cur_step):
        # print(self.cur_branch_nodes)
        node_num = len(self.cur_branch_nodes)

        branch_nodes_coord = np.array([self.cur_branch_nodes[i][0] for i in range(node_num)])
        branch_nodes_rad =  np.array([self.cur_branch_nodes[i][1] for i in range(node_num)])

        # print("branch_node: ",branch_nodes_coord,branch_nodes_rad)
        dist = np.sqrt(np.sum(np.square(cur_node_rad-branch_nodes_coord),axis=1))
        min_dist = np.min(dist)

        dist = dist-branch_nodes_rad-cur_step
        if np.sum(dist<=0)>3 or min_dist == 0:
            return True
        else:
            return False

    def connect_nodes(self,track_neg):
        if track_neg == False:
            self.nodelist[self.cur_node_index - 2].nbr.append(self.cur_node_index - 1)
            self.nodelist[self.cur_node_index - 1].nbr.append(self.cur_node_index - 2)
        else:
            self.nodelist[self.root_index - 1].nbr.append(self.cur_node_index - 1)
            self.nodelist[self.cur_node_index - 1].nbr.append(self.root_index - 1)
        return

    def _Track_Pos(self,cur_node_coord, cur_direction, track_neg, cur_node_rad):
        cur_branch_lenth = 0
        # print(self.cur_node_index,cur_node_coord,cur_direction,cur_node_rad,self.track_step)
        cur_step = max(cur_node_rad, self.track_step)
        next_node_coord = cur_node_coord + cur_direction * cur_step

        while cur_branch_lenth < self.max_branch_lenth:
            cur_branch_lenth += 1
            cur_node_coord = next_node_coord.copy()
            # print(cur_node_coord, self.cur_node_index)

            if out_boundary(cur_node_coord,self.bmin,self.bmax):
                # print('reached boundary', cur_node_coord, self.cur_node_index)
                break

            # Determine whether the current node has reached the tracked area
            ### The current node has reached the area already tracked by the current branch
            if self.overlap_with_cur_branch(cur_node_coord,cur_step):
                # print("current node overlap with current branch ",cur_node_coord, self.cur_node_index)
                break

            ### The current node has reached the area already tracked by the other branch
            cur_node_coord_int = np.round(cur_node_coord).astype(int)
            if self.traced_area[cur_node_coord_int[2],cur_node_coord_int[0],cur_node_coord_int[1]]>0:
                # print('Meet Traced Region',self.cur_node_index,cur_node_coord)
                self.overlap_nodelist.append(self.cur_node_index)
                cur_node = Node(cur_node_coord,self.cur_branch_index,radius=0)
                self.cur_node_index += 1
                self.nodelist.append(cur_node)
                self.connect_nodes(track_neg)
                break

            # Use spe_dnr to predict direction and radii
            predict_result = self.model_predict(cur_node_coord)
            if predict_result is None:
                break

            pre_dir, cur_node_rad = predict_result

            # Determine if cur nodes has reached the soma area
            # reached trarget neuron soma
            if self.soma_mask[cur_node_coord_int[2], cur_node_coord_int[0], cur_node_coord_int[1]] == 1:
                cur_node = Node(cur_node_coord, self.cur_branch_index, cur_node_rad, node_type=1)
                self.cur_node_index += 1
                self.nodelist.append(cur_node)
                self.connect_nodes(track_neg)
                break
            # reached other neuron soma
            elif self.soma_mask[cur_node_coord_int[2], cur_node_coord_int[0], cur_node_coord_int[1]] == -1:
                cur_node = Node(cur_node_coord, self.cur_branch_index, cur_node_rad, node_type=-1)
                self.cur_node_index += 1
                self.nodelist.append(cur_node)
                self.connect_nodes(track_neg)
                break

            # determine next direction
            cos_angle = self.SPE_core_label @ cur_direction
            pre_dir_masked = pre_dir.copy()
            pre_dir_masked[cos_angle < self.cos_angle_th] = 0

            max_id = int(np.argmax(pre_dir_masked))
            pre_direction = self.SPE_core_label[max_id]
            confidence = float(pre_dir_masked[max_id])
            # print("d: ",self.SPE_core_label[max_id],pre_dir[max_id])

            cur_node = Node(cur_node_coord, self.cur_branch_index, cur_node_rad)
            self.cur_node_index += 1
            self.nodelist.append(cur_node)
            self.connect_nodes(track_neg)
            track_neg = False
            self.cur_branch_nodes.append([cur_node_coord,cur_node_rad])

            #joint decision next node
            spe_dnr_flag = True

            if self.seed_tree is not None and self.seed_coords.shape[0] > 0:
                candidate_seed = self.seed_tree.query_ball_point(cur_node_coord, r=float(self.lamd * cur_node_rad))

                if len(candidate_seed) > 0:
                    candidate_seed = np.asarray(candidate_seed, dtype=np.int32)
                    candidate_seed_coord = self.seed_coords[candidate_seed]

                    diff = candidate_seed_coord - cur_node_coord
                    dist_curnode2seeds = np.linalg.norm(diff, axis=1)

                    valid_mask = dist_curnode2seeds >= self.track_step
                    if np.any(valid_mask):
                        candidate_seed = candidate_seed[valid_mask]
                        candidate_seed_coord = candidate_seed_coord[valid_mask]
                        diff = diff[valid_mask]
                        dist_curnode2seeds = dist_curnode2seeds[valid_mask]

                        seed_dir = diff / dist_curnode2seeds.reshape(-1, 1)
                        seed_dir_cos = seed_dir @ cur_direction
                        seed_dir_cos = np.clip(seed_dir_cos, -1.0, 1.0)

                        if seed_dir_cos.max() >= self.cos_angle_th:
                            vid = int(np.argmax(seed_dir_cos))
                            if confidence * 2 < self.seed_scores[candidate_seed[vid]]:
                                spe_dnr_flag = False
                                next_node_coord = candidate_seed_coord[vid]
                                cur_direction = seed_dir[vid]
                                cur_step = dist_curnode2seeds[vid]

            if spe_dnr_flag:
                cur_step = max(cur_node_rad,self.track_step)
                next_node_coord = cur_node_coord + pre_direction * cur_step
                cur_direction = pre_direction

            # print("next_track: ",self.cur_node_index,spe_dnr_flag, cur_direction, next_node_coord, cur_step)

    def _mask_point(self, node_coord, radii, index=0):
        node_coord_int = np.rint(node_coord).astype(int)
        mask_r = int(np.rint(max(radii, self.track_step)))

        x0 = max(node_coord_int[0] - mask_r, self.bmin[0])
        x1 = min(node_coord_int[0] + mask_r + 1, self.bmax[0])
        y0 = max(node_coord_int[1] - mask_r, self.bmin[1])
        y1 = min(node_coord_int[1] + mask_r + 1, self.bmax[1])
        z0 = max(node_coord_int[2] - mask_r, self.bmin[2])
        z1 = min(node_coord_int[2] + mask_r + 1, self.bmax[2])

        if index == 0:
            self.traced_area[z0:z1, x0:x1, y0:y1] = self.cur_node_index
        else:
            self.traced_area[z0:z1, x0:x1, y0:y1] = index

    def connect_overlap_node(self):
        node_coords = np.asarray([self.nodelist[i].position for i in range(self.cur_node_index)], dtype=np.float32)
        overlap_set = set(self.overlap_nodelist)

        for i in self.overlap_nodelist:
            # print("overlap_node: ", i)
            overlap_node = self.nodelist[i]

            #caculate dist of overlap node to other node
            dist = np.sqrt(np.sum(np.square(overlap_node.position-node_coords),axis=1))
            dist[i] = 1000

            while True:
                min_index = int(np.argmin(dist))
                min_dist = float(dist[min_index])

                if min_dist > 4:
                    break

                if min_index in overlap_set:
                    dist[min_index] = 1000.0
                    continue

                if overlap_node.branch_index == self.nodelist[min_index].branch_index:
                    dist[min_index] = 1000.0
                    continue

                if len(self.nodelist[min_index].nbr) >= 3:
                    dist[min_index] = 1000.0
                    continue

                for onbr_index in overlap_node.nbr:
                    self.nodelist[min_index].nbr.append(onbr_index)
                    self.nodelist[onbr_index].nbr.append(min_index)
                    self.nodelist[onbr_index].nbr.remove(i)

                self.nodelist[i].nbr.clear()

                branch_index_min = min(overlap_node.branch_index, self.nodelist[min_index].branch_index)
                branch_index_max = max(overlap_node.branch_index, self.nodelist[min_index].branch_index)

                for j in range(len(self.nodelist)):
                    if self.nodelist[j].branch_index == branch_index_max:
                        self.nodelist[j].branch_index = branch_index_min
                break