import numpy as np
import torch
import tifffile
from repaire.rep_utils import create_SP_feature,bulid_subgraphs,get_Adjacency_matrix

def read_image(tif_path):
    return tifffile.imread(tif_path)

def save_tif(img, path):
    tifffile.imwrite(path, img.astype(np.uint8), compression=None)
    return

def load_swc(filepath):
    # load swc file as a N X 7 numpy array
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                # remove empty units
                if len(cells) != 9:
                    for kk in range(len(cells) - 1, -1, -1):
                        if cells[kk] == '':
                            cells.pop(kk)
                if len(cells) != 7:
                    for i in range(7, len(cells)):
                        # print(i)
                        cells.pop()
                if len(cells) == 7:
                    cells = [float(c) for c in cells]  # transform string to float
                    swc.append(cells[0:7])
    return np.array(swc)

def save_swc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]
    # print(filepath)
    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            # print(swc[i, :])
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)

def generate_data_for_TED(image,swc,is_start=False):
    Ma, Mp, SP_N, SP_step = 16, 16, 10, 0.5
    swc1 = swc.copy()
    # swc_block = change_coord(swc_path)
    # generate Spherical_Patches feature for each node
    spe_feat = create_SP_feature(swc, image, Ma, Mp, SP_N, SP_step)
    spe_feat = (spe_feat-spe_feat.min()) / (spe_feat.max()-spe_feat.min()+1e-5)
    # generate subgraphs
    subtree_feat = bulid_subgraphs(swc)

    data = []
    for subdata in subtree_feat:
        subswc_feat, subadj, idx_list = subdata
        subadj = torch.from_numpy(subadj).float()
        subswc_feat = torch.from_numpy(subswc_feat).float()
        subspe_feat = torch.from_numpy(spe_feat[idx_list]).float()
        # sublab = torch.from_numpy(np.eye(2)[swc_block[idx_list[0], 1].astype(np.int8)]).float()
        data.append([subadj, subswc_feat, subspe_feat])
    return data, swc1

def generate_data_for_TLP(image,swc,is_start=False):
    Ma, Mp, SP_N, SP_step = 16, 16, 10, 1

    feat_xyz = swc[:, 2:5]
    feat_xyz = (feat_xyz-feat_xyz.min())/(feat_xyz.max()-feat_xyz.min()+1)
    feat_xyz = torch.from_numpy(feat_xyz).float()

    feat_sp = create_SP_feature(swc, image, Ma, Mp, SP_N, SP_step)
    feat_sp = torch.from_numpy(feat_sp).float()

    adj_org, adj = get_Adjacency_matrix(swc)
    adj = torch.from_numpy(adj).float()
    return feat_xyz, feat_sp, adj, adj_org