import yaml
import os
import numpy as np
from tqdm import tqdm
import torch
from repaire.rep_io import read_image,load_swc,save_swc,generate_data_for_TED,generate_data_for_TLP
from repaire.rep_network import TED_Net,TLP_Net
from repaire.rep_utils import resample_swc_,Joint_judgment,adj2swc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------- model cache ---------------------- #
_TED_model = None
_TED_model_path = None

_TLP_model = None
_TLP_model_path = None


def _get_TED_model(config):
    global _TED_model, _TED_model_path

    ckpt_path = config["TED_checkpoint_path"]
    if _TED_model is None or _TED_model_path != ckpt_path:
        TED_model = TED_Net(3, max_length=16, device=device)
        TED_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        TED_model.load_state_dict(TED_checkpoint)
        TED_model.eval().to(device)

        _TED_model = TED_model
        _TED_model_path = ckpt_path

    return _TED_model


def _get_TLP_model(config):
    global _TLP_model, _TLP_model_path

    ckpt_path = config["TLP_checkpoint_path"]
    if _TLP_model is None or _TLP_model_path != ckpt_path:
        TLP_model = TLP_Net()
        TLP_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        TLP_model.load_state_dict(TLP_checkpoint)
        TLP_model.eval().to(device)

        _TLP_model = TLP_model
        _TLP_model_path = ckpt_path

    return _TLP_model

# ---------------------- --------- ---------------------- #
def remove_erro_nodes(config,image,swc,is_start=False):
    print("remove_erro_nodes")
    #load model
    # TED_model = TED_Net(3,max_length=16,device=device)
    # TED_checkpoint = torch.load(config["TED_checkpoint_path"], map_location=lambda storage, loc: storage, weights_only=False)
    # TED_model.load_state_dict(TED_checkpoint)
    # TED_model.eval().to(device)

    if swc is None or swc.shape[0] == 0:
        return swc

    # load model
    TED_model = _get_TED_model(config)

    #load data
    data,swc1 = generate_data_for_TED(image,swc)

    result = np.zeros(len(data), dtype=np.uint8)

    # precompute child counts, avoid repeated np.where
    node_ids = swc[:, 0].astype(np.int64, copy=False)
    parent_ids = swc[:, -1].astype(np.int64, copy=False)
    uniq_parent, parent_count = np.unique(parent_ids, return_counts=True)
    child_count_dict = dict(zip(uniq_parent.tolist(), parent_count.tolist()))

    # precompute near-soma skip mask
    skip_mask = np.zeros(len(data), dtype=bool)
    if is_start:
        id_to_row = {int(node_ids[i]): i for i in range(len(node_ids))}

        for i in range(len(data)):
            if i == 0 or swc[i, 1] == 1:
                skip_mask[i] = True
                continue

            p_id = int(swc[i, -1])
            near_soma_flag = False

            if p_id != -1:
                for _ in range(10):
                    if p_id == -1:
                        break

                    p_row = id_to_row.get(p_id, None)
                    if p_row is None:
                        # fallback: compatible with old "id-1" assumption
                        p_row = p_id - 1
                        if p_row < 0 or p_row >= swc.shape[0]:
                            break

                    if swc[p_row, 1] == 1:
                        near_soma_flag = True
                        break

                    p_id = int(swc[p_row, -1])

            if near_soma_flag:
                skip_mask[i] = True

    show_progress = config.get("rep_show_progress", False)
    iterator = range(len(data))
    if show_progress:
        iterator = tqdm(iterator)

    # predict erro nodes
    with torch.inference_mode():
        for i in iterator:
            adj, feat_xyz, feat_sp = data[i]

            # set node near soma is True(skip detection, keep node)
            if is_start and skip_mask[i]:
                continue

            # branch point requirement: child num >= 2
            if child_count_dict.get(int(node_ids[i]), 0) < 2:
                continue

            # set sub_graph less 2 node is False
            if len(feat_xyz) <= 2:
                result[i] = 1
                continue

            feat_xyz = feat_xyz.to(device)
            adj = adj.to(device)
            feat_sp = feat_sp.to(device)

            output = TED_model(adj, feat_xyz, feat_sp)
            result[i] = int(torch.argmax(output, dim=1).item())

    # delete the node is erro
    det_idx = np.where(result == 1)[0]

    # remove boundary detections
    bmax= image.shape[0]-12
    if det_idx.size > 0:
        coords = swc1[det_idx, 2:5]
        bound_mask = np.any(coords <= 13, axis=1) | np.any(coords >= bmax, axis=1)
        det_idx = det_idx[~bound_mask]

    swc_remove = np.delete(swc1, det_idx, axis=0)
    swc_remove = resample_swc_(swc_remove)

    return swc_remove

# ---------------------- --------- ---------------------- #
def repair_edges(config,image,swc,is_start=False):
    print("repair_edges")

    if swc is None or swc.shape[0] == 0:
        return swc

    # load model
    TLP_model = _get_TLP_model(config)

    # TLP_model = TLP_Net()
    # TLP_checkpoint = torch.load(config["TLP_checkpoint_path"], map_location=lambda storage, loc: storage, weights_only=False)
    # TLP_model.load_state_dict(TLP_checkpoint)
    # TLP_model.eval().to(device)

    # load data
    feat_xyz, feat_sp, adj, raw_adj = generate_data_for_TLP(image,swc)
    feat_xyz,feat_sp,adj = feat_xyz.to(device),feat_sp.to(device),adj.to(device)
    adj = adj.squeeze(dim=0)

    # predict edges
    with torch.inference_mode():
        output = TLP_model(feat_xyz, adj, feat_sp)
        output = torch.sigmoid(output).detach().cpu().numpy()

    recover_adj = Joint_judgment(output, raw_adj, swc, bmin=12, bmax=image.shape[0]-12)

    # Keep all connections of soma
    if is_start:
        soma_node = np.where(swc[:,1] == 1)[0]
        recover_adj[soma_node, :] = raw_adj[soma_node, :]
        recover_adj[:, soma_node] = raw_adj[:, soma_node]
    swc_repair = adj2swc(swc,recover_adj)

    return swc_repair


def neuron_repair(config,image,swc,image_path,is_start=False):
    # image = read_image(image_path)
    # swc = load_swc(swc_path)
    print("[rep] raw swc: ",swc.shape)

    # remove the erro reconstruct nodes
    swc_correct = remove_erro_nodes(config,image,swc,is_start)
    print("[rep] swc remove: ",swc_correct.shape)

    # repair the edges between
    swc_repair = repair_edges(config,image,swc_correct,is_start)
    print("swc repair: ", swc_repair.shape)
    if config['rep_save_cash']:
        filename = os.path.basename(image_path)
        print(filename)
        # save_swc(os.path.join(config['rep_save_cash_path'], filename + '_prune.swc'), swc)
        save_swc(os.path.join(config['rep_save_cash_path'], filename + '_remove.swc'), swc_correct)
        save_swc(os.path.join(config['rep_save_cash_path'], filename + '_repair.swc'), swc_repair)
    return

if __name__ == '__main__':
    with open(r"F:\neuron_reconstruction_system\D_LSNARS_test\setup.yaml") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(config)

    IMAGE_PATH = r"I:\test\2025-06-10\23_29_27\mouse18455_teraconvert_tmp\x13977_y11007_z3484.tif"
    SWC_PATH = r"I:\test\2025-06-10\23_29_27\mouse18455_teraconvert_tmp\x13977_y11007_z3484.tif_spe_dnr.swc"
    IMAGE = read_image(IMAGE_PATH)
    SWC = load_swc(SWC_PATH)
    _ = neuron_repair(config,IMAGE,SWC,IMAGE_PATH)