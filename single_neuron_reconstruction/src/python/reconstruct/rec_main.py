import os
import numpy as np
import torch
import yaml

from reconstruct.rec_network import SPE_DNR
from reconstruct.rec_io import read_image,save_image,save_swc
from reconstruct.rec_tracker import Tracker
from reconstruct.rec_utils import compute_trees,nodelist2swc

from segment.seg_main import seg2seed
import cc3d

# --------------------------------------------------------------------------------------------- #
_rec_model = None
_rec_model_path = None
_rec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_rec_model(config):
    global _rec_model, _rec_model_path, _rec_device

    ckpt_path = config["rec_checkpoint_path"]
    if _rec_model is None or _rec_model_path != ckpt_path:
        model = SPE_DNR(NUM_ACTIONS=1024, n=10)
        checkpoint = torch.load(ckpt_path, map_location=_rec_device, weights_only=False)

        if isinstance(checkpoint, dict) and 'net_dict' in checkpoint:
            model.load_state_dict(checkpoint['net_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval().to(_rec_device)

        _rec_model = model
        _rec_model_path = ckpt_path

    return _rec_model, _rec_device

# --------------------------------------------------------------------------------------------- #
def pre_progress(image,soma_mask,seed_points,seg,pad_size,is_start_block=True,z_expand = True):

    image1 = (image-image.min()) / (image.max()-image.min()+1e-5)

    if seg is None:
        seg = np.zeros_like(image1, dtype=image1.dtype)
    else:
        seg = seg.astype(image1.dtype, copy=False)

    image1 = image1*0.6+seg*0.4

    if seed_points is None:
        seed_points1 = np.empty((0, 4), dtype=np.float32)
    else:
        seed_points1 = seed_points.copy()

    #select target soma
    if is_start_block:
        assert soma_mask is not None, "not find soma in start block"

        label, num = cc3d.connected_components(
            (soma_mask > 0).astype(np.uint8),
            connectivity=18,
            return_N=True
        )
        assert num != 0, "not seg soma in start block"

        centarl_coord = np.array(soma_mask.shape) // 2
        soma_mask1 = np.zeros_like(label, dtype=np.int16)

        # obtain soma location
        z0 = max(centarl_coord[0] - 20, 0)
        z1 = min(centarl_coord[0] + 20, soma_mask.shape[0])
        x0 = max(centarl_coord[1] - 20, 0)
        x1 = min(centarl_coord[1] + 20, soma_mask.shape[1])
        y0 = max(centarl_coord[2] - 20, 0)
        y1 = min(centarl_coord[2] + 20, soma_mask.shape[2])

        center_crop = label[z0:z1, x0:x1, y0:y1]
        center_ids = np.unique(center_crop)
        center_ids = center_ids[center_ids > 0]

        if center_ids.size > 0:
            target_mask = np.isin(label, center_ids)
            soma_mask1[(label > 0) & target_mask] = 1
            soma_mask1[(label > 0) & (~target_mask)] = -1
        else:
            soma_mask1[label > 0] = -1
    else:
        if soma_mask is not None:
            soma_mask1 = soma_mask.astype(np.int16, copy=True)
            soma_mask1[soma_mask1 == 1] = -1
        else:
            soma_mask1 = np.zeros_like(image1)

    #pad images
    if z_expand:
        image1 = np.repeat(image1, 2, axis=0)
        soma_mask1 = np.repeat(soma_mask1, 2, axis=0)
        if seed_points1.shape[0] != 0:
            seed_points1[:, 2] *= 2

    image1 = np.pad(image1, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), 'reflect')
    soma_mask1 = np.pad(soma_mask1, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), 'reflect')

    if seed_points1.shape[0] != 0:
        seed_points1[:, :3] += pad_size

    return image1, soma_mask1, seed_points1

def neuron_reconstruct(config,image,soma_mask,seed_points,seg,image_path,is_start_block = False):

    ## load model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SPE_DNR(NUM_ACTIONS=1024,n=10)
    # checkpoint = torch.load(config["rec_checkpoint_path"], map_location=lambda storage, loc: storage,weights_only=False)
    # model.load_state_dict(checkpoint['net_dict'])
    # model.eval().to(device)

    model, device = _get_rec_model(config)

    #params set
    SPE_NUM = config['SPE_NUM']
    SPE_STEP = config['SPE_STEP']
    pad_size = int(SPE_STEP*SPE_NUM)
    z_expand = config['z_expand']

    #load image
    # image = read_image(image_path)
    image1,soma_mask1,seed_points1 = pre_progress(image,soma_mask,seed_points,seg,
                                                  pad_size=pad_size,
                                                  is_start_block=is_start_block,
                                                  z_expand=z_expand)

    print("[rec] image1: ",image1.shape,image1.max(),image1.min())
    print("[rec] soma_mask1: ",soma_mask1.shape,soma_mask1.max(),soma_mask1.min())
    print("[rec] seed_points1: ",seed_points1.shape)

    #trace neuron
    seed_num = seed_points1.shape[0]
    assert seed_num < config['SEED_MAX'], "too much seed points"

    Tracker_ = Tracker(config,image1,soma_mask1,seed_points1,model,pad_size,device)
    with torch.inference_mode():
        Tracker_.tracker()
    Tracker_.connect_overlap_node()

    print("[rec] node_num: ",len(Tracker_.nodelist))

    # transform nodelist to swc
    neuron_tree = compute_trees(Tracker_.nodelist,Tracker_.overlap_nodelist,soma_mark=soma_mask1)
    swc = nodelist2swc(neuron_tree)

    # use this result for multi-neuron reconstruction
    swc[:, 2:5] = swc[:, 2:5] - pad_size
    if z_expand:
        swc[:,4] = swc[:,4]/2

    # Vaa3d starts from 1 but python from 0
    swc[:, 2:5] = swc[:, 2:5] + 1

    # prune
    if config['rec_save_cash']:
        filename = os.path.basename(image_path)
        print(filename)
        save_swc(os.path.join(config['rec_save_cash_path'],filename+'_spe_dnr.swc'),swc)

    return swc


if __name__ == '__main__':
    with open(r"F:\neuron_reconstruction_system\D_LSNARS_test\setup.yaml") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(config)

    IMAGE_PATH = r"F:\neuron_reconstruction_system\D_LSNARS_test\test_sample\x13308_y12068_z5510.tif"
    image = read_image(IMAGE_PATH)
    seg = read_image(r"F:\neuron_reconstruction_system\D_LSNARS_test\result\x13308_y12068_z5510.tif_seg.tif")
    # somamask = np.zeros_like(seg)
    somamask = read_image(r"F:\neuron_reconstruction_system\D_LSNARS_test\result\x13308_y12068_z5510.tif_somamask.tif")
    _, seed_points = seg2seed(seg,somamask)

    _ = neuron_reconstruct(config,
                           image,
                           somamask,
                           seed_points,
                           seg//255,
                           image_path=IMAGE_PATH,
                           is_start_block=True)
