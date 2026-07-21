import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import yaml

from scipy.ndimage import binary_dilation,zoom
from skimage.morphology import ball,disk
from skimage.transform import resize
import cc3d

from soma_detection.det_network import ClassNet,SFSNet
from soma_detection.det_io import read_image,save_image,cut_block,split_block

# --------------------------------------- global  --------------------------------------------------#
_cla_model = None
_seg_model = None
_cla_model_path = None
_seg_model_path = None
cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_zoom(image, out_shape, order=0):
    """
    Use scipy.ndimage.zoom to resize image to exact out_shape.
    order=0: nearest, fastest, and closest to your original code.
    """
    out_shape = np.asarray(out_shape, dtype=np.int32)
    in_shape = np.asarray(image.shape, dtype=np.float32)

    zoom_factor = out_shape / in_shape
    out = zoom(
        image,
        zoom=zoom_factor,
        order=order,
        mode='nearest',
        prefilter=False
    )

    # ensure exact shape
    if out.shape != tuple(out_shape):
        out_fix = np.zeros(tuple(out_shape), dtype=out.dtype)
        slices_src = tuple(slice(0, min(out.shape[i], out_shape[i])) for i in range(len(out_shape)))
        slices_dst = tuple(slice(0, min(out.shape[i], out_shape[i])) for i in range(len(out_shape)))
        out_fix[slices_dst] = out[slices_src]
        out = out_fix

    return out
# --------------------------------------- block_classify --------------------------------------------------#
def _get_cla_model(config):
    global _cla_model, _cla_model_path

    ckpt_path = config["det_cla_checkpoint_path"]
    if _cla_model is None or _cla_model_path != ckpt_path:
        cla_model = ClassNet()
        # use cpu
        cla_checkpoint = torch.load(
            ckpt_path,
            map_location=cpu_device,  # CPU
            weights_only=False
        )
        cla_model.load_state_dict(cla_checkpoint)
        cla_model.eval().to(cpu_device)

        _cla_model = cla_model
        _cla_model_path = ckpt_path

    return _cla_model

def block_classify(config,image):
    # load model
    cla_model = _get_cla_model(config)

    # cla_model = ClassNet()
    # cla_checkpoint = torch.load(config["det_cla_checkpoint_path"], map_location=lambda storage, loc: storage, weights_only=False)
    # cla_model.load_state_dict(cla_checkpoint)
    # cla_model.eval().to(device)

    #image pre_grocess
    image_mip = np.max(image, axis=0)

    image_mip = (image_mip - image_mip.min()) / (image_mip.max() - image_mip.min() + 1e-5)
    image_mip = resize_zoom(image_mip, (256, 256), order=0).astype(np.float32, copy=False)
    image_mip = torch.from_numpy(image_mip).float()
    image_mip = image_mip.unsqueeze(dim=0).unsqueeze(dim=0)

    # image_mip = torch.nn.functional.interpolate(image_mip, size=(256, 256)) #[1,1,256,256]
    # print(image_mip.shape)

    # predict soma
    # image_mip = image_mip.to(device)
    with torch.inference_mode():
        predict = cla_model(image_mip)
        predict = torch.argmax(predict, dim=1)

    pred_value = int(predict.item())
    print("cla_result: ", pred_value)

    return pred_value == 1

# -------------------------------------- block_segment -----------------------------------------------#
def _get_seg_model(config):
    global _seg_model, _seg_model_path

    ckpt_path = config["det_seg_checkpoint_path"]

    if _seg_model is None or _seg_model_path != ckpt_path:
        seg_model = SFSNet()

        seg_checkpoint = torch.load(
            ckpt_path,
            map_location=device,  #  GPU
            weights_only=False
        )
        seg_model.load_state_dict(seg_checkpoint)
        seg_model.eval().to(device)

        _seg_model = seg_model
        _seg_model_path = ckpt_path

    return _seg_model

def block_segment(config,image,is_start=False):
    # load model
    seg_model = _get_seg_model(config)
    # seg_model = SFSNet()
    # seg_checkpoint = torch.load(config["det_seg_checkpoint_path"], map_location=lambda storage, loc: storage,weights_only=False)
    # seg_model.load_state_dict(seg_checkpoint)
    # seg_model.eval().to(device)

    # cut image
    image_shape = np.array(image.shape)
    batch_size = config['det_seg_batch']
    image_resample = resize(image, image_shape // 2, order=0, anti_aliasing=False)
    # image_resample = resize(image,image_shape//2,order=0, anti_aliasing=False)

    block_list, step_num = cut_block(image_resample)

    pred_block = []
    with torch.inference_mode():
        for i in range(0, len(block_list), batch_size):
            input = block_list[i:min(i + batch_size, len(block_list))]
            input = torch.from_numpy(np.array(input)).float().to(device)
            input = input.unsqueeze(1)

            output = seg_model(input)
            output = torch.argmax(output, dim=1).detach().cpu()

            pred_block.extend(output.numpy())

    predict = split_block(pred_block, step_num)
    if isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()

    if is_start:
        predict = binary_dilation(predict,structure=ball(2))
    else:
        predict = binary_dilation(predict,structure=ball(1))

    predict = post_process(predict,
                           area_min=config.get('soma_area_min',500),
                           area_max=config.get('soma_area_max',50000))

    soma_mask = resize_zoom(predict.astype(np.float32, copy=False), image_shape, order=0)
    # soma_mask = resize(predict,image_shape, order=0, anti_aliasing=False)

    # print("soma_area: ", np.sum(soma_mask > 0))
    return soma_mask

def post_process(image,area_min,area_max):
    label = cc3d.connected_components((image > 0).astype(np.uint8), connectivity=26)
    stats = cc3d.statistics(label)
    voxel_counts = stats["voxel_counts"]  # index 0 is background

    print("[soma det]: det soma list")
    for i in range(1, len(voxel_counts)):
        print(voxel_counts[i])

    keep = (voxel_counts >= area_min) & (voxel_counts <= area_max)
    keep[0] = False
    image[~keep[label]] = 0
    return image

# -------------------------------------------------------------------------------------------------#
def soma_detection(config, image, image_path, is_start=False):
    # load image
    image = (image - image.min()) / (image.max() - image.min() + 1e-5).astype(np.float32, copy=False)

    # Determine whether the block contains soma
    if not block_classify(config,image) and not is_start:
        return None
    else:
        #segment the soma from block
        soma_mask = block_segment(config,image,is_start)
        if config['det_seg_save_cash']:
            filename = os.path.basename(image_path)
            print(filename)
            save_image(os.path.join(config['det_seg_save_cash_path'], filename + '_somamask.tif'), soma_mask * 255.0)

        return soma_mask #[0,1]

if __name__ == '__main__':
    with open(r"F:\neuron_reconstruction_system\D_LSNARS_test\setup.yaml") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(config)

    IMAGE_PATH = r"F:\neuron_reconstruction_system\D_LSNARS_test\test_sample\x13308_y12068_z5510.tif"
    soma_detection(config,IMAGE_PATH)
