import os
import numpy as np
import torch
from tqdm import tqdm
import yaml
from segment.seg_network import UNet3d_CBAM1,DTANET
from segment.seg_io import read_image,save_image,cut_block,split_block
from scipy.ndimage import binary_opening,binary_closing
from scipy import ndimage as ndi
from skimage.morphology import skeletonize_3d
from sklearn.cluster import DBSCAN
import cc3d
# Other Segment Method
from segment.WaveUNet3D.WaveUNet3D_network import Neuron_WaveSNet_V4
from segment.SGSNet.SGSNet_network import SGSNet

# ------------------------------- Global Model Cache --------------------------------------- #
_seg_model_cache = {
    'model': None,
    'ckpt_path': None,
    'device': None
}

def _get_segmentation_model(config):
    global _seg_model_cache

    ckpt_path = config["seg_checkpoint_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    if (_seg_model_cache['model'] is not None and
            _seg_model_cache['ckpt_path'] == ckpt_path and
            _seg_model_cache['device'] == device):
        return _seg_model_cache['model'], device

    model_name = config.get('seg_model_name', '')
    if 'UNET3D' in model_name:
        model = UNet3d_CBAM1()
    elif 'DTANET' in model_name:
        model = DTANET()
    elif 'WaveUNet3D' in model_name:
        model = Neuron_WaveSNet_V4()
    elif 'SGSNet' in model_name:
        model = SGSNet()
    else:
        raise ValueError(f"seg_model not exist: {model_name}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval().to(device)
    print("[seg] model: ",model_name)
    # 更新缓存
    _seg_model_cache['model'] = model
    _seg_model_cache['ckpt_path'] = ckpt_path
    _seg_model_cache['device'] = device

    return model, device

# --------------------------------------- Seed extraction Functions ---------------------------------------#

def _farthest_point_sampling(points, k):
    """最远点采样，points: (N, 3)"""
    n = points.shape[0]
    if n <= k:
        return np.arange(n)

    selected = np.empty(k, dtype=np.int64)
    selected[0] = 0
    dist2 = np.sum((points - points[0]) ** 2, axis=1)
    for i in range(1, k):
        idx = int(np.argmax(dist2))
        selected[i] = idx
        new_d2 = np.sum((points - points[idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_d2)
    return selected

def extract_centerline(seg, closing_size=1):
    if not seg.any():
        return np.zeros_like(seg, dtype=np.uint8)

    if closing_size and closing_size > 1:
        # structure = np.ones((closing_size,) * 3, dtype=bool)
        structure = np.ones((1, closing_size, closing_size), dtype=bool)
        seg_proc = binary_closing(seg, structure=structure)
    else:
        seg_proc = seg

    #
    skeleton = skeletonize_3d(seg_proc.astype(np.uint8)) > 0

    if not skeleton.any():
        return np.zeros_like(seg, dtype=np.uint8)

    return skeleton.astype(np.uint8)

def seg2seed(config,seg,soma_mask):

    if soma_mask is not None:
        seg1 = (seg + soma_mask >= 1).astype(np.uint8)
    else:
        seg1 = seg.astype(np.uint8)

    print("[seg] seg2seed-> seg1:",seg1.max(),seg1.min())

    # ----- skeleton extract --------#
    seed_image = extract_centerline(seg1)
    # while seed_num >= config['SEED_MAX']:
    #     seed_image = skeletonize_3d(seg1.astype(np.uint8))  # [0/1]
    #     seed_coord = np.where(seed_image > 0)  # [z,x,y]
    #     seed_num = len(seed_coord[0])
    #     print(seed_num)
    #     if seed_num >= config['SEED_MAX']:
    #         seg1 = binary_erosion(seg1)

    if not seed_image.any():
        print("Warning: no seed_point generated!")
        return seed_image, np.empty((0, 4), dtype=np.float32)

    # ---------- coord extract ------------------ #
    seed_max = config.get('SEED_MAX', 20000)

    seed_coords = np.where(seed_image > 0)  # (z, x, y)
    z_arr = seed_coords[0]
    x_arr = seed_coords[1]
    y_arr = seed_coords[2]

    seed_points = np.stack([x_arr, y_arr, z_arr, seed_image[z_arr, x_arr, y_arr]], axis=1).astype(np.float32)
    current_seed_num = seed_points.shape[0]

    # DBSCAN 去噪
    if current_seed_num >= 5:
        estimator = DBSCAN(eps=3, min_samples=5, n_jobs=-1) # n_jobs=-1 利用多核加速
        labels = estimator.fit_predict(seed_points[:, :3])

        keep_mask = labels != -1
        seed_points = seed_points[keep_mask]
        print(f"Seed points reduced: {current_seed_num} -> {seed_points.shape[0]}")
    else:
        seed_points = np.empty((0, 4), dtype=np.float32)

    # ---------- step 3. control points ----------
    if seed_max is not None and seed_points.shape[0] > seed_max:
        idx = _farthest_point_sampling(seed_points[:, :3], int(seed_max * 0.9))
        seed_points = seed_points[idx]
        print("[seg2seed] after FPS: ", seed_points.shape)

    return seed_image,seed_points

# --------------------------------------- Seg post_process Functions ---------------------------------------#
def remove_stepwise_edge(predict, TH=None):
    if TH is None:
        TH = 256 * 256 * 0.08

    # Z-axis projection cleanup
    predict_sum_z = np.sum(predict, axis=(0, 1))
    clean_z = predict_sum_z <= TH
    predict[:, :, :] = predict[:, :, :] * clean_z[np.newaxis, np.newaxis, :]

    # Y-axis projection cleanup
    predict_sum_y = np.sum(predict, axis=(0, 2))
    clean_y = predict_sum_y <= TH
    predict[:, :, :] = predict[:, :, :] * clean_y[np.newaxis, :, np.newaxis]

    # X-axis projection cleanup
    predict_sum_x = np.sum(predict, axis=(1, 2))
    clean_x = predict_sum_x <= TH
    predict[:, :, :] = predict[:, :, :] * clean_x[:, np.newaxis, np.newaxis]

    return (predict > 0).astype(np.uint8)

def remove_blob_artifacts_fast(seg,
                               min_size=30,
                               large_size=3000,
                               fill_ratio_th=0.12,
                               elongation_th=3.0,
                               connectivity=18,
                               slice_fill_ratio_th=0.10,
                               verbose=True):
    seg = (seg > 0)
    if not seg.any():
        return np.zeros_like(seg, dtype=np.uint8)

    # ---------------- Fast Pre-check 1: Total Foreground ----------------
    fg_total = int(seg.sum())

    if fg_total < large_size:
        if verbose:
            print(f"[post-fast] skip: total foreground={fg_total} < large_size={large_size}")
        return seg.astype(np.uint8)

    # ---------------- Fast Pre-check 2: Slice Area Occupancy ----------------
    seg_u8 = seg.astype(np.uint8)

    z_max = int(np.max(np.sum(seg_u8, axis=(1, 2))))
    x_max = int(np.max(np.sum(seg_u8, axis=(0, 2))))
    y_max = int(np.max(np.sum(seg_u8, axis=(0, 1))))

    z_th = int(seg.shape[1] * seg.shape[2] * slice_fill_ratio_th)
    x_th = int(seg.shape[0] * seg.shape[2] * slice_fill_ratio_th)
    y_th = int(seg.shape[0] * seg.shape[1] * slice_fill_ratio_th)

    if z_max < z_th and x_max < x_th and y_max < y_th:
        if verbose:
            print(f"[post-fast] skip: no large slice occupancy "
                  f"(z_max={z_max}, x_max={x_max}, y_max={y_max})")
        return seg_u8

    # ---------------- connected components ----------------
    labels = cc3d.connected_components(seg_u8, connectivity=connectivity)
    n_cc = int(labels.max())
    if n_cc == 0:
        return np.zeros_like(seg, dtype=np.uint8)

    voxel_counts = np.bincount(labels.ravel(), minlength=n_cc + 1).astype(np.int64)

    # ---------------- Fast Pre-check 3: Max CC Volume ----------------
    max_cc = int(voxel_counts[1:].max()) if n_cc > 0 else 0
    if max_cc < large_size:
        if verbose:
            print(f"[post-fast] skip: max_cc={max_cc} < large_size={large_size}")
        return seg.astype(np.uint8)

    # ---------------- Bounding Box Analysis ----------------
    objects = ndi.find_objects(labels)

    keep = np.zeros(n_cc + 1, dtype=bool)
    keep[0] = False

    removed_small = 0
    removed_blob = 0
    kept_num = 0

    for i, slc in enumerate(objects, start=1):
        if slc is None or voxel_counts[i] == 0:
            continue

        vol = int(voxel_counts[i])

        # Small noise processing
        if vol < min_size:
            removed_small += 1
            continue
        #
        dz = slc[0].stop - slc[0].start
        dx = slc[1].stop - slc[1].start
        dy = slc[2].stop - slc[2].start

        bbox_vol = dz * dx * dy
        fill_ratio = vol / (bbox_vol + 1e-6)

        dims = np.sort(np.array([dz, dx, dy], dtype=np.float32))
        elongation = dims[2] / (dims[1] + 1e-6)

        # Large clump-like artifact
        if vol >= large_size and fill_ratio >= fill_ratio_th and elongation <= elongation_th:
            removed_blob += 1
            continue

        #
        keep[i] = True
        kept_num += 1

    # ---------------- Reconstruction ----------------
    seg_out = keep[labels].astype(np.uint8)

    if verbose:
        print(f"[post-fast] CC total={n_cc}, keep={kept_num}, "
              f"remove_small={removed_small}, remove_blob={removed_blob}")

    return seg_out
# --------------------------------------- Main Functions ---------------------------------------#
def segment_block(config,image,image_path,soma_mask=None):
    # load model
    model, device = _get_segmentation_model(config)

    #load image
    # image = read_image(image_path)
    img_max = image.max()
    if img_max > 0:
        image = (image / img_max).astype(np.float32, copy=False)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    print("[seg] Image normalized:",image.max(),image.min())

    #cut image
    batch_size = config.get('seg_batch', 8)
    block_list, step_num = cut_block(image)

    if isinstance(block_list, list):
        block_array = np.stack(block_list, axis=0)
    else:
        block_array = block_list

    total_blocks = block_array.shape[0]
    pred_block = []
    with torch.no_grad():
        for i in range(0, total_blocks, batch_size):
            end_idx = min(i + batch_size, total_blocks)
            batch_data = block_array[i:end_idx]

            input_tensor = torch.from_numpy(batch_data).float().to(device)
            input_tensor = input_tensor.unsqueeze(1)

            output = model(input_tensor) #[32,2,32,32,32]
            output = torch.argmax(output, dim=1).cpu()

            pred_block.extend(output.numpy())

    pred_array = np.stack(pred_block, axis=0)
    predict = split_block(pred_array, step_num)

    predict = binary_opening(predict > 0, structure=np.ones((2, 2, 1))).astype(np.uint8)

    # Remove background mutations in some images
    predict = remove_stepwise_edge(predict)
    predict = remove_blob_artifacts_fast(predict)

    # seed point generation
    seed_maps,seed_points = seg2seed(config,predict,soma_mask)

    if config['seg_save_cash']:
        filename = os.path.basename(image_path)
        print(filename)
        save_image(os.path.join(config['seg_save_cash_path'],filename+'_seg.tif'),predict*255.0)
        save_image(os.path.join(config['seg_save_cash_path'],filename+'_seedmaps.tif'),seed_maps*255.0)
    return predict,seed_points #[0,1]


if __name__ == '__main__':
    with open(r"E:\vaa3d_plug\D_LSNARS\D_LSNARS_test\setup.yaml") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(config)

    IMAGE_PATH = r"F:\Neruron_Repair\NeuronRDR\data\test_for_origincode\x4544_y11883_z4438.tif"
    _ = segment_block(config,IMAGE_PATH)
