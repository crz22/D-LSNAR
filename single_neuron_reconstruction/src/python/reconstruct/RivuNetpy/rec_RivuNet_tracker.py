import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.ioff()
plt.show = lambda *args, **kwargs: None

import tifffile as tiff
import numpy as np
import warnings

import cc3d
import shutil
from scipy import ndimage as ndi

from rivunetpy.rivunetpy import Tracer
from reconstruct.RivuNetpy.rec_RivuNet_utils import swc2Nodelist,compute_trees,nodelist2swc
from reconstruct.RivuNetpy.rec_RivuNet_utils import tif2imagej_tif,merge_swcs,_swc_to_array
from reconstruct.rec_io import read_image,save_image,save_swc


warnings.filterwarnings(
    "ignore",
    message="HyperStack: Warning, no voxel size found.*"
)
warnings.filterwarnings(
    'ignore',
    category=RuntimeWarning,
    message='invalid value encountered in cast'
)

sys.setrecursionlimit(100000)


def prepare_rivunet_input(seg_image, image=None):
    """
    综合方案：裂缝填补 + 软边界 + 保留背景低信号

    最适合 RivuNet 的输入
    """
    seg = seg_image.astype(np.float32)

    if seg.max() == 0:
        return np.zeros_like(seg)

    from scipy.ndimage import (
        binary_closing, binary_fill_holes,
        binary_dilation, distance_transform_edt,
        gaussian_filter
    )

    # ── Step1: 裂缝填补 ────────────────────────────────────────────
    seg_norm = seg / (seg.max() + 1e-8)
    seg_bin = (seg_norm > 0.1).astype(np.uint8)

    struct = np.ones((3, 3, 3), dtype=bool)
    seg_closed = binary_closing(seg_bin, structure=struct, iterations=2)
    seg_filled = binary_fill_holes(seg_closed)

    # ── Step2: 距离变换 → 中心亮边缘暗 ─────────────────────────────
    dist = distance_transform_edt(seg_filled).astype(np.float32)
    if dist.max() > 0:
        dist = dist / dist.max()

    # ── Step3: 软边界（边缘渐变）────────────────────────────────────
    dilated = binary_dilation(seg_filled, iterations=5)
    seg_soft = gaussian_filter(dilated.astype(np.float32), sigma=2.0)
    seg_soft = np.clip(seg_soft, 0, 1)

    # ── Step4: 构建最终图像 ─────────────────────────────────────────
    if image is not None:
        img = image.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = img * 0
    else:
        img_norm = dist

    # 最终图像 = 内容 × 软掩膜 + 背景基础信号
    bg_level = 0.1  # 背景保留 10% 信号
    result = img_norm * seg_soft + bg_level * (1 - seg_soft) * dist

    # 归一化到 0~255
    result = result / (result.max() + 1e-8) #* 255.0

    return result.astype(np.float32)


def seg_to_smooth_intensity(seg_image, sigma=1.5):
    """
    裂缝填补 + 距离变换 + 高斯平滑

    把稀疏二值分割图变成 RivuNet 可用的灰度图
    """
    seg = (seg_image > 0).astype(np.uint8)

    if seg.max() == 0:
        return seg.astype(np.float32)

    # 1. 形态学闭运算：填补小裂缝
    from scipy.ndimage import binary_closing, binary_fill_holes

    # 小结构元素填补裂缝
    struct_small = np.ones((3, 3, 3), dtype=bool)
    seg_closed = binary_closing(seg, structure=struct_small, iterations=2).astype(np.uint8)

    # 填补内部空洞
    seg_filled = binary_fill_holes(seg_closed).astype(np.uint8)

    # 2. 距离变换：前景内部中心亮，边缘暗
    dist = ndi.distance_transform_edt(seg_filled).astype(np.float32)

    # 3. 归一化到 0~255
    if dist.max() > 0:
        dist = dist / dist.max() #* 255.0

    # 4. 高斯平滑：让梯度更连续
    dist_smooth = ndi.gaussian_filter(dist, sigma=sigma).astype(np.float32)

    return dist_smooth

# --------------------------------------------------------------------------------------------------- #
def RivuNet_tracker(config,image,seg_image,soma_mask,image_path,is_start=False,cash_remove=False):
    # arr = clean_binary_3d(arr)
    if seg_image is not None:
        # image_raw = image.astype(np.float32, copy=False) if image is not None else None
        seg = seg_image.astype(np.float32, copy=False)

        # 用综合方案处理输入
        # image1 = prepare_rivunet_input(seg, image_raw)
        image1 = seg_to_smooth_intensity(seg,sigma=1.0)
        # select target soma
        if is_start:
            assert soma_mask is not None, "not find soma in start block"

            # label, num = measure.label(soma_mask, connectivity=2, return_num=True)  # 1代表４连通，２代表８连通
            label = cc3d.connected_components(soma_mask.astype(np.uint8), connectivity=26)
            num = int(label.max())
            assert num != 0, "not seg soma in start block"

            centarl_coord = np.array(soma_mask.shape) // 2
            soma_mask1 = soma_mask.copy().astype(np.int16)

            # obtain soma location
            local_label = label[centarl_coord[0] - 20:centarl_coord[0] + 20,
                          centarl_coord[1] - 20:centarl_coord[1] + 20, centarl_coord[2] - 20:centarl_coord[2] + 20]
            target_labels = np.unique(local_label)
            target_labels = target_labels[target_labels > 0]

            # 非目标 soma 标记为 -1
            soma_mask1[label > 0] = -1

            # 中心区域对应的目标 soma 标记为 1
            if len(target_labels) > 0:
                soma_mask1[np.isin(label, target_labels)] = 1
            else:
                print("warning: no soma label found near center region")

        else:
            if soma_mask is not None:
                soma_mask1 = soma_mask.copy().astype(np.int16)
                soma_mask1[soma_mask1 == 1] = -1
            else:
                soma_mask1 = np.zeros_like(image1, dtype=np.int16)
        print('soma_mask: ', soma_mask1.min(), soma_mask1.max())
        # image1[soma_mask1>0] = 1

        print('[RivuNet] image1: ', image1.max(), image1.min())

        # ------------------------------------------------------------------- #
        fixed_img_path = image_path + "_hyperstack.tif"
        tif2imagej_tif(image1,fixed_img_path)

        output_dir = os.path.dirname(image_path)
        cashe_save_dir = os.path.join(output_dir,'cash')

        # if os.path.exists(cashe_save_dir):
        #     shutil.rmtree(cashe_save_dir)
        # os.makedirs(cashe_save_dir, exist_ok=True)

        # ------------------------------------------------------------------- #
        tracer = Tracer()
        tracer.set_threshold(20)
        tracer.set_file(fixed_img_path)
        tracer.set_output_dir(cashe_save_dir)
        tracer.asynchronous_off()  # ← 建议打开，避免多进程报错难排查
        tracer.overwrite_cache_on()


        try:
            tracer._segment()

            print("_write_segmentation_to_file")
            tracer._write_segmentation_to_file()  # ← 这一句不能省

            print("_trace_all")
            tracer._trace_all()

            neurons = getattr(tracer, 'neurons', None) or \
                      getattr(tracer, '_neurons', [])

        except RecursionError:
            print("[RivuNet WARN] RecursionError，部分神经元被跳过")
            neurons = getattr(tracer, 'neurons', None) or \
                      getattr(tracer, '_neurons', [])

        except Exception as e:
            print(f"[RivuNet WARN] trace failed: {e}")
            neurons = getattr(tracer, 'neurons', None) or \
                      getattr(tracer, '_neurons', [])

        if neurons is None or len(neurons) == 0:
            print("[RivuNet WARN] No reconstruction was done，返回空 SWC")
            finnal_swc = np.empty([1, 7])
        else:
            print(f"[RivuNet] tracer reconstructed {len(neurons)} neuron branchs")

            # ------------------------------------------------------------------- #
            swc_list = []
            for i, neuron in enumerate(neurons):
                swc_obj = getattr(neuron, 'swc', None)

                if swc_obj is None:
                    print(f"[RivuNet WARN] neuron[{i}] No swc，skip")
                    continue

                swc_data = getattr(swc_obj, '_data', None)
                if swc_data is None:
                    print(f"[RivuNet WARN] neuron[{i}] swc._data is None，skip")
                    continue

                try:
                    swc_arr = _swc_to_array(swc_data)
                    if swc_arr is None or swc_arr.size == 0:
                        print(f"[RivuNet WARN] neuron[{i}] swc is empty，skip")
                        continue
                    swc_list.append(swc_arr)
                except Exception as e:
                    print(f"[RivuNet WARN] neuron[{i}] swc extraction failed，skip: {e}")

            if len(swc_list) == 0:
                finnal_swc = np.empty([1, 7])
            else:
                # ------------------------------------------------------------------- #
                merge_swc = merge_swcs(swc_list)
                print(f"[RivuNet] block neuron nodes: {merge_swc.shape}")

                # ------------------------------------------------------------------- #
                nodelist = swc2Nodelist(merge_swc)
                tree = compute_trees(nodelist, soma_mark=soma_mask1)
                finnal_swc = nodelist2swc(tree)

    else:
        finnal_swc = np.empty([1,7])
    # cash_remove = False
    if cash_remove:
        if os.path.exists(cashe_save_dir):
            shutil.rmtree(cashe_save_dir)
        if os.path.exists(fixed_img_path):
            os.remove(fixed_img_path)

    if config['rec_save_cash']:
        filename = os.path.basename(image_path)
        print(filename)
        save_swc(os.path.join(config['rec_save_cash_path'], filename + '_rivunet.swc'), finnal_swc)
    return


if __name__ == '__main__':
    CONFIGURATION_PATH = r'F:\neuron_reconstruction_system\D_LSNARS_test_0622\setup_RivuNet.yaml'


    image_path = r"F:\neuron_reconstruction_system\whole_reconstruction_resluts\17302_test\2026-06-25\16_13_13\mouse17302_teraconvert_tmp\x16413_y14119_z4516.tif"
    output_dir = r"test_data"

    CONFIG = {'rec_save_cash': True,
              'rec_save_cash_path':os.path.dirname(image_path)}

    image = tiff.imread(image_path)
    image = image/image.max()

    seg_image = tiff.imread(image_path+'_seg.tif')
    seg_image[seg_image>0] = 1

    soma_mask = tiff.imread(image_path+'_somamask.tif')

    RivuNet_tracker(CONFIG,
                    image.copy(),
                    seg_image=seg_image,
                    soma_mask=soma_mask,
                    image_path=image_path,
                    is_start=True
                    )





