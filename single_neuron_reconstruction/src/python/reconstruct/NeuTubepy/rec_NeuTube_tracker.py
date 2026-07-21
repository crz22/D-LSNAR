import os
import numpy as np
from reconstruct.rec_io import read_image,save_image,save_swc
from reconstruct.NeuTubepy.rec_NeuTube_utils import swc2Nodelist,compute_trees,nodelist2swc
from pyneutube import trace_file,trace_volume
import cc3d

def NeuTube_tracker(config,image,seg_image,soma_mask,image_path,is_start=False):
    image1 = image.astype(np.float32, copy=False)

    if seg_image is not None:
        seg = seg_image.astype(np.float32, copy=False)
        image1 = image1 * 0.6 + seg * 0.4

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
        print('soma_mask: ',soma_mask1.min(),soma_mask1.max())

    print('image1: ',image1.max(),image1.min())

    result = trace_volume(image=image1 * 255.0)
    swc = result.neuron.swc
    print('neutube results: ',result.neuron.swc.shape)

    nodelist = swc2Nodelist(swc)
    tree = compute_trees(nodelist,soma_mark=soma_mask1)
    finnal_swc = nodelist2swc(tree)

    if config['rec_save_cash']:
        filename = os.path.basename(image_path)
        print(filename)
        save_swc(os.path.join(config['rec_save_cash_path'], filename + '_neutube.swc'), finnal_swc)
    return

    # return TracingResult(
    #     image_path=None,
    #     threshold=threshold,
    #     seeds=seeds,
    #     chains=chains,
    #     neuron=neuron,
    #     pre_postprocess_neuron=pre_postprocess_neuron,
    #     signal_image=signal_image_result,
    #     binary_image=binary_image_result,
    # )

if __name__ == '__main__':
    result = trace_file(
        "test_data/x18644_y19317_z4636.tif_seg.tif",
        output_swc="test_data/reference_trace.swc",
        visualization_dir="test_data/visualizations",
        config=None,
        n_jobs=1,
        timeout=600,
        verbose=1,
        overwrite=False,
    )
    # result = trace_volume()
    # print(result.threshold, len(result.seeds), len(result.chains))
