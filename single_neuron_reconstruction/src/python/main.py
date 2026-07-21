import os
import argparse
import yaml
import torch
from data_process import image_preprocessing
from reconstruct.rec_main import neuron_reconstruct
from soma_detection.det_main import soma_detection
from segment.seg_main import segment_block
from repaire.rep_main import neuron_repair
from utils import read_image,resample_vaa3d,load_swc,prune_branch

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neuron reconstruction main program")
    parser.add_argument('--input_path', '-i', type=str, help='Input image path')
    parser.add_argument('--configuration_path', '-c', type=str, help='Configuration file path')
    parser.add_argument('--is_start', '-s', type=str, default='0', choices=['0', '1'], help='Is it the starting block (0/1)')
    return parser.parse_args()


def load_config(configuration_path: str, image_path: str) -> dict:
    """
    Load and update the configuration file

    Args:
        configuration_path: .yaml configuration file path
        image_path: Input image path

    Returns:
        config
    """
    if not os.path.exists(configuration_path):
        raise FileNotFoundError(f"The configuration file does not exist: {configuration_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file does not exist: {image_path}")

    with open(configuration_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Set the result saving path to the directory where the image is located
    save_path = os.path.dirname(image_path)
    print(f"Result save path: {save_path}")
    for key in config:
        if "save_cash_path" in key:
            config[key] = save_path

    # Update the model checkpoint to an absolute path
    python_code_path = config.get('Python_code_path', '')
    print(f"Python code path: {python_code_path}")
    for key in config:
        if 'checkpoint_path' in key:
            config[key] = os.path.join(python_code_path, config[key])

    print(f"z_expand: {config.get('z_expand', False)}")
    return config

def main(config,image_path,is_start=False):
    """
       Main process of neuron reconstruction
       Args:
           config: configuration
           image_path: input image path
           is_start: whether it is the starting block flag
    """
    #
    image_name = os.path.basename(image_path)

    #step 1: image_process
    print("-- step 1: image_process --")
    image = read_image(image_path)
    image = image_preprocessing(config,image)

    #step 2: soma detection
    print("-- step 2: start soma detection --")
    soma_mask = None
    if config.get('det_use', True):
        soma_mask = soma_detection(config,image.copy(),image_path,is_start)
        torch.cuda.empty_cache()

    #step 3: segmentation and extract seed points
    print("-- step 3: start segmentation and extract seed points --")
    seg = None
    seed_points = None
    if config.get('seg_use', True):
        seg,seed_points = segment_block(config,image.copy(),image_path,soma_mask)
        torch.cuda.empty_cache()

    #step 4: neuron reconstruction
    print("-- step 4: start neuron reconstruction --")
    rec_model_name = config.get('rec_model_name', 'SPE_DNR').strip()
    print("rec_model: ",rec_model_name)

    if rec_model_name == 'SPE_DNR':
        neuron_reconstruct(config,
                             image.copy(),
                             soma_mask,
                             seed_points=seed_points,
                             seg=seg,
                             image_path=image_path,
                             is_start_block = is_start)
        torch.cuda.empty_cache()
        swc_path = os.path.join(config['rec_save_cash_path'], image_name + '_spe_dnr.swc')

    elif rec_model_name == 'NeuTube':
        from reconstruct.NeuTubepy.rec_NeuTube_tracker import NeuTube_tracker
        NeuTube_tracker(config,
                        image.copy(),
                        seg_image=seg,
                        soma_mask=soma_mask,
                        image_path=image_path,
                        is_start=is_start
                        )
        swc_path = os.path.join(config["rec_save_cash_path"], image_name + "_neutube.swc")

    elif rec_model_name == 'RivuNet':
        from reconstruct.RivuNetpy.rec_RivuNet_tracker import RivuNet_tracker
        RivuNet_tracker(config,
                        image.copy(),
                        seg_image=seg,
                        soma_mask=soma_mask,
                        image_path=image_path,
                        is_start=is_start,
                        cash_remove=True
                        )
        swc_path = os.path.join(config["rec_save_cash_path"], image_name + "_rivunet.swc")

    else:
        raise ValueError('rec_model not find !!!')

    #step 5: resample and repair neuron reconstruction result
    print("-- step 5: resample and repair neuron reconstruction result --")

    ### resample
    output_path = os.path.join(config['resample_save_cash_path'], image_name + '_resample.swc')
    if os.path.exists(swc_path):
        resample_vaa3d(config, swc_path=swc_path, output_path=output_path)
    else:
        print(f"Warning: swc_path not found, resample skipped: {swc_path}")

    ### repair neuron reconstruction result
    if config.get('rep_use', True):
        swc = load_swc(output_path)
        if swc is not None and 0 < swc.shape[0] < 10000:

            swc = prune_branch(swc,bmax=image.shape[0]-12)

            neuron_repair(config, image, swc, image_path, is_start = is_start)

            repaired_swc_path = os.path.join(config['rec_save_cash_path'],image_name + '_repair.swc')
            if os.path.exists(repaired_swc_path):
                resample_vaa3d(config, swc_path=repaired_swc_path, output_path=output_path)
            else:
                print(f"Warning: repaired swc not found: {repaired_swc_path}")

        else:
            print(f"Skip repair: SWC node count={swc.shape[0] if swc is not None else 0}")

    print("python finished")

if __name__ == '__main__':
    args = parse_args()

    IMAGE_PATH = args.input_path
    CONFIGURATION_PATH = args.configuration_path
    IS_START = args.is_start == '1'

    # IMAGE_PATH = r"F:\neuron_reconstruction_system\D_LSNARS_test_block\test_sample\x16058_y15553_z2887.tif"
    # CONFIGURATION_PATH = r'F:\neuron_reconstruction_system\D_LSNARS_test_0622\setup_RivuNet.yaml'
    # IS_START = False

    CONFIG = load_config(CONFIGURATION_PATH, IMAGE_PATH)
    main(CONFIG, IMAGE_PATH, is_start=IS_START)


