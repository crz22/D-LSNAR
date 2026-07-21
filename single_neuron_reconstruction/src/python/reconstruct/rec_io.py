import tifffile
import numpy as np

def read_image(image_path):
    return tifffile.imread(image_path)

def save_image(filepath, block):
    tifffile.imwrite(filepath, block.astype(np.uint8), compression=None)

def save_swc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]

    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)