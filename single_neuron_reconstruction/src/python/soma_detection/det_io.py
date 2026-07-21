import torch
import tifffile
import numpy as np

def read_image(image_path):
    return tifffile.imread(image_path)

def save_image(filepath, block):
    tifffile.imwrite(filepath, block.astype(np.uint8), compression=None)

def cut_block(image,block_szie=[64,64,64],step=[64,64,64]):
    block_szie = np.asarray(block_szie, dtype=np.int32)
    step = np.asarray(step, dtype=np.int32)
    image_size = np.asarray(image.shape, dtype=np.int32)

    step_num = np.ceil(np.maximum(image_size - block_szie, 0) / step + 1).astype(np.int32)
    pad_num = ((step_num - 1) * step + block_szie - image_size).astype(np.int32)
    image_pad = np.pad(
        image,
        ((0, int(pad_num[0])), (0, int(pad_num[1])), (0, int(pad_num[2]))),
        mode='reflect'
    )

    # 快速路径：非重叠切块（你当前就是这种）
    if np.all(step == block_szie):
        z_num, y_num, x_num = step_num.tolist()
        bz, by, bx = block_szie.tolist()

        blocks = image_pad.reshape(z_num, bz, y_num, by, x_num, bx)
        blocks = blocks.transpose(0, 2, 4, 1, 3, 5)
        blocks = np.ascontiguousarray(blocks.reshape(-1, bz, by, bx))
        return blocks, step_num

    # 通用路径：保留兼容
    block_list = []
    for z in range(step_num[0]):
        z0 = z * step[0]
        for y in range(step_num[1]):
            y0 = y * step[1]
            for x in range(step_num[2]):
                x0 = x * step[2]
                block = image_pad[
                        z0:z0 + block_szie[0],
                        y0:y0 + block_szie[1],
                        x0:x0 + block_szie[2]
                        ]
                block_list.append(block)

    block_list = np.ascontiguousarray(np.stack(block_list, axis=0))
    return block_list, step_num

def split_block(block_list, step_num):
    step_num = np.asarray(step_num, dtype=np.int32)
    z_num, y_num, x_num = step_num.tolist()

    if isinstance(block_list, list):
        if torch.is_tensor(block_list[0]):
            block_list = torch.stack(block_list, dim=0)
        else:
            block_list = np.stack(block_list, axis=0)

    bz, by, bx = block_list.shape[-3:]

    if torch.is_tensor(block_list):
        block_z = block_list.reshape(z_num, y_num, x_num, bz, by, bx)
        block_z = block_z.permute(0, 3, 1, 4, 2, 5)
        block_z = block_z.reshape(z_num * bz, y_num * by, x_num * bx)
    else:
        block_z = block_list.reshape(z_num, y_num, x_num, bz, by, bx)
        block_z = block_z.transpose(0, 3, 1, 4, 2, 5)
        block_z = block_z.reshape(z_num * bz, y_num * by, x_num * bx)

    return block_z