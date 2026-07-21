import tifffile
import numpy as np
import torch

def read_image(image_path):
    return tifffile.imread(image_path)

def save_image(filepath, block):
    tifffile.imwrite(filepath, block.astype(np.uint8), compression=None)


def cut_block(image, block_size=[32, 32, 32], step=[32, 32, 32]):
    block_size = np.array(block_size, dtype=np.int32)
    step = np.array(step, dtype=np.int32)
    image_size = np.array(image.shape, dtype=np.int32)

    step_num = np.ceil(np.maximum(image_size - block_size, 0) / step + 1).astype(np.int32)
    pad_num = ((step_num - 1) * step + block_size - image_size).astype(np.int32)

    image_pad = np.pad(
        image,
        ((0, pad_num[0]), (0, pad_num[1]), (0, pad_num[2])),
        mode='reflect'
    )

    # --- 极速路径（当 切块大小 == 步长 时，无重叠） ---
    if np.array_equal(block_size, step):
        z_n, y_n, x_n = step_num.tolist()
        bz, by, bx = block_size.tolist()
        blocks = image_pad.reshape(z_n, bz, y_n, by, x_n, bx)
        blocks = blocks.transpose(0, 2, 4, 1, 3, 5)  # 调整维度顺序

        blocks_flat = np.ascontiguousarray(blocks).reshape(-1, bz, by, bx)
        block_list = list(blocks_flat)
        return block_list, step_num

    # --- 兼容路径（当有重叠切块时，使用常规循环） ---
    block_list = []
    for z in range(step_num[0]):
        z0 = z * step[0]
        for y in range(step_num[1]):
            y0 = y * step[1]
            for x in range(step_num[2]):
                x0 = x * step[2]
                block = image_pad[z0: z0 + block_size[0],
                        y0: y0 + block_size[1],
                        x0: x0 + block_size[2]]
                block_list.append(block)

    return block_list, step_num

def split_block(block_list, step_num):
    # 确保提取出的是 Python 原生 int
    z_n, y_n, x_n = int(step_num[0]), int(step_num[1]), int(step_num[2])

    # 1. 统一打包为整体矩阵
    if isinstance(block_list, list):
        if torch.is_tensor(block_list[0]):
            blocks = torch.stack(block_list, dim=0)
        else:
            blocks = np.stack(block_list, axis=0)
    else:
        blocks = block_list

    _, bz, by, bx = blocks.shape

    # 2. 魔法操作：根据数据类型选择对应的变形函数
    if torch.is_tensor(blocks):
        # PyTorch Tensor 路径
        grid = blocks.reshape(z_n, y_n, x_n, bz, by, bx)
        grid = grid.permute(0, 3, 1, 4, 2, 5).contiguous()
        block_z = grid.reshape(z_n * bz, y_n * by, x_n * bx)
    else:
        # NumPy Array 路径
        grid = blocks.reshape(z_n, y_n, x_n, bz, by, bx)
        grid = grid.transpose(0, 3, 1, 4, 2, 5)
        # NumPy 中的 reshape 默认就是内存连续优先的
        block_z = grid.reshape(z_n * bz, y_n * by, x_n * bx)

    return block_z