import os

import numpy as np


def save_swc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]

    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)
def loadswc(filepath,start,end):
    # load swc file as a N X 7 numpy array
    # n type x y z r parent
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        print(len(lines))
        for l in lines[start:end]:
            if not l.startswith('#'):
                cells = l.split(' ')
                # remove empty units
                if len(cells) != 9:
                    for kk in range(len(cells) - 1, -1, -1):
                        if cells[kk] == '':
                            cells.pop(kk)
                if len(cells) >= 7:
                    cells = [float(c) for c in cells]  # transform string to float
                    swc.append(cells[0:7])
    return np.array(swc)

def split_swc():
    swc_path = r"J:\registered\18465_axon_seed_25um.swc"
    save_dir = r"J:\registered\18465_cut_file"
    # swc = loadswc(swc_path)
    step = 20
    node_num = 30716934//step
    for i in range(step):
        # swc_cut = swc[i*node_num:(i+1)*node_num]
        swc_cut = loadswc(swc_path,i*node_num,(i+1)*node_num)
        swc_cut[:,0] = np.arange(1,swc_cut.shape[0]+1)
        print(swc_cut.shape)
        save_swc(os.path.join(save_dir,'cut_'+str(i)+'.swc'),swc_cut)

split_swc()

