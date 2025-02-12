import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv2
from PIL import Image
from tqdm import tqdm

images = np.load('/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/images.npy', allow_pickle=True, mmap_mode='r')
masks = np.load('/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/masks.npy', allow_pickle=True, mmap_mode='r')

print(images.shape)
print(masks.shape)

out_dir = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/hover_net_format/'"

# A helper function to map 2d numpy array
def flat_for(a, f):
    a = a.reshape(-1)
    for i, v in enumerate(a):
        a[i] = f(v)


# A helper function to unique PanNuke instances indexes to [0..N] range where 0 is background
def map_inst(inst):
    seg_indexes = np.unique(inst)
    new_indexes = np.array(range(0, len(seg_indexes)))
    dict = {}
    for seg_index, new_index in zip(seg_indexes, new_indexes):
        dict[seg_index] = new_index

    flat_for(inst, lambda x: dict[x])

# A helper function to transform PanNuke format to HoverNet data format
def transform(images, masks, path, out_dir, start, finish):

    fold_path = out_dir+path
    try:
        os.mkdir(fold_path)
    except FileExistsError:
        pass
    
    start = int(images.shape[0]*start)
    finish = int(images.shape[0]*finish)
    
    for i in tqdm(range(start, finish)):
        np_file = np.zeros((256,256,5), dtype='int16')

        # add rgb channels to array
        img_int = np.array(images[i],np.int16)
        for j in range(3):
            np_file[:,:,j] = img_int[:,:,j]
        
        # convert inst and type format for mask
        msk = masks[i]

        inst = np.zeros((256,256))
        for j in range(5):
            #copy value from new array if value is not equal 0
            inst = np.where(msk[:,:,j] != 0, msk[:,:,j], inst)
        map_inst(inst)

        types = np.zeros((256,256))
        for j in range(5):
            # write type index if mask is not equal 0 and value is still 0
            types = np.where((msk[:,:,j] != 0) & (types == 0), j+1, types)

        # add padded inst and types to array
        np_file[:,:,3] = inst
        np_file[:,:,4] = types

        np.save(fold_path + '/' + '%d.npy' % (i), np_file)

transform(images, masks, 'train', out_dir=out_dir, start=0, finish=0.8)
transform(images, masks, 'val', out_dir=out_dir, start=0.8, finish=1)