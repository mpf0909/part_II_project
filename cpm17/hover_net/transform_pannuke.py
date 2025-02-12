## code from https://github.com/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation?tab=readme-ov-file 
import numpy as np
import os

images_1 = np.load('dataset/pannuke/fold_1/images/images.npy', mmap_mode='r')
masks_1 = np.load('dataset/pannuke/fold_1/masks/masks.npy', mmap_mode='r')
images_2 = np.load('dataset/pannuke/fold_2/images/images.npy', mmap_mode='r')
masks_2 = np.load('dataset/pannuke/fold_2/masks/masks.npy', mmap_mode='r')
images_3 = np.load('dataset/pannuke/fold_3/images/images.npy', mmap_mode='r')
masks_3 = np.load('dataset/pannuke/fold_3/masks/masks.npy', mmap_mode='r')
out_dir = "dataset/pannuke/hover_net_format"

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
def transform(images, masks, path, out_dir, norm=False):
    os.makedirs(out_dir, exist_ok=True)

    fold_path = os.path.join(out_dir, path)
    os.makedirs(fold_path, exist_ok=True)

    for i in range(images.shape[0]):
        stop = False
        np_file = np.zeros((256, 256, 5), dtype='int16')

        img_int = np.array(images[i], np.int16)
        for j in range(3):
            np_file[:, :, j] = img_int[:, :, j]

        msk = masks[i]

        inst = np.zeros((256, 256))
        for j in range(5):
            inst = np.where(msk[:, :, j] != 0, msk[:, :, j], inst)
        map_inst(inst)

        types = np.zeros((256, 256))
        for j in range(5):
            types = np.where((msk[:, :, j] != 0) & (types == 0), j + 1, types)

        np_file[:, :, 3] = inst
        np_file[:, :, 4] = types

        if not stop:
            save_path = os.path.join(fold_path, f'{path}_{i+1}.npy')
            print(f"Saving to {save_path}")
            np.save(save_path, np_file)

# For the correct evaluation we need to keep training, validation and test folds
transform(images_1, masks_1, 'fold1', out_dir=out_dir, norm=True)
transform(images_2, masks_2, 'fold2', out_dir=out_dir, norm=True)
transform(images_3, masks_3, 'fold3', out_dir=out_dir, norm=True)

print("done")


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

images = np.load('dataset/pannuke/fold_1/images/images.npy', allow_pickle=True, mmap_mode='r')
masks = np.load('dataset/pannuke/fold_1/masks/masks.npy', allow_pickle=True, mmap_mode='r')

print(images.shape)
print(masks.shape)

# This function is modified from https://github.com/meszlili96/PanNukeChallenge.git

out_dir = "dataset/pannuke/hover_net_format"

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
"""