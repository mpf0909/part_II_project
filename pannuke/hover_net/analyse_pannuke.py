# adapted from example.ipynb
# import necessary packages
import sys
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import json
import openslide
import csv
from misc.wsi_handler import get_file_handler
from misc.viz_utils import visualize_instances_dict
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# load file paths for 3 datasets
pannuke_tile_path = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/pannuke/hover_net_format/val_png/"
pannuke_tile_mat_path = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/pannuke/hover_net_format/infer/mat/"

# List all files in the pannuke directory
pannuke_files = [f for f in os.listdir(pannuke_tile_path) if os.path.isfile(os.path.join(pannuke_tile_path, f))]
# Count the number of files
pannuke_file_count = len(pannuke_files)

# load pannuke images
pannuke_image_list = glob.glob(pannuke_tile_path + '*.png')
pannuke_image_list.sort()
nuc_types = {"0": 0,
              "1": 0,
              "2": 0,
              "3": 0,
              "4": 0,
              "5": 0}

pannuke_per_image = []

# count number of each cell type in pannuke images
for i in range(len(pannuke_image_list)):
    itr_dict = {"0": 0,
              "1": 0,
              "2": 0,
              "3": 0,
              "4": 0,
              "5": 0}
    pannuke_image_file = pannuke_image_list[i] 
    basename = os.path.basename(pannuke_image_file)
    pannuke_image_ext = basename.split('.')[-1]
    basename = basename[:-(len(pannuke_image_ext)+1)]
    result_mat = sio.loadmat(pannuke_tile_mat_path + basename + '.mat')
    inst_map = result_mat['inst_map'] 
    inst_type = result_mat['inst_type'] 
    for j in inst_type:
        nuc_types[str(j[0])] +=1
        itr_dict[str(j[0])] +=1
    pannuke_per_image.append(itr_dict)

normalised_nuc_types = {key: value / pannuke_file_count for key, value in nuc_types.items()}

# labels for barcharts
labels = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Non-Neoplastic Epithelial"]

plt.bar(labels, normalised_nuc_types.values())
plt.xlabel("Nucleus Types")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Count")
plt.title("Average Number of Nuclear Types Per Pannuke Disease Image")
plt.savefig("summary_images/pannuke_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()

# need to standardise for amount of tissue
# need to standardise for number of images