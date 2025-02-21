
import sys
import logging
import os

import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 300

# adding the project root folder
sys.path.append('../')
from tiatoolbox.utils.visualization import overlay_prediction_contours
from misc.utils import cropping_center

from net_desc import HoVerNetConic

# Random seed for deterministic
SEED = 5
NUM_TYPES = 7  # The number of nuclei types (+1 for background)

DATA_DIR = '/rds/user/mf774/hpc-work/part_II_project/lizard/hover_net-conic/exp_output/local/data/'
OUT_DIR = '/rds/user/mf774/hpc-work/part_II_project/lizard/hover_net-conic/exp_output/local/infer/'
PRETRAINED = '/rds/user/mf774/hpc-work/part_II_project/lizard/hover_net-conic/pretrained/hovernet-conic.pth'

pretrained = torch.load(PRETRAINED, map_location=torch.device('cpu'))
model = HoVerNetConic(num_types=NUM_TYPES)
model.load_state_dict(pretrained)

# Load necessary data
splits = joblib.load(f'{OUT_DIR}/splits.dat')
valid_indices = splits[0]['valid']  # Using first split by default

output_file = f'{OUT_DIR}/raw/file_map.dat'
output_info = joblib.load(output_file)

# Function Definitions
def process_segmentation(np_map, hv_map, tp_map):
    np_map = cv2.resize(np_map, (0, 0), fx=2.0, fy=2.0)
    hv_map = cv2.resize(hv_map, (0, 0), fx=2.0, fy=2.0)
    tp_map = cv2.resize(tp_map, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    inst_map = model._proc_np_hv(np_map[..., None], hv_map)
    inst_dict = model._get_instance_info(inst_map, tp_map)
    type_map = np.zeros_like(inst_map)
    inst_type_colours = np.array([[v['type']] * 3 for v in inst_dict.values()])
    type_map = overlay_prediction_contours(type_map, inst_dict, line_thickness=-1, inst_colours=inst_type_colours, draw_dot=None)
    pred_map = np.dstack([inst_map, type_map])
    pred_map = cv2.resize(pred_map, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    return pred_map

def process_composition(pred_map):
    pred_map = cropping_center(pred_map, [224, 224])
    inst_map = pred_map[..., 0]
    type_map = pred_map[..., 1]
    uid_list = np.unique(inst_map)[1:]
    
    if len(uid_list) < 1:
        return np.zeros(NUM_TYPES)
    
    uid_types = [np.unique(type_map[inst_map == uid]) for uid in uid_list]
    type_freqs_ = np.unique(uid_types, return_counts=True)
    type_freqs = np.zeros(NUM_TYPES)
    type_freqs[type_freqs_[0]] = type_freqs_[1]
    return type_freqs

# Post-processing
semantic_predictions = []
composition_predictions = []
print("got to line 214")
print(len(output_info))

for input_file, output_root in tqdm(output_info):
    np_map = np.load(f'{output_root}.raw.0.npy')
    hv_map = np.load(f'{output_root}.raw.1.npy')
    tp_map = np.load(f'{output_root}.raw.2.npy')

    pred_map = process_segmentation(np_map, hv_map, tp_map)
    type_freqs = process_composition(pred_map)
    semantic_predictions.append(pred_map)
    composition_predictions.append(type_freqs)

semantic_predictions = np.array(semantic_predictions)
composition_predictions = np.array(composition_predictions)

# Saving results
np.save(f'{OUT_DIR}/valid_pred.npy', semantic_predictions)

TYPE_NAMES = ["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"]
df = pd.DataFrame(composition_predictions[:, 1:].astype(np.int32))
df.columns = TYPE_NAMES
df.to_csv(f'{OUT_DIR}/valid_pred_cell.csv', index=False)

# Load ground truth composition
df = pd.read_csv(f'{DATA_DIR}/counts.csv')
true_compositions = df.to_numpy()[valid_indices]
df = pd.DataFrame(true_compositions.astype(np.int32))
df.columns = TYPE_NAMES
df.to_csv(f'{OUT_DIR}/valid_true_cell.csv', index=False)

semantic_true = np.load(f'{OUT_DIR}/valid_true.npy')
semantic_pred = np.load(f'{OUT_DIR}/valid_pred.npy')

# Visualization
PERCEPTIVE_COLORS = [
    (0, 0, 0), (255, 165, 0), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (0, 0, 255), (255, 255, 0)
]
np.random.seed(SEED)
selected_indices = np.random.choice(len(valid_indices), 4)

for idx in selected_indices:
    img = cv2.imread(output_info[idx][0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    inst_map = semantic_pred[idx][..., 0]
    type_map = semantic_pred[idx][..., 1]
    pred_inst_dict = model._get_instance_info(inst_map, type_map)
    
    inst_map = semantic_true[idx][..., 0]
    type_map = semantic_true[idx][..., 1]
    true_inst_dict = model._get_instance_info(inst_map, type_map)
    
    inst_type_colours = np.array([PERCEPTIVE_COLORS[v['type']] for v in true_inst_dict.values()])
    overlaid_true = overlay_prediction_contours(img, true_inst_dict, inst_colours=inst_type_colours, line_thickness=1)
    
    inst_type_colours = np.array([PERCEPTIVE_COLORS[v['type']] for v in pred_inst_dict.values()])
    overlaid_pred = overlay_prediction_contours(img, pred_inst_dict, inst_colours=inst_type_colours, line_thickness=1)
    
    def save_images(img, overlaid_true, overlaid_pred, filename='comparison.png'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].axis('off')
        
        axes[1].imshow(overlaid_true)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(overlaid_pred)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

save_images(img, overlaid_true, overlaid_pred, 'output.png')

'''
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(overlaid_true)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(overlaid_pred)
    plt.title('Prediction')
    plt.axis('off')
    plt.show()
'''