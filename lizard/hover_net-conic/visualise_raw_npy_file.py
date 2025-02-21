import numpy as np
import matplotlib.pyplot as plt
import joblib
import cv2
import random
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
from net_desc import HoVerNetConic

OUT_DIR = '/rds/user/mf774/hpc-work/part_II_project/lizard/hover_net-conic/toy_data_20x_inference/normal/225363019'
NUM_TYPES = 7
model = HoVerNetConic(num_types=NUM_TYPES)

# semantic_true = np.load(f'{OUT_DIR}/valid_true.npy')
semantic_pred = np.load(f'{OUT_DIR}/valid_pred.npy')

output_file = f'{OUT_DIR}/raw/file_map.dat'
output_info = joblib.load(output_file)

PERCEPTIVE_COLORS = [
    (  0,   0,   0),
    (255, 165,   0),
    (  0, 255,   0),
    (255,   0,   0),
    (  0, 255, 255),
    (  0,   0, 255),
    (255, 255,   0),
]

# Select a few random indices to visualize
# selected_image_index = 2300

def visualize_prediction(idx):
    """ Load and visualize segmentation results for a given index."""
    img = cv2.imread(output_info[idx][0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    inst_map = semantic_pred[idx][..., 0]
    type_map = semantic_pred[idx][..., 1]
    pred_inst_dict = model._get_instance_info(inst_map, type_map)
    
    inst_type_colours = np.array([
        PERCEPTIVE_COLORS[v['type']]
        for v in pred_inst_dict.values()
    ])
    overlaid_pred = overlay_prediction_contours(
        img, pred_inst_dict,
        inst_colours=inst_type_colours,
        line_thickness=1
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(overlaid_pred)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis('off')
    
    plt.savefig(f'{OUT_DIR}/visualization_{idx}.png', bbox_inches='tight')
    plt.close(fig)

# Visualize selected images
MAX_INDEX = 8800
random_indices = random.sample(range(MAX_INDEX), 10)
for idx in random_indices:
    visualize_prediction(idx)