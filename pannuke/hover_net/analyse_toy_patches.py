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
coeliac_tile_path = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/toy_data_40x/train_patches/patches_2048_2048/addenbrookes/coeliac/TB21.00481/TB21.00481. A1.svs/mag_40.0/"
coeliac_tile_mat_path = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/toy_data_40x/train_patches/patches_2048_2048/addenbrookes/coeliac/TB21.00481/TB21.00481. A1.svs/infer/mat/"

normal_tile_path = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/toy_data_40x/train_patches/patches_2048_2048/addenbrookes/normal/TB21.00638/TB21.00638.svs/mag_40.0/"
normal_tile_mat_path = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/toy_data_40x/train_patches/patches_2048_2048/addenbrookes/normal/TB21.00638/TB21.00638.svs/infer/mat/"

other_tile_path = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/toy_data_40x/train_patches/patches_2048_2048/addenbrookes/normal/TB21.00658/TB21.00658.svs/mag_40.0/"
other_tile_mat_path = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/toy_data_40x/train_patches/patches_2048_2048/addenbrookes/normal/TB21.00658/TB21.00658.svs/infer/mat/"

# List all files in the coeliac directory
coeliac_files = [f for f in os.listdir(coeliac_tile_path) if os.path.isfile(os.path.join(coeliac_tile_path, f))]
# Count the number of files
coeliac_file_count = len(coeliac_files)

# List all files in the normal directory
normal_files = [f for f in os.listdir(normal_tile_path) if os.path.isfile(os.path.join(normal_tile_path, f))]
# Count the number of files
normal_file_count = len(normal_files)

other_files = [f for f in os.listdir(other_tile_path) if os.path.isfile(os.path.join(other_tile_path, f))]
other_file_count = len(other_files)
print("coeliac file count", coeliac_file_count)
print("normal file count", normal_file_count)
print("other file count", other_file_count)

# load coeliac images
coeliac_image_list = glob.glob(coeliac_tile_path + '*.png')
coeliac_image_list.sort()
coeliac_nuc_types = {"0": 0,
              "1": 0,
              "2": 0,
              "3": 0,
              "4": 0,
              "5": 0}

# load normal images
normal_image_list = glob.glob(normal_tile_path + '*.png')
normal_image_list.sort()
normal_nuc_types = {"0": 0,
              "1": 0,
              "2": 0,
              "3": 0,
              "4": 0,
              "5": 0}

# load other images
other_image_list = glob.glob(other_tile_path + '*.png')
other_image_list.sort()
other_nuc_types = {"0": 0,
              "1": 0,
              "2": 0,
              "3": 0,
              "4": 0,
              "5": 0}

coeliac_per_image = []
normal_per_image = []
other_per_image = []

# count number of each cell type in coeliacs images
for i in range(len(coeliac_image_list)):
    itr_dict = {"0": 0,
              "1": 0,
              "2": 0,
              "3": 0,
              "4": 0,
              "5": 0}
    coeliac_image_file = coeliac_image_list[i] 
    basename = os.path.basename(coeliac_image_file)
    coeliac_image_ext = basename.split('.')[-1]
    basename = basename[:-(len(coeliac_image_ext)+1)]
    result_mat = sio.loadmat(coeliac_tile_mat_path + basename + '.mat')
    inst_map = result_mat['inst_map'] 
    inst_type = result_mat['inst_type'] 
    for j in inst_type:
        coeliac_nuc_types[str(j[0])] +=1
        itr_dict[str(j[0])] +=1
    coeliac_per_image.append(itr_dict)

for i in range(len(normal_image_list)):
    itr_dict = {"0": 0,
              "1": 0,
              "2": 0,
              "3": 0,
              "4": 0,
              "5": 0}
    normal_image_file = normal_image_list[i] 
    basename = os.path.basename(normal_image_file)
    normal_image_ext = basename.split('.')[-1]
    basename = basename[:-(len(normal_image_ext)+1)]
    # get the corresponding `.mat` file 
    result_mat = sio.loadmat(normal_tile_mat_path + basename + '.mat')
    inst_map = result_mat['inst_map'] 
    inst_type = result_mat['inst_type'] 
    # let's inspect the inst_type output
    # print(np.unique(inst_type))
    for j in inst_type:
        normal_nuc_types[str(j[0])] +=1
        itr_dict[str(j[0])] +=1
    normal_per_image.append(itr_dict)

for i in range(len(other_image_list)):
    itr_dict = {"0": 0,
              "1": 0,
              "2": 0,
              "3": 0,
              "4": 0,
              "5": 0}
    other_image_file = other_image_list[i] 
    basename = os.path.basename(other_image_file)
    other_image_ext = basename.split('.')[-1]
    basename = basename[:-(len(other_image_ext)+1)]
    # get the corresponding `.mat` file 
    result_mat = sio.loadmat(other_tile_mat_path + basename + '.mat')
    inst_map = result_mat['inst_map'] 
    inst_type = result_mat['inst_type'] 
    # let's inspect the inst_type output
    # print(np.unique(inst_type))
    for j in inst_type:
        other_nuc_types[str(j[0])] +=1
        itr_dict[str(j[0])] +=1
    other_per_image.append(itr_dict)

# Add values for the same keys, ensuring numerical order
sorted_keys = sorted(normal_nuc_types.keys() | other_nuc_types.keys(), key=int)  # Sort keys numerically
normal_tot_nuc_types = {key: normal_nuc_types.get(key, 0) + other_nuc_types.get(key, 0) for key in sorted_keys}

normalised_normal_tot_nuc_types = {key: value / (normal_file_count+other_file_count) for key, value in normal_tot_nuc_types.items()}
normalised_coeliac_nuc_types = {key: value / coeliac_file_count for key, value in coeliac_nuc_types.items()}
normalised_normal_nuc_types = {key: value / normal_file_count for key, value in normal_nuc_types.items()}
normalised_other_nuc_types = {key: value / other_file_count for key, value in other_nuc_types.items()}

# labels for barcharts
labels = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Non-Neoplastic Epithelial"]

plt.bar(labels, normalised_coeliac_nuc_types.values())
plt.xlabel("Nucleus Types")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Count")
plt.title("Average Number of Nuclear Types Per Coeliac Disease Image")
plt.savefig("summary_images/coeliac_disease_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()

plt.bar(labels, normalised_normal_nuc_types.values())
plt.xlabel("Nucleus Types")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Count")
plt.title("Average Number of Nuclear Types Per Normal v1 Image")
plt.savefig("summary_images/normal_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()

plt.bar(labels, normalised_other_nuc_types.values())
plt.xlabel("Nucleus Types")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Count")
plt.title("Average Number of Nuclear Types Per Normal v2 Image")
plt.savefig("summary_images/other_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()

plt.bar(labels, normalised_normal_tot_nuc_types.values())
plt.xlabel("Nucleus Types")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Count")
plt.title("Average Number of  Nuclear Types Per Normal Image")
plt.savefig("summary_images/tot_normal_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    coeliac_values = [d[str(i)] for d in coeliac_per_image]
    normal_values = [d[str(i)] for d in normal_per_image]
    other_values = [d[str(i)] for d in other_per_image]
    all_normal_values = normal_values + other_values
    all_values = [coeliac_values, all_normal_values]

    a = f_oneway(coeliac_values, all_normal_values)
    print(a)

    ax = axes[i]
    ax.boxplot(all_values, labels = ["coeliac", "Normal"])
    ax.set_title(f"Number of {labels[i]} Nuclei Per Image")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("summary_images/boxplots.png", dpi=300)
plt.close()

# need to standardise for amount of tissue
# need to standardise for number of images

# Add labels
coeliac_labels = [1] * len(coeliac_per_image)
normal_labels = [0] * len(normal_per_image)
other_labels = [0] * len(other_per_image)

# Combine data
all_data = coeliac_per_image + normal_per_image + other_per_image
all_labels = coeliac_labels + normal_labels + other_labels

# Convert to feature matrix and label vector
X = np.array([[d[str(i)] for i in range(6)] for d in all_data])  # Features
y = np.array(all_labels)  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Save to CSV
with open("classification_report.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])  # Header
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):  # Only write rows for the class-specific metrics
            writer.writerow([label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])