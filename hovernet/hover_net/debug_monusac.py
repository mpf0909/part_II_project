from PIL import Image  
import numpy as np  
# Open the image file  
img = Image.open('dataset/cpm17/test/Images/image_00.png')  
# Convert the PIL image into a NumPy array using numpy.array()  
numpydata = np.array(img)  
# Check the type and shape  
print(type(numpydata))  
print(numpydata.shape)  
"""

import numpy as np
from matplotlib import pyplot as plt

# Load the dataset
images = np.load('dataset/monusac/fold_1/images/images.npy', mmap_mode='r')
types = np.load('dataset/monusac/fold_1/images/types.npy', mmap_mode='r')
masks = np.load('dataset/monusac/fold_1/masks/masks.npy', mmap_mode='r')

# Ensure the data shape is correct
print("Dataset shape:", images.shape)  # Should output (2656, 256, 256, 3)

# Select the first image
first_image = images[0]

# Check the value range of the image
print("Image dtype:", first_image.dtype)
print("Image min value:", first_image.min())
print("Image max value:", first_image.max())

# If the values exceed [0, 1] or are floats, rescale them to [0, 255] and convert to integers
if first_image.dtype in [np.float32, np.float64] and first_image.max() > 1.0:
    first_image = (first_image / first_image.max() * 255).astype(np.uint8)

# Save the corrected image
plt.imshow(first_image)
plt.title("First Image")
plt.axis('off')  # Hide axes for better visualization
plt.savefig("first_image.png")
print("Image saved as 'first_image.png'.")

np.set_printoptions(threshold=np.inf)
print(masks[0])

"""

"""
import scipy.io
import pdb; pdb.set_trace()
mat = scipy.io.loadmat('/rds/user/mf774/hpc-work/hovernet/hover_net/dataset/cpm17/test/Labels/image_01.mat')
print(mat)
"""


