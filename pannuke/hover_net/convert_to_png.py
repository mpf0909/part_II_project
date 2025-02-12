import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def convert_npy_to_rgb_image(input_folder, output_folder, image_format='png'):
    """
    Converts all .npy files in the input_folder to RGB images and saves them in the output_folder.

    Parameters:
    - input_folder (str): Path to the folder containing .npy files.
    - output_folder (str): Path to the folder to save the converted images.
    - image_format (str): Image format to save as ('png' or 'jpg').
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    if not npy_files:
        print("No .npy files found in the input folder.")
        return

    for npy_file in npy_files:
        try:
            # Load the .npy file
            data = np.load(os.path.join(input_folder, npy_file))

            # Extract the RGB channels (indices 0:2)
            if len(data.shape) == 3 and data.shape[2] >= 3:
                rgb_data = data[:, :, 0:3]
            else:
                raise ValueError(f"Invalid data shape {data.shape} for RGB conversion")
            
            # Normalize data to [0, 1] if necessary
            if rgb_data.max() > 1:
                rgb_data = rgb_data / 255.0

            # Save the RGB image
            output_file = os.path.join(output_folder, f"{os.path.splitext(npy_file)[0]}.{image_format}")
            mpimg.imsave(output_file, rgb_data)

            print(f"Converted {npy_file} to {output_file}")

        except Exception as e:
            print(f"Failed to convert {npy_file}: {e}")

if __name__ == "__main__":
    # Set the paths
    input_folder = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/hover_net_format/val/"  # Replace with your input folder
    output_folder = "/rds/user/mf774/hpc-work/pannuke/hover_net/dataset/hover_net_format/val_png/"    # Replace with your output folder

    # Call the conversion function
    convert_npy_to_rgb_image(input_folder, output_folder, image_format='png')
