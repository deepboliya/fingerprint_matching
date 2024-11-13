import os
import cv2
import shutil
import numpy as np

def convert_images_to_grayscale(input_dir, output_dir):
    # Copy the directory structure to the new output directory

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove if already exists to avoid conflicts
    os.makedirs(output_dir)

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            input_file_path = os.path.join(subdir, file)
            
            # Define the corresponding output file path
            relative_path = os.path.relpath(subdir, input_dir)  # Get the relative path
            output_subdir = os.path.join(output_dir, relative_path)
            output_file_path = os.path.join(output_subdir, file)

            # Ensure output sub-directory exists
            os.makedirs(output_subdir, exist_ok=True)

            try:
                # Read the image using OpenCV
                img = cv2.imread(input_file_path)
                if img is None:
                    print(f"Skipping non-image file: {input_file_path}")
                    continue
                
                # Convert to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                eq_img = cv2.equalizeHist(gray_img)
                AMT_image = cv2.adaptiveThreshold(eq_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)  
                print(np.shape(AMT_image))
                # Save the grayscale image to the same path in the output directory structure
                cv2.imwrite(output_file_path, AMT_image)
            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")

# Replace 'original_main_directory' with the name of your main input directory
# Replace 'new_main_directory' with the desired name for the output directory
input_dir = "/home/skully/Acads/ee678-wavelets/final_project_testing/PolyU/processed_contactless_2d_fingerprint_images"
output_dir = "/home/skully/Acads/ee678-wavelets/final_project_testing/PolyU/amt_contactless_2d_fingerprint_images"

# Convert images to grayscale and save in the same structure under new_main_directory
convert_images_to_grayscale(input_dir, output_dir)
