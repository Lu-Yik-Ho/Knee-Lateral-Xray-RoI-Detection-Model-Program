import torch
import pydicom
import os
import glob
import numpy as np
import json
import torch.utils.data
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import ultralytics
import cv2
os.environ['YOLO_VERBOSE'] = 'false'
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import yaml


def apply_clahe(file, output_dir):

    """
    **Applies CLAHE normalisation on an image.**

    Parameters:

        file (png): The path to the image file.

        output_dir (str): The directory for storing output files.

    Returns:
    
        output (png): A CLAHE normalised image stored in the specified output directory.
    """

    # 
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)

    file_name = os.path.splitext(file)[0]
    output_path = os.sep.join([output_dir, file_name.split(os.sep)[-1] + ".png"])
    cv2.imwrite(output_path,clahe_img)

    return None


def crop_img(image_file, out_dir, x_min, y_min, box_width, box_height):

    """
    **Crops an image based on the provided bounding box information.**

    Parameters:

        image_file (png): The path to the image file.

        output_dir (str): The directory for storing output files.

        x_min (float): The x-value of the top left corner.

        y_min (float): The y-value of the top left corner.

        box_width (float): The width of the bounding box.

        box_height (float): The height of the bounding box.

    Returns:
    
        output (png): A cropped image stored in the specified output directory.
    """

    img = Image.open(image_file)

    cropped_img = transforms.functional.crop(img,y_min,x_min,box_height,box_width)

    file_name = os.path.splitext(image_file)[0]
    output_file = os.sep.join([out_dir, file_name.split(os.sep)[-1]+".png"])

    print(output_file)
    with open(str(output_file), 'wb') as png_file:
      cropped_img.save(png_file, "PNG")


def dicom_to_png(file):

    """
    **Retrieves a PNG image from a DICOM file.**

    Parameters:

        file (png): The path to the DICOM file.

    Returns:
    
        output (png): An image stored in the directory of the original DICOM file.
    """
    # read dicom image
    ds = pydicom.dcmread(file)
    print(ds)
    
    # get image data
    shape = ds.pixel_array.shape

    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Write the PNG file
    output_filename = os.path.splitext(file)[0] + '.png'
    with open(output_filename, 'wb') as png_file:
        Image.fromarray(image_2d_scaled).save(png_file, "PNG")


def flip_png(file):
    
    """
    **Flips an image horizontally.**

    Parameters:

        file (png): The path to the image file.

    Returns:
    
        output (png): A flipped image stored in the directory of the original image.
    """

    # Open the image file
    img = Image.open(file)

    # Convert to PyTorch Tesnor (Needed for transforms module)
    img_tensor = transforms.functional.pil_to_tensor(img)

    # Horizontal Flip
    flipped_img_tensor = transforms.functional.hflip(img_tensor)

    # Convert to PIL image
    flipped_img = transforms.functional.to_pil_image(flipped_img_tensor)
    

    # Write the PNG files
    base_filename = os.path.splitext(file)[0]
    with open(base_filename + '_F.png', 'wb') as png_file:
      flipped_img.save(png_file, "PNG")


def normal_img(file, output_img = False, output_dir = None):

    """
    **Applies normalisation on an image.**

    Parameters:

        file (png): The path to the image file.

        output_img (bool): Whether to output the image. Default: False.

        output_dir (str): The directory for storing output files.        

    Returns:
    
        output (png): A normalised image stored in the specified output directory.
    """

    # Opens image file as PIL
    img = Image.open(file)

    # Converts PIL data to PyTorch tensor
    img_tensor = transforms.functional.pil_to_tensor(img)

    # Find the highest value of the image (tensor)
    max_value = torch.max(img_tensor)
    min_value = torch.min(img_tensor)
    print(max_value, min_value)
    min_tensor = torch.ones(img_tensor.size()) * min_value

    # Normalise image
    img_tensor_normal = (img_tensor - min_tensor) / (max_value - min_value)

    # Convert tensor to PIL data
    normal_img = transforms.functional.to_pil_image(img_tensor_normal)

    # Output:
    if output_img:
       if output_img == True:
            base_filename = os.path.splitext(file)[0]
            temp_path = base_filename.split(os.sep)[:]
            output_path = os.path.join(output_dir,temp_path[-1])
            with open(str(output_path) + "_N.png", 'wb') as png_file:
                normal_img.save(png_file, "PNG")


def predict_img(model, image_file):
        
    """
    **Runs a prediction on an image with the loaded model.**

    Parameters:

        model (model): <class 'ultralytics.models.yolo.model.YOLO'> Loaded model.

        file (png): The path to the image file.

    Returns:
    
        output (ultralytics.engine.results.Results): Result data of the prediction.
    """

    results = model.predict(source=image_file)
    return results[0]


def split_png_lateral(file, num, crop_side = "Left"):
    
    """
    **Crops a specified amount from the left or right of an image.**

    Parameters:

        file (png): The path to the image file.

        num (float): The percentage of the image to be cropped.

        crop_side (str): Default: "Left"

            "Left": Crops from the left.

            "Right: Crops from the right.

    Returns:
    
        output (png): A cropped image stored in the directory of the original image.
    """
    # Open the image file
    img = Image.open(file)
    # Convert the image to numpy array
    img_array = np.array(img)

    # Split the image into left and right halves
    
    if crop_side == "Left":
        split_width = int(img_array.shape[1] * num)
        required_half = img_array[:, split_width:]
    elif crop_side == "Right":
        split_width = int(img_array.shape[1] * (1-num))
        required_half = img_array[:, :split_width]
    else:
        print("Unexpected crop_side input. Image will be cropped from the left by default.")
        split_width = int(img_array.shape[1] * num)
        required_half = img_array[:, split_width:]

    # Write the PNG files
    base_filename = os.path.splitext(file)[0]
    with open(base_filename + '_C.png', 'wb') as png_file:
        Image.fromarray(required_half).save(png_file, "PNG")




