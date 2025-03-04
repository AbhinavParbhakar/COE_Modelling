import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os

def convert_images_to_numpy(path:str)->np.ndarray:
    """
    Converts the images stored under the given parameter path into one large ``numpy.ndarray`` object.
    
    Parameters
    ----------
    
    path : ``str``
        Path of the folder containing images
    
    Returns
    -------
    
    ``numpy.ndarray``
        The numpy array containing the images
    """
    absolute_path = os.path.abspath(path)
    nested_path_generator = os.walk(absolute_path)
    image_paths = []
    images_numpy = []
    
    for dirpath,dirnames,filenames in nested_path_generator:
        image_paths.extend([os.path.join(dirpath,name) for name in filenames])       
    
    for image_path in image_paths:
        images_numpy.append(np.array(Image.open(image_path)).reshape((3,256,256)))
    
    return_numpy_array = np.array(images_numpy)
    return return_numpy_array

def generate_target_values_numpy(excel_path:str,image_path:str)->np.ndarray:
    """
    Create a ``numpy.ndarray`` containing parallel matched regression values for the images
    included in the provided folder, matching based on name of image to the row in the excel file.
    
    Parameters
    ----------
    excel_path : ``str``
        The path of the excel file to be opened. Contains the regression values per Study ID.
    image_path : ``str``
        The path of the folder containing the images. Labled with study IDs.
    
    Returns
    -------
    ``numpy.ndarray``
        Numpy array corresponding to the target values matched with given images. 
    """
    absolute_path = os.path.abspath(image_path)
    df = pd.read_excel(excel_path)
    nested_path_generator = os.walk(absolute_path)
    image_id_array = []
    regression_values = []
    
    for dirpath,dirnames,filenames in nested_path_generator:
        image_id_array.extend([name.split('.')[0] for name in filenames])
        
    for image_id in image_id_array:
        values = df['AAWDT'][df['Estimation point'] == int(image_id)].values
        if values.shape[0] != 1:
            print(image_id)
        regression_values.append(values)
    

if __name__ == "__main__":
    image_path = "./data/images"
    excel_path = './data/excel_files/updated_values.xlsx'
    images_ndarray = convert_images_to_numpy(image_path)
    aawdt_ndarray = generate_target_values_numpy(excel_path,image_path)