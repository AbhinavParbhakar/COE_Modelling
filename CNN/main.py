import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset, TensorDataset
from PIL import Image
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rotate
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import subprocess
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import sys
import datetime
import math

# Check if running on Kaggle and install dependencies if not already installed
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    print("Started download for pip packages")
    import torch
    version = torch.__version__.split("+")[0]  # Extracts version without CUDA/CPU suffix
    install_cmd = [
        sys.executable, "-m", "pip", "install", "bayesian-optimization"]
    subprocess.check_call(install_cmd)

from bayes_opt import BayesianOptimization

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # nn.Linear(256,325),
            # nn.BatchNorm1d(325),
            # nn.ReLU(),
            
        )
    
    def forward(self,x):
        x = self.net(x)
        
        return x
    
class CoarseImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.coarse_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=10,stride=3),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=3,stride=3)
        )
        self.coarse_layer_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )
        self.coarse_layer_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )
        self.coarse_layer_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        
        self.coarse_layer_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        self.coarse_layer_upsample_3 = nn.Conv2d(in_channels=15,out_channels=30,kernel_size=1)
        self.coarse_layer_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )

        self.coarse_layer_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.coarse_layer_upsample_4 = nn.Conv2d(in_channels=30,out_channels=45,kernel_size=1)
        self.coarse_layer_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )

        self.coarse_layer_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=60,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        
        self.coarse_layer_upsample_5 = nn.Conv2d(in_channels=45,out_channels=60,kernel_size=1)
    
    def forward(self,coarse_input:torch.FloatTensor):
        # Process coarse input
        x1_layer_2_input = self.coarse_layer_1(coarse_input)
        
        x1 = self.coarse_layer_2_1(x1_layer_2_input)
        x1 = self.coarse_layer_2_2(x1)
        x1  = x1 + x1_layer_2_input
        x1 = self.max_pool(x1)
        
        x1_layer_3_input = self.coarse_layer_upsample_3(x1)
        x1 = self.coarse_layer_3_1(x1)
        x1 = self.coarse_layer_3_2(x1)
        x1 = x1 + x1_layer_3_input
        x1 = self.max_pool(x1)

        x1_layer_4_input = self.coarse_layer_upsample_4(x1)
        x1 = self.coarse_layer_4_1(x1)
        x1 = self.coarse_layer_4_2(x1)
        x1 = x1 + x1_layer_4_input
        x1 = self.max_pool(x1)
        
        x1_layer_5_input = self.coarse_layer_upsample_5(x1)
        x1 = self.coarse_layer_5_1(x1)
        x1 = self.coarse_layer_5_2(x1)
        x1 = x1 + x1_layer_5_input
        x1 = self.avg_pool(x1)
        
        return x1

class ThirtyTwoGranularImageModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.granular_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=7),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=3,stride=1)
        )

        self.granular_layer_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )

        self.granular_layer_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )

        self.granular_layer_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )

        self.granular_layer_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.granular_layer_upsample_3 = nn.Conv2d(in_channels=15,out_channels=30,kernel_size=1)
        self.granular_layer_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_upsample_4 = nn.Conv2d(in_channels=30,out_channels=45,kernel_size=1)
        self.granular_layer_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        self.granular_layer_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=60,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )

        self.granular_layer_upsample_5 = nn.Conv2d(in_channels=45,out_channels=60,kernel_size=1)

    def forward(self,granular_input:torch.FloatTensor):
        x2_layer_2_input = self.granular_layer_1(granular_input)

        x2 = self.granular_layer_2_1(x2_layer_2_input)
        x2 = self.granular_layer_2_2(x2)
        x2  = x2 + x2_layer_2_input
        x2 = self.max_pool(x2)

        x2_layer_3_input = self.granular_layer_upsample_3(x2)
        x2 = self.granular_layer_3_1(x2)
        x2 = self.granular_layer_3_2(x2)
        x2 = x2 + x2_layer_3_input
        x2 = self.max_pool(x2)

        x2_layer_4_input = self.granular_layer_upsample_4(x2)
        x2 = self.granular_layer_4_1(x2)
        x2 = self.granular_layer_4_2(x2)
        x2 = x2 + x2_layer_4_input
        x2 = self.avg_pool(x2)

        return self.flatten(x2)

class SixtyFourGranularImageModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.granular_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=7),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.granular_layer_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )

        self.granular_layer_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )
        
        self.granular_layer_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        
        self.granular_layer_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        self.granular_layer_upsample_3 = nn.Conv2d(in_channels=15,out_channels=30,kernel_size=1)
        self.granular_layer_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_upsample_4 = nn.Conv2d(in_channels=30,out_channels=45,kernel_size=1)
        self.granular_layer_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        self.granular_layer_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=60,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        
        self.granular_layer_upsample_5 = nn.Conv2d(in_channels=45,out_channels=60,kernel_size=1)
        
    def forward(self,granular_input:torch.FloatTensor):
        x2_layer_2_input = self.granular_layer_1(granular_input)
        
        x2 = self.granular_layer_2_1(x2_layer_2_input)
        x2 = self.granular_layer_2_2(x2)
        x2  = x2 + x2_layer_2_input
        x2 = self.max_pool(x2)
        
        x2_layer_3_input = self.granular_layer_upsample_3(x2)
        x2 = self.granular_layer_3_1(x2)
        x2 = self.granular_layer_3_2(x2)
        x2 = x2 + x2_layer_3_input
        x2 = self.max_pool(x2)

        x2_layer_4_input = self.granular_layer_upsample_4(x2)
        x2 = self.granular_layer_4_1(x2)
        x2 = self.granular_layer_4_2(x2)
        x2 = x2 + x2_layer_4_input
        x2 = self.avg_pool(x2)
        
        return self.flatten(x2)

class OneTwentyEightGranularImageModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.granular_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=7,stride=2),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.granular_layer_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )

        self.granular_layer_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )
        
        self.granular_layer_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        
        self.granular_layer_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        self.granular_layer_upsample_3 = nn.Conv2d(in_channels=15,out_channels=30,kernel_size=1)
        self.granular_layer_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_upsample_4 = nn.Conv2d(in_channels=30,out_channels=45,kernel_size=1)
        self.granular_layer_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        self.granular_layer_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=60,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        
        self.granular_layer_upsample_5 = nn.Conv2d(in_channels=45,out_channels=60,kernel_size=1)
        
    def forward(self,granular_input:torch.FloatTensor):
        x2_layer_2_input = self.granular_layer_1(granular_input)
        
        x2 = self.granular_layer_2_1(x2_layer_2_input)
        x2 = self.granular_layer_2_2(x2)
        x2  = x2 + x2_layer_2_input
        x2 = self.max_pool(x2)
        
        x2_layer_3_input = self.granular_layer_upsample_3(x2)
        x2 = self.granular_layer_3_1(x2)
        x2 = self.granular_layer_3_2(x2)
        x2 = x2 + x2_layer_3_input
        x2 = self.max_pool(x2)

        x2_layer_4_input = self.granular_layer_upsample_4(x2)
        x2 = self.granular_layer_4_1(x2)
        x2 = self.granular_layer_4_2(x2)
        x2 = x2 + x2_layer_4_input
        x2 = self.avg_pool(x2)
        
        return self.flatten(x2)

class FiveTwelveGranularImageModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.granular_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=7,stride=3),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.granular_layer_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )

        self.granular_layer_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )
        
        self.granular_layer_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        
        self.granular_layer_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        self.granular_layer_upsample_3 = nn.Conv2d(in_channels=15,out_channels=30,kernel_size=1)
        self.granular_layer_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_upsample_4 = nn.Conv2d(in_channels=30,out_channels=45,kernel_size=1)
        self.granular_layer_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        self.granular_layer_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=60,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        
        self.granular_layer_upsample_5 = nn.Conv2d(in_channels=45,out_channels=60,kernel_size=1)
        
    def forward(self,granular_input:torch.FloatTensor):
        x2_layer_2_input = self.granular_layer_1(granular_input)
        
        x2 = self.granular_layer_2_1(x2_layer_2_input)
        x2 = self.granular_layer_2_2(x2)
        x2  = x2 + x2_layer_2_input
        x2 = self.max_pool(x2)
        
        x2_layer_3_input = self.granular_layer_upsample_3(x2)
        x2 = self.granular_layer_3_1(x2)
        x2 = self.granular_layer_3_2(x2)
        x2 = x2 + x2_layer_3_input
        x2 = self.max_pool(x2)

        x2_layer_4_input = self.granular_layer_upsample_4(x2)
        x2 = self.granular_layer_4_1(x2)
        x2 = self.granular_layer_4_2(x2)
        x2 = x2 + x2_layer_4_input
        x2 = self.max_pool(x2)
        
        x2_layer_5_input = self.granular_layer_upsample_5(x2)
        x2 = self.granular_layer_5_1(x2)
        x2 = self.granular_layer_5_2(x2)
        x2 = x2 + x2_layer_5_input
        x2 = self.avg_pool(x2)
        
        return self.flatten(x2)

class TwoFiftySixGranularImageModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.granular_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=7,stride=2),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.granular_layer_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )

        self.granular_layer_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=15,kernel_size=3,padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU()
        )
        
        self.granular_layer_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        
        self.granular_layer_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=3,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU()          
        )
        self.granular_layer_upsample_3 = nn.Conv2d(in_channels=15,out_channels=30,kernel_size=1)
        self.granular_layer_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=45,kernel_size=3,padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU()
        )
        self.granular_layer_upsample_4 = nn.Conv2d(in_channels=30,out_channels=45,kernel_size=1)
        self.granular_layer_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=45,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        self.granular_layer_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=60,out_channels=60,kernel_size=3,padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU()
        )
        
        self.granular_layer_upsample_5 = nn.Conv2d(in_channels=45,out_channels=60,kernel_size=1)
        
    def forward(self,granular_input:torch.FloatTensor):
        x2_layer_2_input = self.granular_layer_1(granular_input)
        
        x2 = self.granular_layer_2_1(x2_layer_2_input)
        x2 = self.granular_layer_2_2(x2)
        x2  = x2 + x2_layer_2_input
        x2 = self.max_pool(x2)
        
        x2_layer_3_input = self.granular_layer_upsample_3(x2)
        x2 = self.granular_layer_3_1(x2)
        x2 = self.granular_layer_3_2(x2)
        x2 = x2 + x2_layer_3_input
        x2 = self.max_pool(x2)

        x2_layer_4_input = self.granular_layer_upsample_4(x2)
        x2 = self.granular_layer_4_1(x2)
        x2 = self.granular_layer_4_2(x2)
        x2 = x2 + x2_layer_4_input
        x2 = self.max_pool(x2)
        
        x2_layer_5_input = self.granular_layer_upsample_5(x2)
        x2 = self.granular_layer_5_1(x2)
        x2 = self.granular_layer_5_2(x2)
        x2 = x2 + x2_layer_5_input
        x2 = self.avg_pool(x2)
        
        return x2
   
class MultimodalFullModel(nn.Module):
    def __init__(self,granular_image_dimension=256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout()
        
        
        granular_size_dict = {
            256 : 540,
            512 : 540,
            128 : 405,
            64 : 405,
            32 : 405,
        }
        
        granular_model_dict = {
            256 : TwoFiftySixGranularImageModel(),
            512 : FiveTwelveGranularImageModel(),
            128 : OneTwentyEightGranularImageModel(),
            64 : SixtyFourGranularImageModel(),
            32 : ThirtyTwoGranularImageModel(),
        }

        self.key_dimension = 120
        self.value_dimension = 60
        
        
        self.param_key_dimension = 100
        self.param_value_dimension = 2
        
        
        coarse_image_size = 240
        parametric_data_size = 256
        # combination_size = granular_size_dict[granular_image_dimension] + coarse_image_size + parametric_data_size
        combination_size = granular_size_dict[granular_image_dimension] + parametric_data_size + coarse_image_size
        self.parametric_module = NN()
        self.coarse_module = CoarseImageModel()
        self.granular_module = granular_model_dict[granular_image_dimension]
        # self.aerial_module = granular_model_dict[256]
        self.fc1 = nn.Linear(in_features=1292,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=200)
        self.fc3 = nn.Linear(in_features=200,out_features=50)
        self.fc4 = nn.Linear(in_features=50,out_features=1)
        
        
        self.query_weight_granular = nn.Conv2d(in_channels=60,out_channels=self.key_dimension,kernel_size=1,bias=False)
        self.key_weight_granular = nn.Conv2d(in_channels=60,out_channels=self.key_dimension,kernel_size=1,bias=False)
        self.value_weight_granular = nn.Conv2d(in_channels=60,out_channels=self.value_dimension,kernel_size=1,bias=False)
        
        self.query_weight_coarse = nn.Conv2d(in_channels=60,out_channels=self.key_dimension,kernel_size=1,bias=False)
        self.key_weight_coarse = nn.Conv2d(in_channels=60,out_channels=self.key_dimension,kernel_size=1,bias=False)
        self.value_weight_coarse = nn.Conv2d(in_channels=60,out_channels=self.value_dimension,kernel_size=1,bias=False)
        
        self.query_weight_image = nn.Linear(in_features=self.value_dimension,out_features=self.key_dimension,bias=False)
        
        
        self.query_weight_param = nn.Linear(in_features=1,out_features=self.param_key_dimension,bias=False)
        self.key_weight_param = nn.Linear(in_features=1,out_features=self.param_key_dimension,bias=False)
        self.value_weight_param = nn.Linear(in_features=1,out_features=self.param_value_dimension,bias=False)


    def self_attention_coarse_image_embedding(self,coarse_image_embedding:torch.Tensor)->torch.Tensor:
        query = self.query_weight_coarse(coarse_image_embedding)
        key = self.key_weight_coarse(coarse_image_embedding)
        value = self.value_weight_coarse(coarse_image_embedding)  
        
        B, C, H, W = query.shape
        
        query = query.permute(0,2,3,1).reshape(B,-1, self.key_dimension)
        key = key.permute(0,2,3,1).reshape(B,-1, self.key_dimension)
        value = value.permute(0,2,3,1).reshape(B,-1, self.value_dimension)
        
        scores = torch.matmul(query,key.transpose(1,2)) / math.sqrt(self.key_dimension)
        weights = self.softmax(scores)
        return torch.matmul(weights,value).reshape(B,-1)
        

    def self_attention_granular_image_embedding(self,granular_image_embedding:torch.Tensor)->torch.Tensor:        
        # Implement self attention for  granular images
        
        query = self.query_weight_granular(granular_image_embedding)
        key = self.key_weight_granular(granular_image_embedding)
        value = self.value_weight_granular(granular_image_embedding)  
        
        B, C, H, W = query.shape
        
        query = query.permute(0,2,3,1).reshape(B,-1, self.key_dimension)
        key = key.permute(0,2,3,1).reshape(B,-1, self.key_dimension)
        value = value.permute(0,2,3,1).reshape(B,-1, self.value_dimension)
        
        scores = torch.matmul(query,key.transpose(1,2)) / math.sqrt(self.key_dimension)
        weights = self.softmax(scores)
        return torch.matmul(weights,value).reshape(B,-1)
    
    def self_attention_parameters(self,parameter_embedding:torch.Tensor)->torch.Tensor:
        # project parametric embedding into B N 1
        parameter_embedding = torch.unsqueeze(parameter_embedding,dim=-1)
        key = self.key_weight_param(parameter_embedding)
        value = self.value_weight_param(parameter_embedding)
        query = self.query_weight_param(parameter_embedding)
        B = query.shape[0]
        
        scores = torch.matmul(query,key.transpose(1,2)) / math.sqrt(self.param_key_dimension)
        weights = self.softmax(scores)
        print(weights.shape)
        print(value.shape)
        
        return torch.matmul(weights,value).reshape(B,-1)

    def cross_attention_coarse_granular(self,granular_image_embedding:torch.Tensor,coarse_image_embedding:torch.Tensor,flatten=True)->torch.Tensor:
        query = self.query_weight(granular_image_embedding)
        key = self.key_weight(coarse_image_embedding)
        value = self.value_weight(coarse_image_embedding)  
        
        B, C, H, W = query.shape
        
        query = query.permute(0,2,3,1).reshape(B,-1, self.key_dimension) # B 9 10
        key = key.permute(0,2,3,1).reshape(B,-1, self.key_dimension) # B 4 10
        value = value.permute(0,2,3,1).reshape(B,-1, self.value_dimension) # B 4 60
        
        scores = torch.matmul(query,key.transpose(1,2)) / math.sqrt(self.key_dimension) # B 9 4
        weights = self.softmax(scores) # B 9 4
        
        if flatten:
            return torch.matmul(weights,value).reshape(B,-1) # B 540
        else:
            return torch.matmul(weights,value) # B 9 60
    
    def cross_attention_images_parameter(self,images_embedding:torch.Tensor,parametric_embedding:torch.Tensor,flatten=True)->torch.Tensor:
        
        # project parametric embedding into B N 1
        parametric_embedding = torch.unsqueeze(parametric_embedding,dim=-1)
        key = self.key_weight_param(parametric_embedding)
        value = self.value_weight_param(parametric_embedding)
        
        query = self.query_weight_image(images_embedding)
        B, L, D = query.shape
        
        scores = torch.matmul(query,key.transpose(1,2)) / math.sqrt(self.key_dimension) # B 9 4
        weights = self.softmax(scores) # B 9 4

        if flatten:
            return torch.matmul(weights,value).reshape(B,-1) # B 890
        else:
            return torch.matmul(weights,value) # B 9 90
        
        
    def forward(self,coarse_input:torch.FloatTensor,granular_input:torch.FloatTensor,parametric_input:torch.FloatTensor):
        coarse_image_embedding = self.coarse_module(coarse_input)
        granular_image_embedding = self.granular_module(granular_input)
        # aerial_image_embedding = self.aerial_module(aerial_input)
        
        parametric_embeddings = self.parametric_module(parametric_input)
        parametric_embeddings = self.relu(parametric_embeddings)
        
        # image_embeddings = self.cross_attention_coarse_granular(granular_image_embedding,coarse_image_embedding,flatten=False)
        # combination = self.cross_attention_images_parameter(images_embedding=image_embeddings,parametric_embedding=parametric_embeddings,flatten=True)
        
        # query_key_output = self.softmax(torch.mm(input=query,mat2=key.T) / math.sqrt(self.key_dimension))
        # output = torch.mm(query_key_output,value)
        
        coarse_image_embedding = self.self_attention_coarse_image_embedding(coarse_image_embedding)
        granular_image_embedding = self.self_attention_granular_image_embedding(granular_image_embedding)
        parametric_embeddings = self.self_attention_parameters(parametric_embeddings)
        
        print(coarse_image_embedding.shape,granular_image_embedding.shape,parametric_embeddings.shape)
        
        combination = torch.cat(tensors=(coarse_image_embedding,granular_image_embedding,parametric_embeddings),dim=1)
        
        # combination = self.dropout(combination)
        
        output = self.fc1(combination)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.fc4(output)
        
        return output
    
class ImageDataset(Dataset):
    def __init__(self,coarse_images:np.ndarray,granular_images:np.ndarray,targets:np.ndarray,train=False):        
        if train:
            # Training data augmented
            self.coarse_images = self.generate_tensor_from_numpy(coarse_images,train=train)
            self.granular_images = self.generate_tensor_from_numpy(granular_images,train=train)
            self.y = torch.from_numpy(targets.repeat(4,axis=0)).float()
        else:
            self.coarse_images = torch.from_numpy(coarse_images).permute(0,3,1,2) / 255
            self.granular_images = torch.from_numpy(granular_images).permute(0,3,1,2) / 255
            self.y = torch.from_numpy(targets).float()
        
    def generate_tensor_from_numpy(self,array:np.ndarray,train=False)->torch.FloatTensor:
        images = []
        for image in array:
            images.append(ToTensor()(image).unsqueeze(dim=0))
            if train:
                # Generate augmented rotations at intervals of 90 degrees
                image1 = rotate(img=ToTensor()(image),angle=90).unsqueeze(dim=0)
                image2 = rotate(img=ToTensor()(image),angle=180).unsqueeze(dim=0)
                image3 = rotate(img=ToTensor()(image),angle=270).unsqueeze(dim=0)
                images.extend([image1,image2,image3])
        
        return torch.cat(images,dim=0)
    
    def __len__(self):
        return self.coarse_images.shape[0]

    def __getitem__(self, index):
        return self.coarse_images[index], self.granular_images[index], self.y[index]
    

# def grid_search(lr_ranges:list[int],batch_size_ranges:list[int],regularization_ranges:list[int],train_dataset:Dataset,test_dataset:Dataset,annealing_rates:list[float],annealing_ranges:list[int]):
#     """
#     Apply grid search over the provided metrics, save the model and print grid search results.
#     """
#     epochs = 200
#     model = MultimodalFullModel()
#     best_r2 = -5
#     best_annealing_rate = 0
#     best_annealing_range = 0
#     saved_model = None
#     final_save_name = ''
#     best_batch_size = 0
#     best_lr = 0
#     best_l2 = 0
        
#     for lr in lr_ranges:
#         for batch_size in batch_size_ranges:
#             for l2_decay in regularization_ranges:
#                 for annealing_rate in annealing_rates:
#                     for annealing_range in annealing_ranges:
#                         r2, best_model, save_name = train(model=model,epochs=epochs,lr=lr,batch_size=batch_size,decay=l2_decay,train_data=train_dataset,test_data=test_dataset,annealing_range=annealing_range,annealing_rate=lr/annealing_rate)
#                         if r2 > best_r2:
#                             best_r2 = r2
#                             saved_model = best_model
#                             final_save_name = save_name
#                             best_lr = lr
#                             best_l2 = l2_decay
#                             best_batch_size = batch_size
#                             best_annealing_rate = annealing_rate
#                             best_annealing_range = annealing_range
                    
#     with open('search_results','w') as file:
#         file.write(f'Best R2 Score: {best_r2}\n')
#         file.write(f'Best LR: {best_lr}\n')
#         file.write(f'Best Batch Size: {best_batch_size}\n')
#         file.write(f'Best L2 decay: {best_l2}\n')
#         file.write(f'Best annealing rate: {best_lr/best_annealing_rate}')
#         file.write(f'Best ')
        
#     torch.save(saved_model,final_save_name)
    
class ModelTrainer():
    def __init__(self,print_graphs:bool,training_split:float,granular_model_size="256",coarse_model_size="16",save_model=False):
        os_name = os.name
        self.granular_model_size = granular_model_size
        
        if os_name == 'nt':
            coarse_image_path = "./data/coarse images"
            granular_image_path = './data/granular images'
            excel_path = './data/excel_files/duplicates_removed.csv'
        else:
            coarse_image_path = "/kaggle/input/coe-cnn-Experiment/coarse images"
            granular_image_path = "/kaggle/input/coe-cnn-Experiment/granular_images"
            excel_path = "/kaggle/input/coe-cnn-Experiment/Set2.csv"
        
        granular_model_training_locations = {
            '256' : "/kaggle/input/coe-cnn-Experiment/all_high_res",
            '512' : "/kaggle/input/coe-cnn-Experiment/512x512-granular-images",
            '128' : "/kaggle/input/coe-cnn-Experiment/128x128-granular-images",
            '64' : "/kaggle/input/coe-cnn-Experiment/64x64-granular-images",
            '32' : "/kaggle/input/coe-cnn-Experiment/32x32-granular-images"
        }

        coarse_model_training_locations = {
            '32' : "/kaggle/input/coe-cnn-Experiment/32x32-coarse-images",
            '16' : "/kaggle/input/coe-cnn-Experiment/16x16-coarse_IQR",
            '8' : "/kaggle/input/coe-cnn-Experiment/8x8-coarse-images",
            '4' : "/kaggle/input/coe-cnn-Experiment/4x4-coarse-images",
            '2' : "/kaggle/input/coe-cnn-Experiment/2x2-coarse-images"
        }
        
        # aerial_images_path = "/kaggle/input/coe-cnn-Experiment/high_res_granular"
        
        if granular_model_size not in granular_model_training_locations:
            raise Exception(f"Granular Model must match {list(granular_model_training_locations.keys())}")

        if coarse_model_size not in coarse_model_training_locations:
            raise Exception(f"Coarse Model must match {list(coarse_model_training_locations.keys())}")
        
        granular_image_path = granular_model_training_locations[granular_model_size]
        coarse_image_path = coarse_model_training_locations[coarse_model_size]
        self.print_graphs = print_graphs
        self.save_model = save_model
        
        # Load data
        orderings, parametric_features_df = self.get_parametric_features(excel_path)
        self.excel_path = excel_path
        granular_images_ndarray = self.convert_images_to_numpy(image_path=granular_image_path, ordering=orderings)
        coarse_images_ndarray = self.convert_images_to_numpy(image_path=coarse_image_path, ordering=orderings)
        # aerial_images_ndarray = self.convert_images_to_numpy(image_path=aerial_images_path,ordering=orderings)
        
        aawdt_ndarray = self.get_regression_values(file_path=excel_path,ordering=orderings)
        
        # Attach variables to object
        self.granular_model_size = granular_model_size
        self.parametric_features_df = parametric_features_df
        self.granular_images_ndarray = granular_images_ndarray
        # self.aerial_images_ndarray = aerial_images_ndarray
        self.coarse_images_ndarray = coarse_images_ndarray
        self.aawdt_ndarray = aawdt_ndarray
        
        # Split data
        training_split_index = int(granular_images_ndarray.shape[0] * training_split)
        
        param_train_df = parametric_features_df[:training_split_index]
        param_test_df = parametric_features_df[training_split_index:]
        
        param_train = self.preprocess_data(param_train_df)
        param_test = self.preprocess_data(param_test_df)
        granular_train, coarse_train,  = granular_images_ndarray[:training_split_index],coarse_images_ndarray[:training_split_index],
         
        granular_test, coarse_test, = granular_images_ndarray[training_split_index:],coarse_images_ndarray[training_split_index:],
        
        # aerial_train, aerial_test = aerial_images_ndarray[:training_split_index],aerial_images_ndarray[training_split_index:]
        
        aawdt_train, aawdt_test = aawdt_ndarray[:training_split_index],aawdt_ndarray[training_split_index:]

        
        self.train_dataset = TensorDataset(
            torch.from_numpy(coarse_train).permute(0,3,1,2) / 255,
            torch.from_numpy(granular_train).permute(0,3,1,2) / 255,
            # torch.from_numpy(aerial_train).permute(0,3,1,2) / 255,
            torch.from_numpy(param_train),
            torch.from_numpy(aawdt_train)
            )
        self.test_dataset = TensorDataset(
            torch.from_numpy(coarse_test).permute(0,3,1,2) / 255,
            torch.from_numpy(granular_test).permute(0,3,1,2) / 255,
            # torch.from_numpy(aerial_test).permute(0,3,1,2) / 255,
            torch.from_numpy(param_test),
            torch.from_numpy(aawdt_test)
            )

    def preprocess_data(self,df:pd.DataFrame)->np.ndarray:
        """
        Using the given data frame containing the information for studies, apply StandardScaling for 
        the 'Lat, Long, Speed' columns. Apply OneHotEncoding for the 'roadclass' col. 
        
        Then return the output as a nd.ndarray 
        """
        transform = ColumnTransformer(transformers=[
            ('Standard Scale',StandardScaler(),['Lat','Long','Speed','Lanes','Population2021','PopPerSqKm2021']),
        ],remainder='passthrough')
        
        return transform.fit_transform(df).astype(np.float32)

    def get_parametric_features(self,file_path:str)->tuple[list,pd.DataFrame]:
        """
        Create a ``pandas.DataFrame`` containing features.
        
        Parameters
        ----------
        excel_path : ``str``
            The path of the excel file to be opened. Contains the regression values per Study ID.
            Clean the data such that an equal distribution of road classes are present in the training and test dataset, based on the split,
            and then cojoin the data, such that the first ``train_split`` number of elements belong to the training set, while the next number of elements
            belong to the test set.  
        Returns
        -------
        ``pandas.DataFrame``
            Features in the form of a df.
        """
        df = pd.read_csv(file_path)
        df[['Collector','Local','Major Arterial','Minor Arterial']] = pd.get_dummies(df['Road Class'],dtype=int)
        shuffle_index = pd.Series(np.random.permutation(df.shape[0]))
        df = df.iloc[shuffle_index]
        ordering = df['Estimation_point'].tolist()
        df = df[['Lat','Long','Collector','Local','Major Arterial','Minor Arterial','Speed','Lanes','Population2021','PopPerSqKm2021']]
        
        return ordering, df

    def convert_images_to_numpy(self,image_path:str,ordering:list)->np.ndarray:
        """
        Converts the images stored under the given parameter path into one large ``numpy.ndarray`` object.
        Uses the ordering provided to order the images in the outputted array.
        
        Parameters
        ----------
        
        image_path : ``str``
            Path of the folder containing images
        excel_path : ``str``
            Path of the address for the excel file
        
        Returns
        -------
        
        ``numpy.ndarray``
            The numpy array containing the images
        """
        file_names = ordering
        absolute_path = os.path.abspath(image_path)
        nested_path_generator = os.walk(absolute_path)
        image_paths = None
        images_numpy = []
        
        
        for dirpath,dirnames,filenames in nested_path_generator:
            image_paths = {name.split('.')[0] : os.path.join(dirpath,name) for name in filenames}
            
            
            
        for file_name in file_names:
            image_path = image_paths[str(int(file_name))]
            image_array = np.array(Image.open(image_path))
            images_numpy.append(image_array)
        
        return_numpy_array = np.array(images_numpy)
        return return_numpy_array

    def get_regression_values(self,file_path:str,ordering:list)->np.ndarray:
        """
        Create a ``numpy.ndarray`` containing parallel matched regression values for the images.
        
        Parameters
        ----------
        excel_path : ``str``
            The path of the excel file to be opened. Contains the regression values per Study ID.    
        Returns
        -------
        ``numpy.ndarray``
            Tuple of numpy array corresponding to target values matched with given images. 
        """
        regression_values = []
        df = pd.read_csv(file_path)
        for id in ordering:
            regression_values.append(df['AAWDT'][df['Estimation_point'] == id].values[0])
        targets = np.array(regression_values)
        targets = targets.reshape((-1,1))
        return targets.astype(np.float32)

    def graph_ground_truth(self,x_values:tuple,y_values:list[list],title:str,xlabel:str,ylabel:str):
        plot_config = {
            0 : {
                'color':'darkorange',
                'marker' : 'd',
                'label':'Actual AAWDT'
            },
            1 : {
                'color':'seagreen',
                'marker' : 'd',
                'label':'Estimated AAWDT'
            }
        }
        for i,labels in enumerate(y_values):
            plt.plot(x_values,labels,color=plot_config[i]['color'],marker=plot_config[i]['marker'],label=plot_config[i]['label'])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(visible=True,)
        plt.title(title)
        plt.savefig(f'{title}-comparison.png')
        
        plt.cla()
        
        # Scale down AAWDT by a 1000
        for y_value in y_values:
            y_value = y_value / 1000
        
        plt.plot(y_values[0],y_values[0],color='red',linestyle='dashed')
        plt.scatter(y_values[0],y_values[1])
        plt.xlabel('Ground Truth AAWDT')
        plt.ylabel('Prediction AAWDT')
        plt.legend()
        plt.grid(visible=True,)
        plt.title(title)
        plt.savefig(f'{title}-slope.png')
        
        

    def create_graph(self,x_values:tuple,y_values:list[list],title:str,xlabel:str,ylabel:str,ground_truth=False):
        """
        Given the graph details, plot the graph and save it under ``<title>.png``. ENSURE THAT THE TRAINING Y VALUES ARE PLACED FIRST if ``ground_truth`` = ``False`` (default).
        """
        plt.figure(figsize=(10,6))
        
        if ground_truth:
            self.graph_ground_truth(x_values=x_values,y_values=y_values,title=title,xlabel=xlabel,ylabel=ylabel)
        else:
            plot_config = {
                0 : {
                    'color':'darkorange',
                    'marker' : 'd',
                    'label':'Training'
                },
                1 : {
                    'color':'seagreen',
                    'marker' : 'd',
                    'label':'Validation'
                }
            }
            for i,labels in enumerate(y_values):
                plt.plot(x_values,labels,color=plot_config[i]['color'],marker=plot_config[i]['marker'],label=plot_config[i]['label'])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(visible=True,)
            plt.title(title)
            plt.savefig(f'{title}.png')
    
    def kfold(self,epochs: int, lr: float, batch_size: int, decay: float,annealing_rate=0.0001,annealing_range=30,num_fold=10,save_name="kfold_results"):
        """
        Conduct k-fold CV, and saves the output results as an excel file under the name ``num_fold`` ``save_name``.xlsx.
        Returns the average MAPE across all folds.
        
        
        Parameters
        ----------
            epochs: ``int``
                The number of epochs
            num_fold : ``int``
                The number of folds to use during Cross Validation 
        """
        num_samples = self.parametric_features_df.shape[0]
        fold_breaks = math.ceil(num_samples / num_fold)
        fold_indices = [min(i * fold_breaks,num_samples) for i in range(num_fold + 1)]
        fold_subsets = []
        avg_score = 0
        dataframes = []
        
        i = 1
        while i <= len(fold_indices):
            if i != len(fold_indices):
                val_indices = [i for i in range(fold_indices[i-1],fold_indices[i])]
                training_indices = [i for i in range(0,fold_indices[i-1])]
                training_indices.extend([i for i in range(fold_indices[i],num_samples)])
                fold_subsets.append((training_indices,val_indices))
            i += 1
        
        fold = 1
        for training_indices, val_indices in fold_subsets:
            param_train_df = self.parametric_features_df.iloc[training_indices]
            param_test_df = self.parametric_features_df.iloc[val_indices]
            
            param_train = self.preprocess_data(param_train_df)
            param_test = self.preprocess_data(param_test_df)
            granular_train, coarse_train,  = self.granular_images_ndarray[training_indices],self.coarse_images_ndarray[training_indices],
            
            granular_test, coarse_test = self.granular_images_ndarray[val_indices],self.coarse_images_ndarray[val_indices],
            
            aawdt_train, aawdt_test = self.aawdt_ndarray[training_indices],self.aawdt_ndarray[val_indices]
            
            # aerial_train, aerial_test = self.aerial_images_ndarray[training_indices],self.aerial_images_ndarray[val_indices]

            
            self.train_dataset = TensorDataset(
                torch.from_numpy(coarse_train).permute(0,3,1,2) / 255,
                torch.from_numpy(granular_train).permute(0,3,1,2) / 255,
                # torch.from_numpy(aerial_train).permute(0,3,1,2) / 255,
                torch.from_numpy(param_train),
                torch.from_numpy(aawdt_train)
                )
            self.test_dataset = TensorDataset(
                torch.from_numpy(coarse_test).permute(0,3,1,2) / 255,
                torch.from_numpy(granular_test).permute(0,3,1,2) / 255,
                # torch.from_numpy(aerial_test).permute(0,3,1,2) / 255,
                torch.from_numpy(param_test),
                torch.from_numpy(aawdt_test)
                )
            
            metric = self.train_model(epochs=epochs,lr=lr,batch_size=batch_size,decay=decay,annealing_rate=annealing_rate,annealing_range=annealing_range)
            print(f'Metric for fold {fold}: {metric}')
            results = self.get_training_featues_with_predictions()
            dataframes.append(results)
            avg_score += metric
            fold += 1
        
        final_result = pd.concat(objs=dataframes,axis=0,ignore_index=True)
        final_result.to_excel(f'{num_fold}{save_name}.xlsx',index=False)
        return avg_score / num_fold
            
            
        
    
    def train_model(self,epochs: int, lr: float, batch_size: int, decay: float,annealing_rate=0.0001,annealing_range=30):
        self.model = MultimodalFullModel(int(self.granular_model_size))
        best_r2,save_name,model_copy,best_preds,best_targets = self.train(model=self.model,epochs=int(epochs),lr=lr,batch_size=int(batch_size),decay=decay,annealing_range=int(annealing_range),annealing_rate=annealing_rate,train_data=self.train_dataset,test_data=self.test_dataset,create_graphs=self.print_graphs)
        self.best_preds = best_preds
        self.best_targets = best_targets
        if self.save_model:
            torch.save(model_copy,save_name)
        
        return best_r2
    
    def get_training_featues_with_predictions(self)->pd.DataFrame:
        df = pd.read_csv(self.excel_path)
        column_names = ['AAWDT','Pred. AAWDT']
        prediction_df = pd.DataFrame(data=np.concatenate((self.best_targets,self.best_preds),axis=1),columns=column_names)
        df['AAWDT'] =  df['AAWDT'].astype(int)
        prediction_df['AAWDT'] = prediction_df['AAWDT'].astype(int)
        result_df = df.merge(prediction_df,on='AAWDT')
        return result_df
        
    
    def train(self,model:nn.Module,epochs: int, lr: float, batch_size: int, decay: float, train_data:Dataset, test_data:Dataset,create_graphs=False,annealing_rate=0.0001,annealing_range=30)->tuple:
        """
        Parameters
        ----------
        model : ``torch.nn.Module``
            The model architecture over which training should occur.
        epochs : ``int``    
            Number of iterations to train the model
        lr : ``float``
            Learning to set for training
        batch_size : ``int``
            Batch size to use during training
        train_data : ``torch.utils.data.Dataset``
            Training data
        test_data : ``torch.utils.data.Dataset``
            Test data
        create_graphs : ``bool``
            Specifies whether not training graphs should be generated (False by Default)
            
        Returns
        -------
        ``tuple[best_achieved_r2,copy_of_model,path_stored_model.pth]``
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        loss_fn = torch.nn.functional.huber_loss
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        scheduler = CosineAnnealingLR(optimizer=optim,T_max=annealing_range,eta_min=annealing_rate)
        training_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True,drop_last=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=True)
        train_r2_values = []
        valid_r2_values = []
        train_rmse_values = []
        valid_rmse_values = []
        train_mape_values = []
        valid_mape_values = []
        epochs_values = []
        early_stop = False
        
        early_stopping_threshold = 50
        early_stopping_index = 1
        best_mape = 200
        i = 0
        
        checkpoint = None
        best_preds = None
        best_targets = None
        with open('training.txt', 'w') as file:
            while i < epochs and not early_stop:
                model.train()
                all_targets, all_preds = [], []

                for coarse_input, granular_input, param_input, target in training_loader:
                    optim.zero_grad()
                    pred = model(coarse_input.to(device), granular_input.to(device),param_input.to(device))
                    loss = loss_fn(pred, target.to(device))
                    loss.backward()
                    optim.step()

                    all_targets.append(target.detach().cpu().numpy())
                    all_preds.append(pred.detach().cpu().numpy())

                # Compute metrics after epoch
                all_targets = np.concatenate(all_targets)
                all_preds = np.concatenate(all_preds)
                training_loss = np.sqrt(mean_squared_error(all_targets, all_preds))
                r2 = r2_score(all_targets, all_preds)
                mape = mean_absolute_percentage_error(all_targets, all_preds)

                epochs_values.append(i)
                train_rmse_values.append(training_loss)
                train_mape_values.append(mape*100)
                train_r2_values.append(r2)
                file.write(f'\n---------------------------\nTraining Epoch: {i}\n')
                file.write(f'r2 score is {r2:.3f}\n')
                file.write(f'MAPE is {mape * 100:.2f}%\n')
                file.write(f'RMSE is {training_loss:.3f}\n')
                file.write('---------------------------\n')

                model.eval()
                valid_targets, valid_preds = [], []
                with torch.no_grad():
                    for coarse_input, granular_input, param_input, target in test_loader:
                        pred = model(coarse_input.to(device), granular_input.to(device),param_input.to(device))
                        valid_targets.append(target.detach().cpu().numpy())
                        valid_preds.append(pred.detach().cpu().numpy())

                valid_targets = np.concatenate(valid_targets)
                valid_preds = np.concatenate(valid_preds)
                valid_loss = np.sqrt(mean_squared_error(valid_targets, valid_preds))
                valid_r2 = r2_score(valid_targets, valid_preds)
                valid_mape = mean_absolute_percentage_error(valid_targets, valid_preds)
                
                #scheduler.step(mean_squared_error(valid_targets, valid_preds))
                valid_mape_values.append(valid_mape * 100)
                valid_rmse_values.append(valid_loss)
                valid_r2_values.append(valid_r2)
                
                scheduler.step()
                

                file.write(f'\n---------------------------\nValidation Epoch: {i}\n')
                file.write(f'r2 score is {valid_r2:.3f}\n')
                file.write(f'MAPE is {valid_mape * 100:.2f}%\n')
                file.write(f'RMSE is {valid_loss:.3f}\n')
                file.write('---------------------------\n')
                
                # Early Stopping Mechanism
                if valid_mape < best_mape:
                    best_mape = valid_mape
                    checkpoint = {"Saved Model":model.state_dict()}
                    best_preds = valid_preds
                    best_targets = valid_targets
                    early_stopping_index = 1
                else:
                    early_stopping_index += 1
                
                if early_stopping_index == early_stopping_threshold:
                    early_stop = True
                i += 1
        
        if create_graphs:
            self.create_graph(epochs_values,[train_mape_values,valid_mape_values],"Multimodal MAPE","Epochs","MAPE (%)")
            self.create_graph(epochs_values,[train_rmse_values,valid_rmse_values],"Multimodal RMSE","Epochs","RMSE")
            self.create_graph(epochs_values,[train_r2_values,valid_r2_values],"Multimodal R2Score","Epochs","R2Score")
            self.create_graph([i + 1 for i in range(best_targets.shape[0])],y_values=[best_targets.reshape(best_targets.shape[0]),best_preds.reshape(best_preds.shape[0])],title="Ground Truth",xlabel="Data Point",ylabel="AAWDT",ground_truth=True)
        save_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_copy = MultimodalFullModel(int(self.granular_model_size)).to(device=device).load_state_dict(checkpoint['Saved Model'])
        
        return (best_mape,save_name,model_copy,best_preds,best_targets)
    

if __name__ == "__main__":

    
    lr_ranges = [0.005]
    batch_size_ranges = [8,16,24]
    regularization_ranges = [0.05]
    annealing_rates = [2,5,10]
    annealing_ranges = [10,15,30,60]
    
    pbounds = {
        'lr' : (0.0001,0.001),
        'epochs' : (50,200),
        'batch_size' : (5,32),
        'decay' : (0, 0.00005),
        'annealing_range' : (10,200),
        'annealing_rate' : (0.0001,0.05)
    }
    
    annealing_range = 45
    annealing_rate = 0.01734348027944808
    batch_size = 15
    decay = 0.00002694
    epochs = 112
    lr = 0.0007166975503570836
    
    coarse_param = ['32', '16', '8', '4', '2']
    granular_param = ['256', '512', '128', '64', '32']

    save_data = {
        'Granular Param' : [],
        'Coarse Param' : [],
        'Score' : []
    }
    
    
    # for coarse in coarse_param:
    #     for granular in granular_param:
    #         save_data['Coarse Param'].append(coarse)
    #         save_data['Granular Param'].append(granular)
    trainer = ModelTrainer(print_graphs=False,save_model=False,training_split=0.85,granular_model_size="256",coarse_model_size="16")
    score = trainer.kfold(epochs=epochs,lr=lr,batch_size=batch_size,decay=decay,annealing_range=annealing_range,annealing_rate=annealing_rate,num_fold=10)
    print(score)
    # best_r2 = trainer.train_model(epochs=epochs,lr=lr,batch_size=batch_size,decay=decay,annealing_range=annealing_range,annealing_rate=annealing_rate)
    # results = trainer.get_training_featues_with_predictions()
    # results.to_excel('Prediction Results.xlsx',index=False)
    # print(best_r2)
    #         save_data['Score'].append(best_r2)
            
    # save_df = pd.DataFrame(data=save_data)
    # save_df.to_excel('Sensitivity_results.xlsx')
    # optimizer = BayesianOptimization(
    #     f=trainer.train_model,
    #     pbounds=pbounds,
    #     random_state=1
    # )
    
    # optimizer.maximize(
    #     init_points = 4,
    #     n_iter = 20
    # )
    # print(optimizer.max)
    #grid_search(lr_ranges=lr_ranges,batch_size_ranges=batch_size_ranges,regularization_ranges=regularization_ranges,train_dataset=train_dataset,test_dataset=test_dataset,annealing_rates=annealing_rates,annealing_ranges=annealing_ranges)
    #best_rmse, best_model, save_name = train(epochs=epochs,lr=lr,batch_size=batch_size,decay=l2_decay,train_data=train_dataset,test_data=test_dataset)
    # torch.save(best_model,save_name)
    
    