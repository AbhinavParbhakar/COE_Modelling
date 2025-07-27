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
from copy import deepcopy
from sklearn.decomposition import PCA
import time

# Check if running on Kaggle and install dependencies if not already installed   
print("Started download for pytorch_geometric")
import torch
version = torch.__version__.split("+")[0]  # Extracts version without CUDA/CPU suffix
install_cmd = [
    sys.executable, "-m", "pip", "install", "geopandas",
    "torch_geometric", "bayesian-optimization",
    "-f", f"https://data.pyg.org/whl/torch-{version}+cu121.html"
]
subprocess.check_call(install_cmd)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from bayes_opt import BayesianOptimization
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils import k_hop_subgraph

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256,540),
            nn.BatchNorm1d(540),
            nn.ReLU(),
            
        )
    
    def forward(self,x):
        x = self.net(x)
        
        return x
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
        
        return self.flatten(x1)

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
        
        return self.flatten(x2)
   
class MultimodalFullModel(nn.Module):
    def __init__(self,granular_image_dimension=256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout()
        
        
        granular_size_dict = {
            400 : 540,
            200 : 540,
            100 : 540,
            50 : 540,
            25 : 540,
        }
        
        granular_model_dict = {
            400 : TwoFiftySixGranularImageModel(),
            200 : TwoFiftySixGranularImageModel(),
            100 : TwoFiftySixGranularImageModel(),
            50 : TwoFiftySixGranularImageModel(),
            25 : TwoFiftySixGranularImageModel(),
        }

        self.key_dimension = 120
        self.value_dimension = 60
        
        
        self.param_key_dimension = 100
        self.param_value_dimension = 2
        
        
        coarse_image_size = 240
        parametric_data_size = 540
        # combination_size = granular_size_dict[granular_image_dimension] + coarse_image_size + parametric_data_size
        combination_size = granular_size_dict[granular_image_dimension] + parametric_data_size
        self.parametric_module = NN()
        self.granular_module = granular_model_dict[granular_image_dimension]
        # self.aerial_module = granular_model_dict[256]
        self.fc1 = nn.Linear(in_features=combination_size,out_features=500)
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
        
        
    def forward(self,granular_input:torch.FloatTensor,parametric_input:torch.FloatTensor):
        granular_image_embedding = self.granular_module(granular_input)
        # aerial_image_embedding = self.aerial_module(aerial_input)
        
        parametric_embeddings = self.parametric_module(parametric_input)
        parametric_embeddings = self.relu(parametric_embeddings)
        
        # image_embeddings = self.cross_attention_coarse_granular(granular_image_embedding,coarse_image_embedding,flatten=False)
        # combination = self.cross_attention_images_parameter(images_embedding=image_embeddings,parametric_embedding=parametric_embeddings,flatten=True)
        
        # query_key_output = self.softmax(torch.mm(input=query,mat2=key.T) / math.sqrt(self.key_dimension))
        # output = torch.mm(query_key_output,value)
        
        # coarse_image_embedding = self.self_attention_coarse_image_embedding(coarse_image_embedding)
        # granular_image_embedding = self.self_attention_granular_image_embedding(granular_image_embedding)
        # parametric_embeddings = self.self_attention_parameters(parametric_embeddings)
                
        combination = torch.cat(tensors=(granular_image_embedding,parametric_embeddings),dim=1)
        
        # combination = self.dropout(combination)
        
        combination = self.fc1(combination)
        output = self.relu(combination)
        output = self.dropout(output)
        output = self.fc2(output)
        # output = self.relu(output)
        # output = self.fc3(output)
        # output = self.relu(output)
        # output = self.fc4(output)
        
        return output
    
class MM_GNN(nn.Module):
    def __init__(self, adj_matrix:Data,granular_image_size=256):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj_matrix = adj_matrix.to(device)
        self.edge_index = adj_matrix.edge_index
        self.edge_weight = adj_matrix.edge_weight
        # self.multi_modal_module = MultimodalFullModel(granular_image_dimension=granular_image_size)
        
        self.bn1 = nn.BatchNorm1d(num_features=500)
        self.bn2 = nn.BatchNorm1d(num_features=250)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.gcn1 = GATConv(in_channels=-1,out_channels=100,heads=5,concat=True,edge_dim=1)
        self.gcn2 = GATConv(in_channels=-1,out_channels=50, heads=5,edge_dim=1)
        self.gcn3 = GATConv(in_channels=-1,out_channels=32,heads = 4,edge_dim=1)
        self.gcn4 = GATConv(in_channels=-1,out_channels=8, heads = 4,edge_dim=1)
        self.gcn5 = GATConv(in_channels=-1,out_channels=1,edge_dim=1)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()      
        
        self.relu = nn.ReLU()
    def generate_batches(self,data:list,batch_size:int,shuffle=False)->list[list]:
        """
        Given the data as a one-dimensional ist and the batch size, generate batches of the data and return it
        """
        prev_index = 0
        next_index = 0
        batches = []
        
        if shuffle:
            data = np.random.permutation(data).tolist()
        
        while prev_index < len(data):
            next_index = min(len(data),prev_index + batch_size)
            batch = data[prev_index:next_index]
            batches.append(batch)
            prev_index = next_index
        
        return batches

    def mm_checkpoint(self,coarse_images,granular_images,params):
        return self.multi_modal_module(coarse_images,granular_images,params)
    
    def forward(self,x):
        x = self.gcn1(x=x,edge_index=self.edge_index,edge_attr=self.edge_weight,)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gcn2(x=x,edge_index=self.edge_index,edge_attr=self.edge_weight)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.gcn3(x=x,edge_index=self.edge_index,edge_attr=self.edge_weight)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.gcn4(x=x,edge_index=self.edge_index,edge_attr=self.edge_weight)
        x = self.relu(x)
        x = self.gcn5(x,edge_index=self.edge_index,edge_attr=self.edge_weight)
        
        return x



class FullDataset():
    def __init__(self,data:torch.Tensor,target:torch.Tensor,training_indices:list,valid_indices:list):
        self.data = data
        self.target = target
        self.training_indices = training_indices
        self.valid_indices = valid_indices
    


class ModelTrainer():
    def __init__(self,print_graphs:bool,training_split:float,adj_matrix_path:str,granular_model_size="256",coarse_model_size="16",save_model=False):
        os_name = os.name
        self.granular_model_size = granular_model_size
        
        if os_name == 'nt':
            coarse_image_path = "./data/coarse images"
            granular_image_path = './data/granular images'
            excel_path = './data/excel_files/duplicates_removed.csv'
        else:
            coarse_image_path = "/kaggle/input/coe-cnn-experiment/coarse images"
            granular_image_path = "/kaggle/input/coe-cnn-experiment/granular_images"
            excel_path = "/kaggle/input/coe-cnn-experiment/Final_Dataset.csv"
        
        granular_model_training_locations = {
            '25' : "/kaggle/input/coe-cnn-experiment/Final_aerial_final",
            '512' : "/kaggle/input/coe-cnn-Experiment/512x512-granular-images",
            '128' : "/kaggle/input/coe-cnn-Experiment/128x128-granular-images",
            '64' : "/kaggle/input/coe-cnn-Experiment/64x64-granular-images",
            '32' : "/kaggle/input/coe-cnn-Experiment/32x32-granular-images"
        }

        coarse_model_training_locations = {
            '32' : "/kaggle/input/coe-cnn-Experiment/32x32-coarse-images",
            '16' : "/kaggle/input/coe-cnn-Experiment/16x16-coarse_IQR",
            '8' : "/kaggle/input/coe-cnn-Experiment/coarse_all_images_8x8",
            '4' : "/kaggle/input/coe-cnn-Experiment/4x4-coarse-images",
            '2' : "/kaggle/input/coe-cnn-Experiment/2x2-coarse-images"
        }
        
        # aerial_images_path = "/kaggle/input/coe-cnn-Experiment/high_res_granular"
        
        if granular_model_size not in granular_model_training_locations:
            raise Exception(f"Granular Model must match {list(granular_model_training_locations.keys())}")

        if coarse_model_size not in coarse_model_training_locations:
            raise Exception(f"Coarse Model must match {list(coarse_model_training_locations.keys())}")
        
        internal_embedding_path = "/kaggle/input/coe-cnn-experiment/Internal_embeddings.csv"
        granular_image_path = granular_model_training_locations[granular_model_size]
        coarse_image_path = coarse_model_training_locations[coarse_model_size]
        self.print_graphs = print_graphs
        self.save_model = save_model
        
        # Load data
        # self.adj_matrix = self.get_adjacency_matrix(adj_matrix_path)
        self.internal_embeddings_ndarray = pd.read_csv(internal_embedding_path).to_numpy(dtype=np.float32)
        orderings, parametric_features_df = self.get_parametric_features(excel_path)
        self.parametric_features_df = parametric_features_df
        self.excel_path = excel_path
        self.granular_images_ndarray = self.convert_images_to_numpy(image_path=granular_image_path, ordering=orderings)
        # coarse_images_ndarray = self.convert_images_to_numpy(image_path=coarse_image_path, ordering=orderings)
        # aerial_images_ndarray = self.convert_images_to_numpy(image_path=aerial_images_path,ordering=orderings)
        
        aawdt_ndarray = self.get_regression_values(file_path=excel_path,ordering=None)
        
        # Attach variables to object
        self.saved_model_path = '/kaggle/input/coe-cnn-experiment/multimodalCNN_updated.pt'
        self.granular_model_size = granular_model_size
        # self.parametric_features_df = parametric_features_df
        # self.granular_images_ndarray = granular_images_ndarray
        # # self.aerial_images_ndarray = aerial_images_ndarray
        # self.coarse_images_ndarray = coarse_images_ndarray
        self.aawdt_ndarray = aawdt_ndarray
        
        
    def generate_embeddings(self,apply_pca=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultimodalFullModel(int(self.granular_model_size)).to(device=device)
        model.load_state_dict(torch.load(self.saved_model_path))
        
        
        params = self.preprocess_data(self.parametric_features_df)
        
        dataset = TensorDataset(
            # torch.from_numpy(self.coarse_images_ndarray).permute(0,3,1,2) / 255,
            torch.from_numpy(self.granular_images_ndarray).permute(0,3,1,2) / 255,
            torch.from_numpy(params)
        )
        
        features = []
        dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=100)
        
        model.eval()
        for granular, param in dataloader:
            # coarse = coarse.to(device)
            granular = granular.to(device)
            param = param.to(device)
            output = model(granular,param)
            
            features.append(output.detach().cpu().numpy())
        
        features_output = np.concatenate(features,axis=0)
        
        if apply_pca:
            pca = PCA(n_components=120)
            features_output = pca.fit_transform(features_output).astype(np.float32)
        
        columns = [i for i in range(features_output.shape[1])]
        
        df = pd.DataFrame(data=features_output,columns=columns)
        
        df.to_csv('Internal_embeddings_200.csv',index=False)

    def get_adjacency_matrix(self,path:str)->Data:
        """
        Given the path listed, return a Data object with the provided edge weights and indices and mock data
        """
        df = pd.read_csv(path)
        num_rows, num_cols = df.shape
        origins = []
        destinations = []
        weights = []
        for i in range(num_rows):
            for j in range(num_cols):
                distance = df.iloc[i,j]
                if i != j and not np.isinf(distance):
                    origins.append(i)
                    destinations.append(j)
                    if distance == 0.0:
                        weights.append(1)
                    else:
                        weights.append( 1 / distance)
                    
        
        edge_index = torch.tensor([origins,destinations],dtype=torch.long)
        edge_weights = torch.tensor(weights,dtype=torch.float32)
        mock_data = torch.rand(size=(num_rows,1))
        
        data = Data(x=mock_data,edge_index=edge_index,edge_weight=edge_weights)
        
        return data
        
    def preprocess_data(self,df:pd.DataFrame)->np.ndarray:
        """
        Using the given data frame containing the information for studies, apply StandardScaling for 
        the 'Lat, Long, Speed' columns. Apply OneHotEncoding for the 'roadclass' col. 
        
        Then return the output as a nd.ndarray 
        """
        transform = ColumnTransformer(transformers=[
            ('Standard Scale',StandardScaler(),['Lat','Long','Speed','Lanes']),
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
        ordering = df['Estimation_point'].tolist()
        df = df[['Lat','Long','Collector','Local','Major Arterial','Minor Arterial','Speed','Lanes',]]
        
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
        # for id in ordering:
        #     regression_values.append(df['AAWDT'][df['Estimation_point'] == id].values[0])
        targets = df['AAWDT'].to_numpy()
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
        non_zero_indices = np.where(self.aawdt_ndarray == 0.0)[0].tolist()
        # non_zero_indices = []
        num_samples = self.internal_embeddings_ndarray.shape[0] - len(non_zero_indices)
        start_index = non_zero_indices[len(non_zero_indices) - 1] + 1
        # start_index = 0
        fold_breaks = math.ceil(num_samples / num_fold)
        fold_indices = [min(i * fold_breaks,num_samples) + start_index for i in range(num_fold + 1)]
        fold_subsets = []
        avg_score = 0
        dataframes = []
        
        i = 1
        while i <= len(fold_indices):
            if i != len(fold_indices):
                val_indices = [i for i in range(fold_indices[i-1],fold_indices[i])]
                training_indices = [i for i in range(start_index,fold_indices[i-1])]
                training_indices.extend([i for i in range(fold_indices[i],num_samples + start_index)])
                fold_subsets.append((training_indices,val_indices))
            i += 1
        
        fold = 1
        for training_indices, val_indices in fold_subsets:
            dataset = FullDataset(
                data=torch.from_numpy(self.internal_embeddings_ndarray),
                target=torch.from_numpy(self.aawdt_ndarray),
                training_indices=training_indices,
                valid_indices=val_indices
                
            )
            # print(f"Starting training for Fold {fold}")
            time_start = time.time()
            metric = self.train_model(epochs=epochs,lr=lr,batch_size=batch_size,decay=decay,annealing_rate=annealing_rate,annealing_range=annealing_range,dataset=dataset)
            time_end = time.time()
            # print(f'Metric for fold {fold}: {metric}, training time: {time_end - time_start}')
            results = self.get_training_featues_with_predictions()
            dataframes.append(results)
            avg_score += metric
            fold += 1
        
        final_result = pd.concat(objs=dataframes,axis=0,ignore_index=True)
        final_result.to_excel(f"{num_fold}{save_name}{self.excel_path.split('/')[-1].split('.')[0]}.xlsx",index=False)
        print(avg_score / num_fold)
        return (avg_score / num_fold) * -1
            
            
        
    
    def train_model(self,epochs: int, lr: float, batch_size: int, decay: float, dataset:FullDataset,annealing_rate=0.0001,annealing_range=30,):
        self.model = MM_GNN(granular_image_size=int(self.granular_model_size),adj_matrix=self.adj_matrix)
        best_r2,save_name,model_copy,best_preds,best_targets,indices = self.train(model=self.model,epochs=int(epochs),lr=lr,batch_size=int(batch_size),decay=decay,annealing_range=int(annealing_range),annealing_rate=annealing_rate,dataset=dataset,create_graphs=self.print_graphs)
        self.best_preds = best_preds
        self.best_targets = best_targets
        self.indices = indices
        if self.save_model:
            torch.save(model_copy,save_name)
        
        return best_r2
    
    def get_training_featues_with_predictions(self)->pd.DataFrame:
        df = pd.read_csv(self.excel_path)
        column_names = ['Pred. AAWDT']
        prediction_df = pd.DataFrame(data=self.best_preds,columns=column_names,index=self.indices)
        
        df['AAWDT'] =  df['AAWDT'].astype(int)
        result_df = df.join(prediction_df,how='inner')
        return result_df
    
    
    def generate_batches(self,data:list,batch_size:int,shuffle=False)->list[list]:
        """
        Given the data as a one-dimensional ist and the batch size, generate batches of the data and return it
        """
        prev_index = 0
        next_index = 0
        batches = []
        
        if shuffle:
            data = np.random.permutation(data).tolist()
        
        while prev_index < len(data):
            next_index = min(len(data),prev_index + batch_size)
            batch = data[prev_index:next_index]
            batches.append(batch)
            prev_index = next_index
        
        return batches
    
    def train(self,model:nn.Module,epochs: int, lr: float, batch_size: int, decay: float,dataset:FullDataset,create_graphs=False,annealing_rate=0.0001,annealing_range=30)->tuple:
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
        training_indices = dataset.training_indices
        valid_indices = dataset.valid_indices
        target = dataset.target.to(device)
        data = dataset.data.to(device)
        with open('training.txt', 'w') as file:
            while i < epochs and not early_stop:
                model.train()
                all_targets, all_preds = [], []     
                optim.zero_grad()           
                pred = model(data)
                loss = loss_fn(pred[training_indices], target[training_indices])
                loss.backward()
                optim.step()
                
                all_targets.append(target[training_indices].detach().cpu().numpy())
                all_preds.append(pred[training_indices].detach().cpu().numpy())

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
                # with torch.no_grad():
                #     for coarse_input,granular_input,param_input,target,graph,center_ids in valid_loader:
                #         pred = model(coarse_input, granular_input,param_input,device,graph,center_ids)
                #         target = target.to(device)
                #         valid_targets.append(pred.detach().cpu().numpy())
                #         valid_preds.append(target.detach().cpu().numpy())

                valid_targets.append(target[valid_indices].detach().cpu().numpy())
                valid_preds.append(pred[valid_indices].detach().cpu().numpy())                

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
                    checkpoint = {"Saved Model":deepcopy(model.state_dict())}
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
        # model_copy = MultimodalFullModel(int(self.granular_model_size)).to(device=device).load_state_dict(checkpoint['Saved Model'])
        model_copy = MM_GNN(adj_matrix=self.adj_matrix,granular_image_size=256).to(device=device)
        model_copy.load_state_dict(checkpoint['Saved Model'])
        
        return (best_mape,save_name,model_copy,best_preds,best_targets,valid_indices)
    
    def generate_estimates(self,epochs: int, lr: float, batch_size: int, decay: float,annealing_rate=0.0001,annealing_range=30):
        non_zero_indices = np.where(self.aawdt_ndarray > 0.0)[0].tolist()
        zero_indices = np.where(self.aawdt_ndarray == 0.0)[0].tolist()
        dataset = FullDataset(
            data=torch.from_numpy(self.internal_embeddings_ndarray),
            target=torch.from_numpy(self.aawdt_ndarray),
            training_indices=non_zero_indices,
            valid_indices=None
        )
        best_mape,save_name,model_copy,best_preds,best_targets = self.train(
            model=MM_GNN(adj_matrix=self.adj_matrix,granular_image_size=256),
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            decay=decay,
            dataset=dataset,
            create_graphs=False,
            annealing_range=annealing_range,
            annealing_rate=annealing_rate
        )
        print(best_mape)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_points_tensor = torch.from_numpy(self.internal_embeddings_ndarray).to(device=device)
        model_copy.eval()
        
        estimates = model_copy(all_points_tensor)
        estimates = estimates.detach().cpu().numpy().reshape(estimates.shape[0])
        estimate_results_df = pd.DataFrame(data=pd.Series(estimates),columns=['Esimated AAWDT'],index=zero_indices)
        df = pd.read_csv(self.excel_path)
        results = df.join(estimate_results_df,how='inner')
        print(results.shape)
        results.to_excel('Estimate_results.xlsx')
        
        
    

if __name__ == "__main__":
    

    lr_ranges = [0.005]
    batch_size_ranges = [8,16,24]
    regularization_ranges = [0.05]
    annealing_rates = [2,5,10]
    annealing_ranges = [10,15,30,60]
    
    pbounds = {
        'lr' : (0.0001,0.001),
        'epochs' : (50,500),
        'batch_size' : (5,32),
        'decay' : (0, 0.00005),
        'annealing_range' : (10,500),
        'annealing_rate' : (0.0001,0.05)
    }
    
    annealing_range = 242.9688059056526
    annealing_rate = 0.05
    batch_size = 5.0
    decay = 0.0
    epochs = 500.0
    lr = 0.0001

    coarse_param = ['32', '16', '8', '4', '2']
    granular_param = ['256', '512', '128', '64', '32']

    save_data = {
        'Granular Param' : [],
        'Coarse Param' : [],
        'Score' : []
    }
    
    
    for coarse in coarse_param:
        for granular in granular_param:
            save_data['Coarse Param'].append(coarse)
            save_data['Granular Param'].append(granular)
    thresholds = [i for i in range(50,500 + 1,50)]
    adj_matrix_paths = [f'/kaggle/input/coe-cnn-Experiment/Adjacency_Matrix_{distance}_meters.csv' for distance in thresholds]
    scores = []
    # for adj_path in adj_matrix_paths:
    # adj_path = '/kaggle/input/coe-cnn-Experiment/Adjacency_Matrix_500_meters.csv'
    adj_path = '/kaggle/input/coe-cnn-experiment/Adjacency_Matrix_150_meters.csv' 
    trainer = ModelTrainer(print_graphs=False,save_model=False,training_split=0.85,granular_model_size="25",coarse_model_size="8",adj_matrix_path=adj_path)
    trainer.generate_embeddings(apply_pca=False)
    # trainer.kfold(epochs=epochs,lr=lr,batch_size=batch_size,decay=decay,annealing_range=annealing_range,annealing_rate=annealing_rate,num_fold=10,save_name="experiment")
    # score = trainer.kfold(epochs=epochs,lr=lr,batch_size=batch_size,decay=decay,annealing_range=annealing_range,annealing_rate=annealing_rate)
    # print(score)
        # scores.append(scores)
    
    # plt.xlabel(xlabel="Neighbor Threshold (meters)")
    # plt.ylabel(ylabel="Mean Average Percent Error")
    # plt.title(label="MAPE vs Neighborhood threshold")
    # plt.plot(thresholds,scores)
    # plt.savefig('Results.png')
    # best_r2 = trainer.train_model(epochs=epochs,lr=lr,batch_size=batch_size,decay=decay,annealing_range=annealing_range,annealing_rate=annealing_rate)
    # results = trainer.get_training_featues_with_predictions()
    # results.to_excel('Prediction Results.xlsx',index=False)
    # print(best_r2)
    #         save_data['Score'].append(best_r2)
            
    # save_df = pd.DataFrame(data=save_data)
    # save_df.to_excel('Sensitivity_results.xlsx')
    # optimizer = BayesianOptimization(
    #     f=trainer.kfold,
    #     pbounds=pbounds,
    #     random_state=1
    # )
    
    # optimizer.maximize(
    #     init_points = 10,
    #     n_iter = 30
    # )
    # print(optimizer.max)
    #grid_search(lr_ranges=lr_ranges,batch_size_ranges=batch_size_ranges,regularization_ranges=regularization_ranges,train_dataset=train_dataset,test_dataset=test_dataset,annealing_rates=annealing_rates,annealing_ranges=annealing_ranges)
    #best_rmse, best_model, save_name = train(epochs=epochs,lr=lr,batch_size=batch_size,decay=l2_decay,train_data=train_dataset,test_data=test_dataset)
    # torch.save(best_model,save_name)
    
    