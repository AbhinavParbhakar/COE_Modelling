import os
import subprocess
import sys

# Check if running on Kaggle and install dependencies if not already installed
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    print("Started download for pytorch_geometric")
    import torch
    version = torch.__version__.split("+")[0]  # Extracts version without CUDA/CPU suffix
    install_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch_geometric", "pyg_lib", "torch_scatter",
        "torch_sparse", "torch_cluster", "torch_spline_conv",
        "-f", f"https://data.pyg.org/whl/torch-{version}+cu121.html"
    ]
    subprocess.check_call(install_cmd)


import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.mixture import GaussianMixture

def preprocess_data(df:pd.DataFrame)->np.ndarray:
    """
    Using the given data frame containing the information for studies, apply StandardScaling for 
    the 'Lat, Long, Speed' columns. Apply OneHotEncoding for the 'roadclass, land usage' columns. 
    
    Then return the output as a nd.ndarray 
    """
    transform = ColumnTransformer(transformers=[
        ('Standard Scale',StandardScaler(),['Lat','Long','Speed (km/h)']),
        ('One Hot Encode',OneHotEncoder(),['roadclass','Land Usage'])
    ])
    
    return transform.fit_transform(df)

def create_features(file_source:str)->tuple[Data,torch.Tensor]:
    """
    Given a data source linking to the the excel file storing the data, create input for the GNN,
    and return a tuple containing firstly the graph data, and secondly the corresponding labels
    """
    df = pd.read_excel(file_source)
    
    # Split X and Y
    features = df.drop(level=['Volume','UniqueID','Date'],axis=1)
    targets = df['Volume']
    
    # Scale and get the data as a np.array
    features_array = preprocess_data(features)
    
    # Generate the weights for the graphs by creating clusters
    em = GaussianMixture(n_components=16,random_state=0)
    em.fit(features_array)
    clusters = em.predict(features_array)
    clusters = clusters.reshape((-1,1))
    encoded_values = OneHotEncoder().fit_transform(clusters)
    
    # Create the edge_index
    source = []
    destination = []
    
    for i in range(features_array.shape[0]):
        for j in range(features_array.shape[0]):
            cluster_i = clusters[i][0]
            cluster_j = clusters[j][0]
            if i != j and cluster_i == cluster_j:
                
                # Add one way
                source.append[i]
                destination.append[j]
                
                # Add the other way
                source.append[j]
                destination.append[i]
    
    source_array = np.array(source).reshape((1,-1))
    destination_array = np.array(destination).reshape((1,-1))
    edges = np.concat((source_array,destination_array),axis=0)
    
    
    
    
            
    

if __name__ == "__main__":
    data_source = "/kaggle/input/coe-datatest-2/features2-4-2025.xlsx"
    features = create_features(data_source)
    
