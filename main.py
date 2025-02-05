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
        "torch_geometric", "pyg_lib", "torch_scatter", "torcheval",
        "torch_sparse", "torch_cluster", "torch_spline_conv",
        "-f", f"https://data.pyg.org/whl/torch-{version}+cu121.html"
    ]
    subprocess.check_call(install_cmd)


import pandas as pd
import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GCNConv
from torcheval.metrics import R2Score
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adagrad
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.mixture import GaussianMixture


class GNN(nn.Module):
    def __init__(self,in_channels:int,hidden_num:int,output:int):
        super().__init__()
        self.l1 = GCNConv(in_channels=in_channels,out_channels=hidden_num)
        self.l2 = GCNConv(in_channels=hidden_num,out_channels=output)
        self.relu = nn.ReLU()
    
    def forward(self,data:Data):
        nodes, edges = data.x,data.edge_index
        x = self.l1(nodes,edges)
        x = self.relu(x)
        x = self.l2(x,edges)
        
        return x

def train(data:Data,lr=0.001,epochs=100):
    """
    Train the GNN model on the trained data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
    data = data.to(device)
    model = GNN(in_channels=data.x.size()[1],hidden_num=data.x.size()[1] * 2,output=1)
    model = model.to(device)
    
    optim = Adagrad(params=model.parameters(),lr=lr)
    loss_fn = mse_loss
    
    model.train()
    with open('results.txt','w') as file:
        for i in range(epochs):
            model.train()
            optim.zero_grad()
            output = model(data)
            loss = loss_fn(output[data.train_mask],data.y[data.train_mask])
            loss.backward()
            optim.step()
            
            if i % 10 == 0:
                with torch.no_grad():
                    output = model(data)
                    metric = R2Score()
                    metric.update(output[data.val_mask],data.y[data.val_mask])
                    score = metric.compute().item()
                    file.write(f'R_2 Score after epoch {i-1} is {score}\n')
    
        file.close()
                
        
    

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

def create_features(file_source:str)->Data:
    """
    Given a data source linking to the the excel file storing the data, create input for the GNN,
    and returns the graph data after splitting the features using RandomNode Split
    """
    df = pd.read_excel(file_source)
    
    # Split X and Y
    features = df.drop(labels=['Volume','UniqueID','Date'],axis=1)
    targets = df['Volume']
    
    # Scale and get the data as a np.array
    features_array = preprocess_data(features)
    
    # Generate the weights for the graphs by creating clusters
    em = GaussianMixture(n_components=16,random_state=0)
    em.fit(features_array)
    clusters = em.predict(features_array)
    clusters = clusters.reshape((-1,1))
    
    # Create the edge_index
    source = []
    destination = []
    
    for i in range(features_array.shape[0]):
        for j in range(features_array.shape[0]):
            cluster_i = clusters[i][0]
            cluster_j = clusters[j][0]
            if i != j and cluster_i == cluster_j:
                
                # Add one way
                source.append(i)
                destination.append(j)
                
                # Add the other way
                source.append(j)
                destination.append(i)
    
    source_array = np.array(source).reshape((1,-1))
    destination_array = np.array(destination).reshape((1,-1))
    edges = np.concatenate((source_array,destination_array),axis=0)
    print(edges.shape)  
    print(features_array.shape) 
    
    x = torch.tensor(features_array,dtype=torch.float)
    edge_index = torch.tensor(edges,dtype=torch.long) 
    y = torch.tensor(targets.to_numpy().reshape((-1,1)),dtype=torch.float)
    
    dataset = Data(x=x,edge_index=edge_index,y=y)
    split = RandomNodeSplit(split='train_rest',num_test=0.1,num_val=0.1)
    dataset : Data = split(dataset)
    
    return dataset

if __name__ == "__main__":
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        data_source = "/kaggle/input/coe-datatest-2/features2-4-2025.xlsx"
    else:
        data_source = './data/excel_files/features2-4-2025.xlsx'
    dataset = create_features(data_source)
    train(data=dataset)
    
