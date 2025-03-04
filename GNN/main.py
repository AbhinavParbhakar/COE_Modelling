import os
import subprocess
import sys

# Check if running on Kaggle and install dependencies if not already installed
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    print("Started download for pytorch_geometric")
    import torch
    version = torch.__version__.split("+")[0]  # Extracts version without CUDA/CPU suffix
    install_cmd = [
        sys.executable, "-m", "pip", "install", "geopandas",
        "torch_geometric", "pyg_lib", "torch_scatter", "torcheval",
        "torch_sparse", "torch_cluster", "torch_spline_conv",
        "-f", f"https://data.pyg.org/whl/torch-{version}+cu121.html"
    ]
    subprocess.check_call(install_cmd)


import pandas as pd
import torch
import numpy as np
from shapely import Point, distance, LineString, MultiLineString
from shapely.ops import linemerge
import geopandas as gpd
import math

from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GCNConv, GAT
from torcheval.metrics import R2Score
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adagrad
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import tqdm


class GNN(nn.Module):
    def __init__(self,in_channels:int,hidden_num:int,output:int):
        super().__init__()
        
        print(hidden_num//4)
        print(hidden_num//16)
        print(hidden_num//64)
        
        
        self.l1 = GCNConv(in_channels=in_channels,out_channels=hidden_num)
        self.l2 = GCNConv(in_channels=hidden_num,out_channels=hidden_num//2)
        self.l3 = GCNConv(in_channels=hidden_num//2,out_channels=hidden_num//4)
        self.l4 = GCNConv(in_channels=hidden_num//4,out_channels=hidden_num//8)
        self.l5 = GCNConv(in_channels=hidden_num//8,out_channels=output)
        self.relu = nn.ReLU()
    
    def forward(self,data:Data):
        nodes, edges = data.x,data.edge_index
        x = self.l1(nodes,edges)
        x = self.relu(x)
        x = self.l2(x,edges)
        x = self.relu(x)
        x = self.l3(x,edges)
        x = self.relu(x)
        x = self.l4(x,edges)
        x = self.relu(x)
        x = self.l5(x,edges)
        
        return x

def train(data:Data,lr=0.01,epochs=1000):
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
    r2_scores = []
    valid_scores = []
    with open('results.txt','w') as file:
        for i in range(epochs):
            model.train()
            optim.zero_grad()
            output = model(data)
            loss = loss_fn(output[data.train_mask],data.y[data.train_mask])
            loss.backward()
            optim.step()
            
            if i % 200 == 0:
                with torch.no_grad():
                    output = model(data)
                    metric = R2Score()
                    mae = nn.L1Loss()
                    metric.update(output[data.val_mask],data.y[data.val_mask])
                    loss :torch.Tensor = mae(output[data.val_mask],data.y[data.val_mask])
                    loss = loss.item()
                    score = metric.compute().item()
                    metric.reset()
                    r2_scores.append(score)
                    valid_scores.append(loss)
                    file.write(f'R_2 Score after epoch {i} is {score}\n')
                    file.write(f'MAE after epoch {i} is {loss}\n\n\n')
    
        file.close()
        x_values = [i for i in range(len(r2_scores))]
        plt.figure(figsize=(10,6))
        plt.plot(x_values,r2_scores)
        plt.xlabel('Epoch (Every 200)')
        plt.ylabel('r2 Scores')
        plt.title("Validation r2 Scores over Epochs")
        plt.savefig('GNN_r2_scores.png')
        
        
        plt.figure(figsize=(10,6))
        plt.plot(x_values,valid_scores)
        plt.xlabel('Epoch (Every 200)')
        plt.ylabel('MAE Scores')
        plt.title("Validation MAE Scores over Epochs")
        plt.savefig('GNN_mae_scores.png')     
        
    

def preprocess_data(df:pd.DataFrame)->np.ndarray:
    """
    Using the given data frame containing the information for studies, apply StandardScaling for 
    the 'Lat, Long, Speed' columns. Apply OneHotEncoding for the 'roadclass, land usage' columns. 
    
    Then return the output as a nd.ndarray 
    """
    transform = ColumnTransformer(transformers=[
        ('Standard Scale',StandardScaler(),['Lat','Long','speed_First']),
        ('One Hot Encode',OneHotEncoder(),['roadclass','Aggregate_First','descriptio_First'])
    ])
    
    return transform.fit_transform(df)

    
def output_adj_matrix(df:pd.DataFrame,roadclass_gdf:gpd.GeoDataFrame)->np.ndarray:
    """
    Given the DataFrame and the GeoDataFrame, match each road segment, where
    the matching distance equals the value specified in the 'Proximity_Distance' col in the df.
    """
    df = df.copy(deep=True)
    roadclass_gdf = roadclass_gdf.copy(deep=True)
    CoE_crs = 3780
    
    
    # Create the geometry col in df    
    df['geometry'] = df.apply(lambda x : Point(x['Long'],x['Lat']),axis=1)
    features_gpd = gpd.GeoDataFrame(data=df,geometry='geometry',crs='EPSG:4326')
    features_gpd = features_gpd.to_crs(epsg=CoE_crs)
    roadclass_gdf = roadclass_gdf.to_crs(epsg=CoE_crs)
    roadclass_gdf['fake_geo'] = roadclass_gdf['geometry']
    
    # Join points to a road segment
    joined = features_gpd.sjoin_nearest(right=roadclass_gdf,how='inner',distance_col='distance')
     
    
    features_gpd['road_geo'] = joined['fake_geo']
    features_gpd = features_gpd.set_geometry('road_geo')
    features_gpd = features_gpd.drop('geometry',axis=1)
    features_gpd = features_gpd.to_crs(epsg=4326)
    features_gpd.to_csv('./data/excel_files/data_with_lines.csv',index=False)

    
    # # Elongate the points, by intersecting the features_lines with lines that they touch from the original
    # elongated_lines : gpd.GeoDataFrame = features_gpd.sjoin(df=roadclass_gdf,how='left',)
    # elongated_lines : gpd.GeoDataFrame = elongated_lines[elongated_lines.apply(lambda x: not x['road_geo'].equals(x['fake_geo']),axis=1)]
    
    # grouped_lines = elongated_lines.groupby('Estimation_point',as_index=False).apply(return_merged_line)
    
    
    # grouped_gf = gpd.GeoDataFrame(grouped_lines,geometry='merged_lines',crs=f'EPSG:{CoE_crs}')
    
    # joined_lines = features_gpd.merge(grouped_gf,on='Estimation_point',how='left')
    
    # missed_lines : gpd.GeoDataFrame = joined_lines[joined_lines['merged_lines'].isna()]
    # missed_lines['merged_lines'] = missed_lines['road_geo']
    
    # print(joined_lines.active_geometry_name)
    # joined_lines = joined_lines.set_geometry('merged_lines')
    
    # comparison_df = joined_lines.copy()
    # comparison_df['og_lines'] = comparison_df['merged_lines']
    
    # intersections = joined_lines.sjoin(comparison_df,how='left')
     
    # intersections = intersections[(intersections['merged_lines'] != intersections['og_lines'])]
     
    
    
    
    # pl = features_gpd.plot(figsize=(10,6))
    # features_gpd.to_file('.data/shape_files/extended_lines')

    
    # features_copy = features_gpd.copy(deep=True)
    # features_gpd['road_geo'] = features_gpd.geometry.buffer(100)
    # # features_gpd.to_file('./data/shape_files/buffer')
    
    # joined_results : gpd.GeoDataFrame = features_gpd.sjoin(df=features_gpd,how='inner')
    # cut_results : gpd.GeoDataFrame = joined_results[joined_results['Estimation_point_right'] != joined_results["Estimation_point_left"]]
    
    # cut_results.to_file('./data/shape_files/merged_results_final')

def return_merged_line(x):
    lines = MultiLineString(x['fake_geo'].to_list())
    merged_line = linemerge(lines)
    return pd.Series({'Estimation_point':x['Estimation_point'].unique().tolist()[0],'merged_lines':merged_line}) 

def create_features(file_source:str,shape_file:str)->Data:
    """
    Given a data source linking to the the excel file storing the data, create input for the GNN,
    and returns the graph data after splitting the features using RandomNode Split
    """
    df = pd.read_excel(file_source)  
     
    # Split X and Y
    targets = df['AAWDT']
    features = df.drop('AAWDT',axis=1)
    
    
    # Create the adjacency matrix
    roadclass_gdf = gpd.read_file(shape_file)
    output_adj_matrix(features,roadclass_gdf)
    
    # Scale and get the data as a np.array
    features_array = preprocess_data(features)

    
    
    # Create the edge_index
    source = []
    destination = []
    
    # for i in range(features_array.shape[0]):
    #     for j in range(features_array.shape[0]):
    #         cluster_i = clusters[i][0]
    #         cluster_j = clusters[j][0]
    #         if i != j and cluster_i == cluster_j:
                
    #             # Add one way
    #             source.append(i)
    #             destination.append(j)
                
    #             # Add the other way
    #             source.append(j)
    #             destination.append(i)
    
    # source_array = np.array(source).reshape((1,-1))
    # destination_array = np.array(destination).reshape((1,-1))
    # edges = np.concatenate((source_array,destination_array),axis=0)
    # print(edges.shape)  
    # print(features_array.shape) 
    
    # x = torch.tensor(features_array,dtype=torch.float)
    # edge_index = torch.tensor(edges,dtype=torch.long) 
    # y = torch.tensor(targets.to_numpy().reshape((-1,1)),dtype=torch.float)
    
    # dataset = Data(x=x,edge_index=edge_index,y=y)
    # split = RandomNodeSplit(split='train_rest',num_test=0.0,num_val=0.1)
    # dataset : Data = split(dataset)
    
    return None

if __name__ == "__main__":
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        data_source = "/kaggle/input/coe-datatest/coe_data_v3.xlsx"
        shape_file = '/kaggle/input/coe-datatest/RoadClass_CoE.shp'
    else:
        data_source = './data/excel_files/coe_data_v3.xlsx'
        shape_file = './data/shape_files/RoadClass_CoE.shp'
    dataset = create_features(data_source,shape_file)
    #train(data=dataset)
    
