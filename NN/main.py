import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import pandas as pd
from torch.nn.functional import mse_loss
from torch.optim import Adagrad, Adam
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import permutation
import os
import subprocess
import sys

torch.manual_seed(25)

if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    print("Started download for pytorch_geometric")
    import torch
    version = torch.__version__.split("+")[0]  # Extracts version without CUDA/CPU suffix
    install_cmd = [
        sys.executable, "-m", "pip", "install",
        "torcheval"
    ]
    subprocess.check_call(install_cmd)
from torcheval.metrics import R2Score

class ModelDataset(Dataset):
    def __init__(self,features:np.ndarray,targets:np.ndarray):
        self.x = torch.tensor(features,dtype=torch.float)
        self.y = torch.tensor(targets,dtype=torch.float)
        
    def __len__(self)->int:
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(32,160),
            nn.ReLU(),
            
            nn.Linear(160,80),
            nn.ReLU(),
            
            nn.Linear(80,40),
            nn.ReLU(),
            
            nn.Linear(40,20),
            nn.ReLU(),
            
            nn.Linear(20,10),
            nn.ReLU(),
            
            nn.Linear(10,1)
        )
    
    def forward(self,x):
        x = self.net(x)
        
        return x

def train(model:nn.Module,train_loader:DataLoader,val_loader:DataLoader,)->None:
    loss_fun = mse_loss
    epochs = 200
    lr = 0.0005
    l2 = 0.07
    val_freq=10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_r2 = R2Score()
    val_r2 = R2Score()
    
    model = model.to(device=device)
    
    optim = Adam(model.parameters(),lr=lr,weight_decay=l2)
    
    val_mae_scores = []
    train_mae_scores = []
    with open('nn-results.txt','w') as file:
        for i in range(epochs):
            model.train()
            for features,targets in train_loader:
                optim.zero_grad()
                features = features.to(device)
                targets = targets.to(device)
                
                output = model(features)
                loss = loss_fun(output,targets)
                loss.backward()
                optim.step()
            
            if i % int(epochs/val_freq) == 0:
                with torch.no_grad():
                    val_mae = 0
                    train_mae = 0
                    mae = torch.nn.functional.l1_loss
                    
                    for features, targets in train_loader:
                        features = features.to(device)
                        targets = targets.to(device)
                        output = model(features)
                        train_mae += mae(output,targets).item()
                        train_r2.update(output,targets)
                    
                    
                    for features, targets in val_loader:
                        features = features.to(device)
                        targets = targets.to(device)
                        output = model(features)
                        score = mae(output,targets)
                        val_r2.update(output,targets)
                        val_mae += score.item()
                    
                    train_r2_score = train_r2.compute().item()
                    valid_r2_score = val_r2.compute().item()
                    train_r2.reset()
                    val_r2.reset()
                    val_mae = val_mae/len(val_loader)
                    train_mae = train_mae/len(train_loader)
                    file.write(f'\n-----------------------------------------------\n')
                    file.write(f'Training R2 after epoch{i}: {train_r2_score:.3f}\n')
                    file.write(f'Training MAE after epoch{i}: {train_mae:.3f}\n')
                    file.write(f'\nValidation R2 after epoch{i}: {valid_r2_score:.3f}\n')
                    file.write(f'Validation MAE after epoch {i}: {val_mae:.3f}\n')
                    file.write(f'\n-----------------------------------------------\n')
                    val_mae_scores.append(val_mae)
                    train_mae_scores.append(train_mae)

        file.close()
        
        epochs = [i + 1 for i in range(len(val_mae_scores))]
        plt.figure(figsize=(10,6))
        plt.plot(epochs,val_mae_scores,"bo-",label="Val. MAE")
        plt.plot(epochs,train_mae_scores,"ro-",label="Train MAE")
        plt.xlabel('Epochs')
        plt.ylabel('MAE Score')
        plt.legend()
        plt.grid(visible=True,color='k')
        plt.title("Training and Validation MAE")
        plt.savefig('NN_mae_scores.png')
                    

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

def create_features(file_source:str):
    """
    Given a data source linking to the the excel file storing the data, create input for the NN,
    and returns the data as a tuple, containing (targets, labels)
    """
    df = pd.read_excel(file_source)
    
    # Split X and Y
    features = df.drop(labels=['Volume','UniqueID','Date'],axis=1)
    targets = df['Volume']
    
    # Scale and get the data as a np.array
    features_array = preprocess_data(features)
    em = GaussianMixture(n_components=16,random_state=0)
    em.fit(features_array)
    clusters = em.predict(features_array)
    clusters = clusters.reshape((-1,1))
    encoded_clusters = OneHotEncoder().fit_transform(clusters).toarray()
    features = np.concatenate([encoded_clusters,features_array],axis=1)
    targets = targets.to_numpy().reshape((-1,1))

    return (features,targets)


    
if __name__ == "__main__":
    batch_size = 16
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        file_path = "/kaggle/input/coe-datatest-2/features2-4-2025.xlsx"
    else:
        file_path = './data/excel_files/features2-4-2025.xlsx'
    features,targets = create_features(file_path)
    ordering = permutation(features.shape[0])
    shuffled_features = features[ordering]
    shuffled_targets = targets[ordering]
    test_indices = features.shape[0] // 5
    
    x_test = shuffled_features[:test_indices]
    y_test = shuffled_targets[:test_indices]
    
    x_train = shuffled_features[test_indices:]
    y_train = shuffled_targets[test_indices:]
    
    x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.33, random_state=42)
    
    test_set = ModelDataset(x_test,y_test)
    train_set = ModelDataset(x_train,y_train)
    valid_set = ModelDataset(x_valid,y_valid)
    
    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=True)
    
    train(NN(),train_loader,valid_loader)
    