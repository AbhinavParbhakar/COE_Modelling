import torch
from torch.utils.data import DataLoader,Dataset, TensorDataset
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
from PIL import Image
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

class ImageParameterNet(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=7,stride=2),
            nn.BatchNorm2d(num_features=15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=15,out_channels=7,kernel_size=7),
            nn.BatchNorm2d(num_features=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=7,out_channels=3,kernel_size=7),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.parameteric_1 = nn.Sequential(
            nn.Linear(in_features=8,out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        
        self.parameteric_2 = nn.Sequential(
            nn.Linear(in_features=64,out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        
        self.parameteric_3 = nn.Sequential(
            nn.Linear(in_features=128,out_features=312),
            nn.BatchNorm1d(num_features=312),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=675,out_features=320),
            nn.BatchNorm1d(320),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(320,120),
            nn.BatchNorm1d(120),
            nn.ReLU()
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(120,40),
            nn.BatchNorm1d(40),
            nn.ReLU()
        )
        
        self.fc4 = nn.Linear(40,1)

    def forward(self,images:torch.FloatTensor,parametric_features:torch.FloatTensor):
        x1 = self.conv1(images)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        
        x1 = self.flatten(x1)
        
        x2 = self.parameteric_1(parametric_features)
        x2 = self.parameteric_2(x2)
        x2 = self.parameteric_3(x2)
        
        combined_x = torch.cat(tensors=(x1,x2),dim=1)
        
        output = self.fc1(combined_x)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        
        return output
        

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(16,64),
            nn.ReLU(),
            
            nn.Linear(64,32),
            nn.ReLU(),
            
            nn.Linear(32,16),
            nn.ReLU(),
            
            nn.Linear(16,8),
            nn.ReLU(),
            
            nn.Linear(8,1),
        )
    
    def forward(self,x):
        x = self.net(x)
        
        return x

def train(model:nn.Module,train_loader:DataLoader,val_loader:DataLoader,cluster_name="None")->None:
    loss_fun = mse_loss
    epochs = 200
    lr = 0.001
    l2 = 0.07
    val_freq=10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_r2 = R2Score()
    val_r2 = R2Score()
    
    model = model.to(device=device)
    
    optim = Adam(model.parameters(),lr=lr,weight_decay=l2)
    
    val_mae_scores = []
    train_mae_scores = []
    with open(f'nn-results_cluster{cluster_name}.txt','w') as file:
        for i in range(epochs):
            model.train()
            for features,images,targets in train_loader:
                optim.zero_grad()
                features = features.to(device)
                images = images.to(device)
                targets = targets.to(device)
                
                output = model(features,images)
                loss = loss_fun(output,targets)
                loss.backward()
                optim.step()
            
            if i % int(epochs/val_freq) == 0:
                with torch.no_grad():
                    val_mae = 0
                    train_mae = 0
                    mae = torch.nn.functional.l1_loss
                    
                    for features,images, targets in train_loader:
                        features = features.to(device)
                        images = images.to(device)
                        targets = targets.to(device)
                        
                        output = model(features,images)
                        train_mae += mae(output,targets).item()
                        train_r2.update(output,targets)
                    
                    
                    for features, images, targets in val_loader:
                        features = features.to(device)
                        images = images.to(device)
                        targets = targets.to(device)
                        
                        output = model(features,images)
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
        plt.savefig(f'NN_mae_scores.png')

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
        image_array = np.float32(np.array(Image.open(image_path)))
        images_numpy.append(image_array.reshape((3,image_array.shape[0],image_array.shape[1])))
    
    return_numpy_array = np.array(images_numpy)
    return return_numpy_array

def preprocess_data(df:pd.DataFrame)->np.ndarray:
    """
    Using the given data frame containing the information for studies, apply StandardScaling for 
    the 'Lat, Long, Speed' columns. Apply OneHotEncoding for the 'roadclass, land usage' columns. 
    
    Then return the output as a nd.ndarray 
    """
    transform = ColumnTransformer(transformers=[
        ('Standard Scale',StandardScaler(),['Lat','Long','speed_First']),
        ('One Hot Encode',OneHotEncoder(),['roadclass',])
    ])
    
    return transform.fit_transform(df)

def create_features(file_source:str,image_source:str):
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
    em = GaussianMixture(n_components=12,random_state=0)
    em.fit(features_array)
    clusters = em.predict(features_array)
    clusters = clusters.reshape((-1,1))
    
    temp_features = np.concatenate([features_array,clusters],axis=1)
    
    columns = [str(i) for i in range(temp_features.shape[1])]
    data = {col:temp_features[:,int(col)] for col in columns}
    data["targets"] = targets.to_numpy()
    temp_df = pd.DataFrame(data=data)
    print(temp_df.head())    
    clusters = []
    
    for name,data in iter(temp_df.groupby(by=["16"],as_index=False)):
        clusters.append(data.reset_index(drop=True))
    
    # encoded_clusters = OneHotEncoder().fit_transform(clusters).toarray()
    # features = np.concatenate([encoded_clusters,features_array],axis=1)
    # targets = targets.to_numpy().reshape((-1,1))

    return clusters

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
        values = df[df['Estimation_point'] == int(image_id)].values
        regression_values.append(values[-1])
    
    data =  np.array(regression_values)
    columns = df.columns.to_list()
    
    ordered_df = pd.DataFrame(data=data,columns=columns)
    targets = ordered_df['AAWDT'].values
    targets = targets.reshape((-1,1))
    ordered_df = ordered_df[['Lat','Long','speed_First','roadclass']]
    
    features = preprocess_data(ordered_df)
    print(features.shape)
    
    return features, targets


    
if __name__ == "__main__":
    batch_size = 16
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        excel_file_path = "/kaggle/input/coe-cnn-Experiment/base_data.xlsx"
        granular_image_path = "/kaggle/input/coe-cnn-Experiment/granular images"
    else:
        granular_image_path = './data/granular images'
        excel_file_path = './data/excel_files/base_data.xlsx'
        
    images_ndarray = convert_images_to_numpy(granular_image_path)
    parametric_features,targets = generate_target_values_numpy(excel_file_path,granular_image_path)
    ordering = permutation(parametric_features.shape[0])
    
    shuffled_images = images_ndarray[ordering]
    shuffled_features = parametric_features[ordering]
    shuffled_targets = targets[ordering]
    test_indices = parametric_features.shape[0] // 5
    
    x_test_features = shuffled_features[:test_indices]
    x_test_images = shuffled_images[:test_indices]
    y_test = shuffled_targets[:test_indices]
    
    x_train_features = shuffled_features[test_indices:]
    x_train_images = shuffled_images[test_indices:]
    y_train = shuffled_targets[test_indices:]
    
    train_set = TensorDataset(torch.from_numpy(x_train_features),torch.from_numpy(x_train_images),torch.from_numpy(y_train))
    valid_set = TensorDataset(torch.from_numpy(x_test_features),torch.from_numpy(x_test_images),torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=True)
        
    train(ImageParameterNet(),train_loader,valid_loader)
    