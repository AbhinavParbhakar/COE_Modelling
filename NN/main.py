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
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error
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

    def forward(self,parametric_features:torch.FloatTensor,images:torch.FloatTensor):
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
            nn.Linear(8,64),
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

def train(model: torch.nn.Module, epochs: int, lr: float, batch_size: int, decay: float, train_data, test_data,validation_frequency=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    training_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)
    train_r2_values = []
    valid_r2_values = []
    train_rmse_values = []
    valid_rmse_values = []
    train_mape_values = []
    valid_mape_values = []
    epochs_values = []
    early_stop = False
    
    early_stopping_threshold = 10
    early_stopping_index = 1
    best_rmse = 400000
    i = 0
    with open('training.txt', 'w') as file:
        while i < epochs and not early_stop:
            model.train()
            all_targets, all_preds = [], []

            for coarse_input, granular_input, target in training_loader:
                optim.zero_grad()
                pred = model(coarse_input.to(device), granular_input.to(device))
                loss = torch.nn.functional.mse_loss(pred, target.to(device))
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
                for coarse_input, granular_input, target in test_loader:
                    pred = model(coarse_input.to(device), granular_input.to(device))
                    valid_targets.append(target.detach().cpu().numpy())
                    valid_preds.append(pred.detach().cpu().numpy())

            valid_targets = np.concatenate(valid_targets)
            valid_preds = np.concatenate(valid_preds)
            valid_loss = np.sqrt(mean_squared_error(valid_targets, valid_preds))
            valid_r2 = r2_score(valid_targets, valid_preds)
            valid_mape = mean_absolute_percentage_error(valid_targets, valid_preds)
            
            valid_mape_values.append(valid_mape * 100)
            valid_rmse_values.append(valid_loss)
            valid_r2_values.append(valid_r2)
            

            file.write(f'\n---------------------------\nValidation Epoch: {i}\n')
            file.write(f'r2 score is {valid_r2:.3f}\n')
            file.write(f'MAPE is {valid_mape * 100:.2f}%\n')
            file.write(f'RMSE is {valid_loss:.3f}\n')
            file.write('---------------------------\n')
            
            # Early Stopping Mechanism
            if valid_loss < best_rmse:
                best_rmse = valid_loss
                early_stopping_index = 1
            else:
                early_stopping_index += 1
            
            if early_stopping_index == early_stopping_threshold:
                early_stop = True
            i += 1    
    create_graph(epochs_values,[train_mape_values,valid_mape_values],"Multimodal Model MAPE","Epochs","MAPE (%)")
    create_graph(epochs_values,[train_rmse_values,valid_rmse_values],"Multimodal Model RMSE","Epochs","RMSE")
    create_graph(epochs_values,[train_r2_values,valid_r2_values],"Multimodal Model R2Score","Epochs","R2Score")
    

def create_graph(x_values:tuple,y_values:list[list],title:str,xlabel:str,ylabel:str):
    """
    Given the graph details, plot the graph and save it under ``<title>.png``. ENSURE THAT THE TRAINING Y VALUES ARE PLACED FIRST
    """
    plt.figure(figsize=(10,6))
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

def convert_images_to_numpy(image_path:str,excel_path:str)->np.ndarray:
    """
    Converts the images stored under the given parameter path into one large ``numpy.ndarray`` object.
    Uses the ordering of the inputs in the excel path to order the images in the outputted array.
    
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
    df = pd.read_csv(excel_path)
    file_names = df['Estimation_point'].tolist()
    absolute_path = os.path.abspath(image_path)
    nested_path_generator = os.walk(absolute_path)
    image_paths = None
    images_numpy = []
    
    
    for dirpath,dirnames,filenames in nested_path_generator:
        image_paths = {name.split('.')[0] : os.path.join(dirpath,name) for name in filenames}  
        
    for file_name in file_names:
        image_path = image_paths[str(file_name)]
        image_array = np.array(Image.open(image_path))
        images_numpy.append(image_array)
    
    return_numpy_array = np.array(images_numpy)
    return return_numpy_array

def preprocess_data(df:pd.DataFrame)->np.ndarray:
    """
    Using the given data frame containing the information for studies, apply StandardScaling for 
    the 'Lat, Long, Speed' columns. Apply OneHotEncoding for the 'roadclass, land usage' columns. 
    
    Then return the output as a nd.ndarray 
    """
    transform = ColumnTransformer(transformers=[
        ('Standard Scale',StandardScaler(),['Latitude','Longitude','Speed']),
        ('One Hot Encode',OneHotEncoder(),['Road_Class'])
    ])
    
    return transform.fit_transform(df).astype(np.float32)

def get_parametric_features(file_path:str)->np.ndarray:
    """
    Create a ``numpy.ndarray`` containing features.
    
    Parameters
    ----------
    excel_path : ``str``
        The path of the excel file to be opened. Contains the regression values per Study ID.    
    Returns
    -------
    ``numpy.ndarray``
        Numpy array corresponding to corresponding features.
    """
    df = pd.read_csv(file_path)
    df = df[['Latitude','Longitude','Road_Class','Speed']]
    
    features = preprocess_data(df)
    return features

def get_regression_values(file_path:str)->np.ndarray:
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
    df = pd.read_csv(file_path)
    
    targets = df['AAWDT'].values
    targets = targets.reshape((-1,1))
    return targets.astype(np.float32)

    
if __name__ == "__main__":
    batch_size = 16
    epochs = 50
    lr = 0.001
    l2 = 0.07
    val_freq=10
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        excel_file_path = "/kaggle/input/coe-cnn-experiment/duplicates_removed.csv"
        granular_image_path = "/kaggle/input/coe-cnn-experiment/granular images"
    else:
        granular_image_path = './data/granular images'
        excel_file_path = './data/excel_files/duplicates_removed.csv'
        
    images_ndarray = convert_images_to_numpy(granular_image_path,excel_file_path)
    parametric_features = get_parametric_features(excel_file_path)
    targets = get_regression_values(excel_file_path)
    
    np.random.set_state()
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
        
    train_set = TensorDataset(torch.from_numpy(x_train_features),torch.from_numpy(x_train_images).permute(0,3,1,2) / 255,torch.from_numpy(y_train))
    valid_set = TensorDataset(torch.from_numpy(x_test_features),torch.from_numpy(x_test_images).permute(0,3,1,2) / 255,torch.from_numpy(y_test))
            
    train(model=ImageParameterNet(),epochs=epochs,lr=lr,batch_size=batch_size,decay=l2,train_data=train_set,test_data=valid_set,validation_frequency = val_freq)
    