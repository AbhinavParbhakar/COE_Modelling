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

# Check if running on Kaggle and install dependencies if not already installed
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    print("Started download for pytorch_geometric")
    import torch
    version = torch.__version__.split("+")[0]  # Extracts version without CUDA/CPU suffix
    install_cmd = [
        sys.executable, "-m", "pip", "install", "torchgeo"]
    subprocess.check_call(install_cmd)

from torchgeo.models import resnet18,ResNet18_Weights

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
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
        
        return self.flatten(x1)

class GranularImageModel(nn.Module):
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
    def __init__(self,):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
        self.parametric_module = NN()
        self.coarse_module = CoarseImageModel()
        self.granular_module = GranularImageModel()
        self.fc1 = nn.Linear(in_features=2012,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=200)
        self.fc3 = nn.Linear(in_features=200,out_features=50)
        self.fc4 = nn.Linear(in_features=50,out_features=1)
        
    
        
    def forward(self,coarse_input:torch.FloatTensor,granular_input:torch.FloatTensor,parametric_input):
        coarse_image_embedding = self.coarse_module(coarse_input)
        granular_image_embedding = self.granular_module(granular_input)
        parametric_embeddings = self.parametric_module(parametric_input)
        parametric_embeddings = self.relu(parametric_embeddings)

        combination = torch.cat(tensors=(coarse_image_embedding,granular_image_embedding,parametric_embeddings),dim=1)
        combination = self.dropout(combination)
        
        output = self.fc1(combination)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.fc4(output)
        
        return output
    

def convert_images_to_numpy(image_path:str,ordering:list)->np.ndarray:
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

def generate_target_values_numpy(file_path:str)->np.ndarray:
    """
    Create a ``numpy.ndarray`` containing the AAWDT values for each sample in the provided file
    
    Parameters
    ----------
    file_path : ``str``
        The path of the excel/csv file to be opened. Contains the regression values per Study ID.
    
    Returns
    -------
    ``numpy.ndarray``
        Numpy array corresponding to the target values matched with given images. 
    """
    df = pd.read_csv(file_path)
    df = df.drop(df[df['Road_Class'] == 'Alley'].index,axis=0)
    regression_values = df['AAWDT'].values
    
    return regression_values.reshape((-1,1)).astype(np.float32)

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
    


def train(model: torch.nn.Module, epochs: int, lr: float, batch_size: int, decay: float, train_data, test_data,):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = torch.nn.functional.huber_loss
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = CosineAnnealingLR(optimizer=optim,T_max=15,eta_min=0.0001)
    training_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=True)
    train_r2_values = []
    valid_r2_values = []
    train_rmse_values = []
    valid_rmse_values = []
    train_mape_values = []
    valid_mape_values = []
    epochs_values = []
    early_stop = False
    
    early_stopping_threshold = 30
    early_stopping_index = 1
    best_rmse = 400000
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
            if valid_loss < best_rmse:
                best_rmse = valid_loss
                checkpoint = {"Saved Model":model.state_dict()}
                best_preds = valid_preds
                best_targets = valid_targets
                early_stopping_index = 1
            else:
                early_stopping_index += 1
            
            if early_stopping_index == early_stopping_threshold:
                early_stop = True
            i += 1
    
    create_graph(epochs_values,[train_mape_values,valid_mape_values],"Multimodal MAPE","Epochs","MAPE (%)")
    create_graph(epochs_values,[train_rmse_values,valid_rmse_values],"Multimodal RMSE","Epochs","RMSE")
    create_graph(epochs_values,[train_r2_values,valid_r2_values],"Multimodal R2Score","Epochs","R2Score")
    create_graph([i + 1 for i in range(best_targets.shape[0])],y_values=[best_targets.reshape(best_targets.shape[0]),best_preds.reshape(best_preds.shape[0])],title="Ground Truth",xlabel="Data Point",ylabel="AAWDT",ground_truth=True)
    save_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_copy = MultimodalFullModel().to(device=device).load_state_dict(checkpoint['Saved Model'])
    torch.save(model_copy,f'{save_name}.pth')

def preprocess_data(df:pd.DataFrame)->np.ndarray:
    """
    Using the given data frame containing the information for studies, apply StandardScaling for 
    the 'Lat, Long, Speed' columns. Apply OneHotEncoding for the 'roadclass, land usage' columns. 
    
    Then return the output as a nd.ndarray 
    """
    transform = ColumnTransformer(transformers=[
        ('Standard Scale',StandardScaler(),['Latitude','Longitude','Speed',]),
        ('One Hot',OneHotEncoder(),['Road_Class'])
    ])
    
    return transform.fit_transform(df).astype(np.float32)

def get_parametric_features(file_path:str, train_split=0.85)->pd.DataFrame:
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
    unique_split_values = df['Road_Class'].unique()
    train_groupings = []
    test_groupings = []
    
    for unique_val in unique_split_values:
        unique_val_df = df[df['Road_Class'] == unique_val]
        train_index = int(unique_val_df.shape[0] * train_split)
        train_data = unique_val_df[:train_index]
        test_data = unique_val_df[train_index:]
        train_groupings.append(train_data)
        test_groupings.append(test_data)
    
    train_df = pd.concat(train_groupings,axis=0)
    test_df = pd.concat(test_groupings,axis=0)
    df = pd.concat([train_df,test_df],axis=0)
    
    return df

def get_regression_values(file_path:str,ordering:list)->np.ndarray:
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

def graph_ground_truth(x_values:tuple,y_values:list[list],title:str,xlabel:str,ylabel:str):
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
    
    

def create_graph(x_values:tuple,y_values:list[list],title:str,xlabel:str,ylabel:str,ground_truth=False):
    """
    Given the graph details, plot the graph and save it under ``<title>.png``. ENSURE THAT THE TRAINING Y VALUES ARE PLACED FIRST if ``ground_truth`` = ``False`` (default).
    """
    plt.figure(figsize=(10,6))
    
    if ground_truth:
        graph_ground_truth(x_values=x_values,y_values=y_values,title=title,xlabel=xlabel,ylabel=ylabel)
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
    

if __name__ == "__main__":
    os_name = os.name
    if os_name == 'nt':
        coarse_image_path = "./data/coarse images"
        granular_image_path = './data/granular images'
        excel_path = './data/excel_files/duplicates_removed.csv'
    else:
        coarse_image_path = "/kaggle/input/coe-cnn-Experiment/coarse images"
        granular_image_path = "/kaggle/input/coe-cnn-Experiment/granular images"
        excel_path = "/kaggle/input/coe-cnn-Experiment/duplicates_removed.csv"
    
    
    # Hyper parameters
    epochs = 200
    lr = 0.005
    batch_size = 16
    l2_decay = 0.05
    training_split = 0.85
    model = MultimodalFullModel()
    
    # Load data
    parametric_features_df = get_parametric_features(excel_path)
    orderings = parametric_features_df['Estimation_point'].tolist()
    granular_images_ndarray = convert_images_to_numpy(image_path=granular_image_path, ordering=orderings)
    
    coarse_images_ndarray = convert_images_to_numpy(image_path=coarse_image_path, ordering=orderings)
    
    aawdt_ndarray = get_regression_values(file_path=excel_path,ordering=orderings)
 
    # Shuffle data
    # random_permutation = np.random.permutation(granular_images_ndarray.shape[0])
    
    # granular_images_ndarray = granular_images_ndarray[random_permutation]
    
    # coarse_images_ndarray = coarse_images_ndarray[random_permutation]
    
    # aawdt_ndarray = aawdt_ndarray[random_permutation]
    
    # parametric_features_ndarray = parametric_features_ndarray[random_permutation]
    
    # Split data
    training_split_index = int(granular_images_ndarray.shape[0] * training_split)
    
    param_train_df = parametric_features_df[:training_split_index]
    param_test_df = parametric_features_df[training_split_index:]
    
    param_train = preprocess_data(param_train_df)
    param_test = preprocess_data(param_test_df)
    granular_train, coarse_train,  = granular_images_ndarray[:training_split_index],coarse_images_ndarray[:training_split_index],
    
    granular_test, coarse_test, = granular_images_ndarray[training_split_index:],coarse_images_ndarray[training_split_index:],
    
    aawdt_train, aawdt_test = aawdt_ndarray[:training_split_index],aawdt_ndarray[training_split_index:]

    
    train_dataset = TensorDataset(
        torch.from_numpy(coarse_train).permute(0,3,1,2) / 255,
        torch.from_numpy(granular_train).permute(0,3,1,2) / 255,
        torch.from_numpy(param_train),
        torch.from_numpy(aawdt_train)
        )
    test_dataset = TensorDataset(
        torch.from_numpy(coarse_test).permute(0,3,1,2) / 255,
        torch.from_numpy(granular_test).permute(0,3,1,2) / 255,
        torch.from_numpy(param_test),
        torch.from_numpy(aawdt_test)
        )
    
    
    train(model=model,epochs=epochs,lr=lr,batch_size=batch_size,decay=l2_decay,train_data=train_dataset,test_data=test_dataset)
    
    