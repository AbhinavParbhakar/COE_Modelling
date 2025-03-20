import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rotate
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import subprocess
import sys

# Check if running on Kaggle and install dependencies if not already installed
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    print("Started download for pytorch_geometric")
    import torch
    version = torch.__version__.split("+")[0]  # Extracts version without CUDA/CPU suffix
    install_cmd = [
        sys.executable, "-m", "pip", "install", "torchgeo"]
    subprocess.check_call(install_cmd)

from torchgeo.models import resnet18,ResNet18_Weights

torch.manual_seed(25)

class FinetunedModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.pretrained = resnet18(ResNet18_Weights.SENTINEL2_RGB_MOCO)
        
        
        for parameter in self.pretrained.parameters():
            parameter.requires_grad = True
        
        self.relu = nn.ReLU()
        self.processes_one = nn.Linear(in_features=1000,out_features=300)
        self.processes_two = nn.Linear(in_features=1000,out_features=300)
        self.fc1 = nn.Linear(in_features=600,out_features=200)
        self.fc2 = nn.Linear(in_features=200,out_features=50)
        self.fc3 = nn.Linear(in_features=50,out_features=1)
        
        self.dropout = nn.Dropout1d()

        
    def forward(self,coarse_input:torch.FloatTensor,granular_input:torch.FloatTensor):
        coarse_input = self.relu(self.pretrained(coarse_input))
        granular_input = self.relu(self.pretrained(granular_input))
        
        coarse_input = self.dropout(coarse_input)
        coarse_input = self.processes_one(coarse_input)
        coarse_input = self.relu(coarse_input)
        
        granular_input = self.dropout(granular_input)
        granular_input = self.processes_two(granular_input)
        granular_input = self.relu(granular_input)
        
        
        combination = torch.cat((coarse_input,granular_input),dim=1)
        
        output = self.fc1(combination)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        
        return output

class CrossAttentionCNN(nn.Module):
    def __init__(self, hidden_size = 500,output_size=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        # Based on the paper I read, make such that both dimesions are roughly the same
        # Based on the ResNet-18 architecture.
        self.coarse_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=10,stride=3),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=3,stride=3)
        )

        self.granular_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=15,kernel_size=7,stride=2),
            nn.BatchNorm2d(num_features=15),
            nn.MaxPool2d(kernel_size=3,stride=2)
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

        self.w_q = nn.Linear(in_features=960,out_features=hidden_size,bias=False)
        self.w_k = nn.Linear(in_features=540,out_features=hidden_size,bias=False)
        self.w_v = nn.Linear(in_features=540,out_features=output_size,bias=False)
        self.softmax = nn.Softmax(dim=1)
        
        self.fc1 = nn.Linear(in_features=1500,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=200)
        self.fc3 = nn.Linear(in_features=200,out_features=50)
        self.fc4 = nn.Linear(in_features=50,out_features=1)
    
        
    def forward(self,coarse_input:torch.FloatTensor,granular_input:torch.FloatTensor):
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
        
        # Process granular input
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
        
        coarse_image_embedding = self.flatten(x1)
        granular_image_embedding = self.flatten(x2)
        
        # attention_scores = self.cross_attention(main_vector=granular_image_embedding,cross_vector=coarse_image_embedding,hidden_size=self.hidden_size,output_size=self.output_size)
        combination = torch.cat(tensors=(coarse_image_embedding,granular_image_embedding),dim=1)
        output = self.fc1(combination)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        
        output = self.fc4(output)
        
        return output
    
    def cross_attention(self,main_vector:torch.FloatTensor,cross_vector:torch.FloatTensor,hidden_size=500,output_size=1000)->torch.Tensor:
        """
        Implement cross attention
        """
        Q = self.w_q(cross_vector)
        K = self.w_k(main_vector)
        V = self.w_v(main_vector) # N * output_size
        
        attention = self.softmax( torch.mm(Q , torch.t(K)) / hidden_size ** 0.5) # Size N * N
        
        attention_scores = torch.mm(attention,V)
        
        return attention_scores

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
    with open('training.txt', 'a') as file:
        for i in range(epochs):
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

            if i % (epochs // 10) == 0:
                epochs_values.append(i)
                train_rmse_values.append(training_loss)
                train_mape_values.append(mape)
                train_r2_values.append(r2)
                file.write(f'\n---------------------------\nTraining Epoch: {i}\n')
                file.write(f'r2 score is {r2:.3f}\n')
                file.write(f'MAPE is {mape * 100:.2f}%\n')
                file.write(f'RMSE is {training_loss:.3f}\n')
                file.write('---------------------------\n')

            # Validation
            if i % (epochs // 10) == 0:
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
                
                valid_mape_values.append(valid_mape)
                valid_rmse_values.append(valid_loss)
                valid_r2_values.append(valid_r2)
                
                file.write(f'\n---------------------------\nValidation Epoch: {i}\n')
                file.write(f'r2 score is {valid_r2:.3f}\n')
                file.write(f'MAPE is {valid_mape * 100:.2f}%\n')
                file.write(f'RMSE is {valid_loss:.3f}\n')
                file.write('---------------------------\n')
    
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
    epochs = 50
    lr = 0.0005
    batch_size = 16
    l2_decay = 0.005
    training_split = 0.85
    model = CrossAttentionCNN()
    
    # Load data
    granular_images_ndarray = convert_images_to_numpy(image_path=granular_image_path, excel_path=excel_path)
    
    coarse_images_ndarray = convert_images_to_numpy(image_path=coarse_image_path, excel_path=excel_path)
    
    aawdt_ndarray = generate_target_values_numpy(file_path=excel_path)
 
    # Shuffle data
    np.random.seed(0)
    random_permutation = np.random.permutation(granular_images_ndarray.shape[0])
    
    granular_images_ndarray = granular_images_ndarray[random_permutation]
    
    coarse_images_ndarray = coarse_images_ndarray[random_permutation]
    
    aawdt_ndarray = aawdt_ndarray[random_permutation]
    
    # Split data
    training_split_index = int(granular_images_ndarray.shape[0] * training_split)
    
    granular_train, coarse_train = granular_images_ndarray[:training_split_index],coarse_images_ndarray[:training_split_index]
    
    granular_test, coarse_test = granular_images_ndarray[training_split_index:],coarse_images_ndarray[training_split_index:]
    
    aawdt_train, aawdt_test = aawdt_ndarray[:training_split_index],aawdt_ndarray[training_split_index:]
    
    train_dataset = ImageDataset(coarse_images=coarse_train,granular_images=granular_train,targets=aawdt_train,)
    
    test_dataset = ImageDataset(coarse_images=coarse_test,granular_images=granular_test,targets=aawdt_test)
    
    
    train(model=model,epochs=epochs,lr=lr,batch_size=batch_size,decay=l2_decay,train_data=train_dataset,test_data=test_dataset)
    