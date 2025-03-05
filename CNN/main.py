import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
from torchvision import transforms
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
torch.seed()

class GranularCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            # 256 * 256 * 3
            
            nn.Conv2d(in_channels=3,out_channels=20,kernel_size=5),
            nn.BatchNorm2d(num_features=20),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=20,out_channels=30,kernel_size=10),
            nn.BatchNorm2d(num_features=30),
            nn.ReLU(),
            nn.MaxPool2d(2))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=20,kernel_size=9),
            nn.BatchNorm2d(num_features=20),
            nn.ReLU(),
            nn.MaxPool2d(2))
            
            # 47 * 47 * 20
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=10,kernel_size=10),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.MaxPool2d(2))
            
            # 20 * 20 * 10
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=5, kernel_size=5),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(2))
            
            # 9 * 9 * 5
        self.fc = nn.Sequential(
            nn.Flatten(),
            # (N, 405)
            nn.Linear(in_features=640, out_features=260),
            nn.ReLU(),
            
            nn.Linear(260,90),
            nn.ReLU(),
            
            nn.Linear(90, 30),
            nn.ReLU(),
            
            nn.Linear(30,10),
            nn.ReLU(),
            
            nn.Linear(10,1))
        
    def forward(self,x):
        x = self.layer1 (x)
        x = self.layer2 (x)
        x = self.layer3 (x)
        x = self.layer4 (x)
        #x = self.layer5 (x)
        x = self.fc(x)
        return x

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
        images_numpy.append(np.float32(np.array(Image.open(image_path)).reshape((3,256,256))))
    
    return_numpy_array = np.array(images_numpy)
    return return_numpy_array

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
        values = df['AAWDT'][df['Estimation_point'] == int(image_id)].values
        regression_values.append(values[-1])
    
    return np.array(regression_values).reshape((-1,1)).astype(np.float32)

class ImageDataset(Dataset):
    def __init__(self,images:np.ndarray,targets:np.ndarray):
        image_transform = transforms.Compose([transforms.ToTensor()])
        self.x = torch.from_numpy(images)
        self.y = torch.from_numpy(targets)
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index],self.y[index]

def train_zoomedCNN(epochs:int,lr:float,batch_size:int,decay:float,train_data:Dataset,test_data:Dataset)->None:
    """
    Train the ``GranularCNN`` based on the hyperparameters given. This function
    utilizes the ``torch.optim.Adam`` optimizer internally for backpropogation.
    
    Parameters
    ----------
    epochs : ``int``
        The number of iterations to apply during training.
    lr : ``float``
        The learning rate to apply during gradient descent.
    batch_size : ``int``
        Batch size to use during training.
    decary : ``float``
        The L2 regularization to apply during training
    train_data : ``torch.utils.Dataset``
        Training data
    test_data : ``torch.utils.Dataset``
        Test data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GranularCNN().to(device=device)
    optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=decay)
    training_loader = DataLoader(dataset=train_data,batch_size=batch_size)
    test_loader = DataLoader(dataset=test_data,batch_size=batch_size)
    with open('training.txt','w') as file:
        for i in range(epochs):
            model.train()
        
            training_loss = 0 # rmse
            r2 = 0
            for input, target in training_loader:
                optim.zero_grad()
                input = input.to(device)
                target = target.to(device)
                pred = model(input)
                loss = torch.nn.functional.mse_loss(pred,target)
                training_loss += loss.item() ** (0.5)
                r2 += r2_score(y_true=target.numpy(force=True),y_pred=pred.numpy(force=True))
                loss.backward()
                optim.step()
                
            if i % int(epochs/10) == 0:
                training_loss = training_loss/len(training_loader)
                r2 = r2/len(training_loader)
                file.write('\n---------------------------\n')
                file.write(f'Training Epoch: {i}\n')
                file.write(f'r2 score is {r2:.3f}\n')
                file.write(f'RMSE is {training_loss:.3f}\n')
                file.write('---------------------------\n')
            
            if i % int(epochs/10) == 0:
                valid_loss = 0 # rmse
                valid_r2 = 0
                for input, target in test_loader:
                    with torch.no_grad():
                        input = input.to(device)
                        target = target.to(device)
                        pred = model(input)
                        loss = torch.nn.functional.mse_loss(pred,target)
                        valid_loss += loss.item() ** (0.5)
                        valid_r2 += r2_score(y_true=target.numpy(force=True),y_pred=pred.numpy(force=True))
                    
                valid_loss = valid_loss/len(test_loader)
                valid_r2 = valid_r2/len(test_loader)
                file.write('\n---------------------------\n')
                file.write(f'Validation Epoch: {i}\n')
                file.write(f'r2 score is {valid_r2:.3f}\n')
                file.write(f'RMSE is {valid_loss:.3f}\n')
                file.write('---------------------------\n')
                
        
    

if __name__ == "__main__":
    os_name = os.name
    if os_name == 'nt':
        image_path = "./data/images"
        excel_path = './data/excel_files/base_data.xlsx'
    else:
        image_path = "/kaggle/input/coe-cnn-Experiment/Images"
        excel_path = "/kaggle/input/coe-cnn-Experiment/base_data.xlsx"
    
    epochs = 100
    lr = 0.005
    batch_size = 10
    l2_decay = 0.005
    
    images_ndarray = convert_images_to_numpy(path=image_path)
    aawdt_ndarray = generate_target_values_numpy(excel_path=excel_path,image_path=image_path)
    
    x_train,x_test,y_train,y_test = train_test_split(images_ndarray,aawdt_ndarray,test_size=0.15)
    train_dataset = ImageDataset(images=x_train,targets=y_train)
    test_dataset = ImageDataset(images=x_test,targets=y_test)
    train_zoomedCNN(epochs=epochs,lr=lr,batch_size=batch_size,decay=l2_decay,train_data=train_dataset,test_data=test_dataset)
    