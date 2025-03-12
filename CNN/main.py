import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
import subprocess
import sys

# # Check if running on Kaggle and install dependencies if not already installed
# if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
#     print("Started download for pytorch_geometric")
#     import torch
#     version = torch.__version__.split("+")[0]  # Extracts version without CUDA/CPU suffix
#     install_cmd = [
#         sys.executable, "-m", "pip", "install", "satlaspretrain-models"
#     ]
#     subprocess.check_call(install_cmd)

# import satlaspretrain_models

torch.manual_seed(25)

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
        
        self.fc1 = nn.Linear(in_features=self.output_size,out_features=500)
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
        
        attention_scores = self.cross_attention(main_vector=granular_image_embedding,cross_vector=coarse_image_embedding,hidden_size=self.hidden_size,output_size=self.output_size)
        output = self.fc1(attention_scores)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu()
        
        output = self.fc4(output)
        
        return output
    
    def cross_attention(self,main_vector:torch.FloatTensor,cross_vector:torch.FloatTensor,hidden_size=500,output_size=1000)->torch.Tensor:
        """
        Implement cross attention
        """
        Q = self.w_q(cross_vector)
        K = self.w_k(main_vector)
        V = self.w_v(main_vector) # N * output_size
        
        attention = self.softmax( torch.mm(Q * torch.t(K)) / hidden_size ** 0.5) # Size N * N
        
        attention_scores = torch.mm(attention,V)
        
        return attention_scores
        

        
        


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
    def __init__(self,coarse_images:np.ndarray,granular_images:np.ndarray,targets:np.ndarray):
        self.coarse_images = torch.from_numpy(coarse_images)
        self.granular_images = torch.from_numpy(granular_images)
        self.y = torch.from_numpy(targets)
        
    def __len__(self):
        return self.coarse_images.shape[0]

    def __getitem__(self, index):
        return self.coarse_images[index], self.granular_images[index], self.y[index]

def train(model:nn.Module,epochs:int,lr:float,batch_size:int,decay:float,train_data:Dataset,test_data:Dataset)->None:
    """
    Train the provided model based on the hyperparameters given. This function
    utilizes the ``torch.optim.Adam`` optimizer internally for backpropogation.
    
    Parameters
    ----------
    model : ``torch.nn.Module``
        The model to be trained
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
    
    model = model.to(device=device)
    optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=decay)
    training_loader = DataLoader(dataset=train_data,batch_size=batch_size)
    test_loader = DataLoader(dataset=test_data,batch_size=batch_size)
    with open('training.txt','w') as file:
        for i in range(epochs):
            model.train()
        
            training_loss = 0 # rmse
            r2 = 0
            for coarse_input,granular_input, target in training_loader:
                optim.zero_grad()
                coarse_input = coarse_input.to(device)
                granular_input = granular_input.to(device)
                target = target.to(device)
                pred = model(coarse_input,granular_input)
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
        coarse_image_path = "./data/coarse images"
        granular_image_path = './data/granular images'
        excel_path = './data/excel_files/base_data.xlsx'
    else:
        coarse_image_path = "/kaggle/input/coe-cnn-Experiment/coarse images"
        granular_image_path = "/kaggle/input/coe-cnn-Experiment/granular images"
        excel_path = "/kaggle/input/coe-cnn-Experiment/base_data.xlsx"
    
    
    # Hyper parameters
    epochs = 100
    lr = 0.005
    batch_size = 10
    l2_decay = 0.005
    training_split = 0.85
    model = CrossAttentionCNN()
    
    # Load data
    granular_images_ndarray = convert_images_to_numpy(path=granular_image_path)
    
    coarse_images_ndarray = convert_images_to_numpy(path=coarse_image_path)
    
    aawdt_ndarray = generate_target_values_numpy(excel_path=excel_path,image_path=granular_image_path)
 
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
    
    train_dataset = ImageDataset(coarse_images=coarse_train,granular_images=granular_train,targets=aawdt_train)
    
    test_dataset = ImageDataset(coarse_images=coarse_test,granular_images=granular_test,targets=aawdt_test)
    
    
    # test_dataset = ImageDataset(images=x_test,targets=y_test)
    # # model = GranularCNN()
    # weights_manager = satlaspretrain_models.Weights()
    # # model = weights_manager.get_pretrained_model(model_identifier='Sentinel2_Resnet50_SI_RGB',device='cpu')
    
    # # for name,module in model.named_children():
    # #     print(name)
    train(model=model,epochs=epochs,lr=lr,batch_size=batch_size,decay=l2_decay,train_data=train_dataset,test_data=test_dataset)
    