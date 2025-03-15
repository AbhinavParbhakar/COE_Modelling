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

def convert_images_to_numpy(path:str,use_shortened_verion=False)->np.ndarray:
    """
    Converts the images stored under the given parameter path into one large ``numpy.ndarray`` object.
    
    Parameters
    ----------
    
    path : ``str``
        Path of the folder containing images
    use_shortened_version : ``bool``
        (Default ``False``), If ``True``, then the internal dict storing the ids that Mingjian provided is used
    
    Returns
    -------
    
    ``numpy.ndarray``
        The numpy array containing the images
    """
    absolute_path = os.path.abspath(path)
    nested_path_generator = os.walk(absolute_path)
    image_paths = []
    images_numpy = []
    internal_dict = {'101290': True, '101294': True, '112259': True, '101262': True, '111951': True, '111952': True, '101266': True, '111829': True, '111827': True, '101190': True, '101186': True, '111781': True, '111783': True, '111525': True, '111527': True, '111521': True, '100648': True, '111687': True, '111689': True, '111821': True, '111823': True, '111873': True, '111947': True, '100492': True, '100498': True, '112755': True, '112753': True, '100768': True, '308916': True, '308924': True, '330089': True, '330085': True, '525230': True, '540731': True, '539279': True, '539283': True, '420384': True, '414388': True, '408952': True, '435005': True, '435729': True, '435737': True, '438977': True, '438981': True, '414356': True, '414348': True, '415980': True, '435163': True, '433265': True, '414284': True, '414292': True, '433949': True, '409880': True, '435133': True, '409888': True, '440765': True, '410112': True, '407404': True, '407410': True, '408144': True, '434341': True, '434345': True, '407396': True, '407390': True, '404264': True, '438941': True, '404320': True, '110401': True, '110403': True, '105332': True, '105336': True, '110745': True, '110719': True, '105400': True, '105404': True, '110715': True, '110717': True, '105062': True, '105066': True, '105086': True, '110953': True, '104342': True, '104346': True, '111127': True, '111007': True, '102806': True, '111165': True, '400024': True, '102766': True, '110631': True, '110713': True, '110473': True, '110397': True, '444801': True, '421800': True, '330639': True, '319454': True, '319460': True, '444649': True, '420429': True, '420431': True, '441277': True, '444757': True, '444761': True, '318040': True, '407628': True, '407636': True, '317908': True, '401440': True, '441237': True, '401376': True, '401384': True, '104526': True, '330001': True, '102648': True, '113135': True, '113137': True, '305784': True, '336069': True, '336073': True, '305824': True, '336521': True, '336525': True, '305816': True, '336513': True, '336517': True, '303500': True, '302252': True, '230458': True, '264581': True, '264575': True, '441301': True, '441305': True, '420392': True, '444929': True, '417904': True, '444201': True, '441285': True, '414396': True, '443765': True, '444749': True, '444753': True, '316724': True, '316732': True, '334861': True, '316652': True, '111745': True, '111747': True, '441229': True, '112957': True, '112955': True, '102992': True, '102996': True, '316650': True, '315656': True, '315945': True, '315918': True, '112275': True, '112277': True, '102972': True, '102976': True, '330349': True, '311160': True, '331929': True, '228466': True, '518876': True, '541763': True, '541299': True, '541303': True, '540511': True, '540515': True, '514280': True, '514292': True, '536767': True, '536771': True, '515348': True, '517564': True, '530747': True, '530755': True, '515960': True, '515176': True, '530565': True, '530739': True, '530743': True, '514712': True, '514704': True, '530735': True, '512952': True, '532459': True, '512012': True, '512004': True, '536763': True, '511354': True, '511360': True, '510288': True, '510296': True, '540499': True, '540495': True, '508456': True, '532443': True, '532439': True, '530695': True, '530699': True, '534751': True, '505456': True, '534747': True, '504768': True, '540479': True, '254915': True, '238006': True, '254909': True, '254903': True, '235864': True, '235876': True, '254885': True, '234280': True, '233926': True, '254513': True, '261923': True, '220120': True, '262075': True, '218590': True, '258983': True, '217930': True, '258959': True, '213448': True, '213460': True, '208516': True, '259535': True, '208338': True, '208342': True, '207256': True, '263033': True, '263065': True, '262439': True, '262441': True, '207232': True, '204752': True, '208504': True, '204760': True, '263531': True, '263525': True, '205450': True, '205438': True, '204748': True, '204740': True, '200116': True, '257201': True, '253313': True, '253307': True, '208690': True, '250853': True, '250859': True, '213364': True, '213354': True, '253325': True, '253331': True, '213388': True, '213400': True, '254735': True, '213412': True, '213424': True, '213436': True, '251411': True, '251417': True, '218038': True, '218026': True, '1004777': True, '1005262': True, '1005278': True, '1005286': True, '1006205': True, '1007587': True, '1008096': True, '1011636': True, '1013660': True, '1013711': True, '1013726': True, '1016796': True, '1016800': True, '1016667': True, '1016676': True, '1012625': True, '1012656': True, '1021899': True, '1023786': True, '1026214': True, '1026217': True, '982144': True, '984694': True, '1127352': True, '1124386': True, '1121502': True, '1060999': True, '1060528': True, '1097536': True, '1095321': True, '1095327': True, '1095441': True, '1099111': True, '1099113': True, '1099116': True, '1099120': True, '1101137': True, '1101203': True, '1113318': True, '1114507': True, '1119655': True, '1119698': True, '1119704': True, '1119747': True, '1119751': True, '1119759': True, '1000812': True, '1003043': True, '1004752': True, '1004769': True, '1004773': True, '1004780': True, '1004833': True, '1004857': True, '1004859': True, '1004876': True, '1004884': True, '1004895': True, '1004923': True, '1004933': True, '1004938': True, '1004958': True, '1004961': True, '1005245': True, '1005407': True, '1005409': True, '1005415': True, '1005419': True, '1005420': True, '1005421': True, '1005423': True, '1005623': True, '1005629': True, '1005646': True, '1005678': True, '1011493': True, '1011495': True, '1011525': True, '1007832': True, '1007849': True, '1011531': True, '1011554': True, '1011568': True, '1011626': True, '1026241': True, '1012576': True, '1012612': True, '1013443': True, '1013450': True, '1016359': True, '1016366': True, '1016370': True, '1016373': True, '1016615': True, '1016812': True, '1016816': True, '1017285': True, '1017057': True, '1029418': True, '1017474': True, '1019480': True, '1019462': True, '1013129': True, '1022029': True, '1002142': True, '1125265': True, '1117960': True, '1121507': True, '1061528': True, '1063539': True, '1072212': True, '1095072': True, '1095077': True, '1095334': True, '1095337': True, '1099109': True, '1107903': True, '1107905': True, '1107910': True, '1107914': True, '1119787': True, '1118021': True, '1118045': True, '1118229': True, '1105278': True, '1105596': True, '1000814': True, '1000817': True, '1003025': True, '1004737': True, '1005275': True, '1005290': True, '1005302': True, '1005305': True, '1006053': True, '1004765': True, '1003054': True, '1003060': True, '1003066': True, '1005279': True, '1005281': True, '1005284': True, '1005285': True, '1005292': True, '1005293': True, '1005297': True, '1005329': True, '1005333': True, '1005412': True, '1005425': True, '1005426': True, '1005427': True, '1005428': True, '1005585': True, '1005681': True, '1005686': True, '1005726': True, '1005739': True, '1011492': True, '1006216': True, '1006226': True, '1006229': True, '1007564': True, '1007576': True, '1007611': True, '1007623': True, '1011543': True, '1011562': True, '1012607': True, '1012666': True, '1013388': True, '1013406': True, '1013447': True, '1013639': True, '1013691': True, '1013706': True, '1013770': True, '1016352': True, '1016360': True, '1016396': True, '1016801': True, '1016810': True, '1016401': True, '1016803': True, '1017065': True, '1023782': True, '1011487': True, '1012597': True, '1016346': True, '1017448': True, '1017472': True, '1019484': True, '1020796': True, '1019408': True, '1019454': True, '1019470': True, '1012647': True, '1012652': True, '1012774': True, '1012785': True, '1012811': True, '1013134': True, '1013266': True, '1014039': True, '1025814': True, '1025818': True, '965308': True, '965310': True, '965314': True, '965476': True, '965483': True, '965491': True, '965719': True, '965725': True, '965731': True, '965763': True, '966018': True, '962850': True, '962854': True, '962887': True, '962893': True, '963182': True, '963184': True, '963259': True, '963269': True, '963279': True, '963321': True, '963326': True, '963332': True, '1012508': True, '1012514': True, '1123634': True, '1123639': True, '1123645': True, '1123694': True, '1123704': True, '1125237': True, '1135943': True, '1139399': True, '1139400': True, '1139402': True, '1139461': True, '1141443': True, '1131481': True, '1131470': True, '1131503': True, '1060214': True, '1060218': True, '1060220': True, '1063544': True, '1063548': True, '1072217': True, '1077852': True, '1091636': True, '1091640': True, '1091664': True, '1091670': True, '1093045': True, '1095446': True, '1108360': True, '1105250': True, '1004743': True, '1004854': True, '1004889': True, '1005424': True, '1005431': True, '1005689': True, '1005691': True, '1011516': True, '1013698': True, '1016339': True, '1016785': True, '1016791': True, '1016806': True, '1016639': True, '1016640': True, '1016651': True, '1016687': True, '1019365': True, '1019376': True, '1019388': True, '1019487': True, '1019490': True, '1019493': True, '1019530': True, '1012778': True, '1013260': True, '1012624': True, '1012645': True, '1012653': True, '1014678': True, '1016320': True, '959988': True, '960019': True, '960025': True, '960032': True, '960037': True, '960043': True, '960049': True, '977131': True, '966615': True, '966709': True, '966715': True, '966722': True, '967188': True, '967565': True, '967568': True, '967572': True, '967577': True, '1120986': True, '1123614': True, '1130815': True, '1130705': True, '1061533': True, '1063576': True, '1072734': True, '1072739': True, '1072750': True, '1072752': True, '1072759': True, '1072851': True, '1072856': True, '1072875': True, '1072887': True, '1073858': True, '1072185': True, '1072192': True, '1072199': True, '1073898': True, '1077416': True, '1094455': True, '1094463': True, '1094467': True, '1095313': True, '1095481': True, '1101147': True, '1114474': True, '1114494': True, '1120598': True, '1119767': True, '1005247': True, '1005251': True, '1005256': True, '1005258': True, '1028258': True, '1005430': True, '1005753': True, '1006019': True, '1006022': True, '1006023': True, '1011488': True, '1013704': True, '1013710': True, '1016644': True, '1017469': True, '1017470': True, '1017471': True, '1020799': True, '1019516': True, '1012659': True, '1016327': True, '959977': True, '960054': True, '965172': True, '965178': True, '965214': True, '965241': True, '965516': True, '965522': True, '965527': True, '965531': True, '960449': True, '960450': True, '960451': True, '960454': True, '960455': True, '960493': True, '960497': True, '962711': True, '982828': True, '1123629': True, '1125226': True, '1127371': True, '1060213': True, '1060469': True, '1063554': True, '1073553': True, '1077813': True, '1091028': True, '1095325': True, '1101131': True, '1112010': True}


    
    for dirpath,dirnames,filenames in nested_path_generator:
        for name in filenames:
            file_id = name.split('.')[0]
            if file_id in internal_dict and use_shortened_verion:
                image_paths.append(os.path.join(dirpath,name))     
    
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
    
    return transform.fit_transform(df).astype(np.float32)

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

def generate_target_values_numpy(excel_path:str,image_path:str,use_shortened_verion=False)->tuple[np.ndarray, np.ndarray]:
    """
    Create a ``numpy.ndarray`` containing parallel matched regression values for the images
    included in the provided folder, matching based on name of image to the row in the excel file.
    
    Parameters
    ----------
    excel_path : ``str``
        The path of the excel file to be opened. Contains the regression values per Study ID.
    image_path : ``str``
        The path of the folder containing the images. Labled with study IDs.
    use_shortened_version : ``bool``
        (Default ``False``), If ``True``, then the internal dict storing the ids that Mingjian provided is used
    
    Returns
    -------
    ``tuple[numpy.ndarray, numpy.ndarray]``
        Tuple of numpy array corresponding to corresponding features and the target values matched with given images. 
    """
    absolute_path = os.path.abspath(image_path)
    df = pd.read_excel(excel_path)
    nested_path_generator = os.walk(absolute_path)
    image_id_array = []
    regression_values = []
    internal_dict = {'101290': True, '101294': True, '112259': True, '101262': True, '111951': True, '111952': True, '101266': True, '111829': True, '111827': True, '101190': True, '101186': True, '111781': True, '111783': True, '111525': True, '111527': True, '111521': True, '100648': True, '111687': True, '111689': True, '111821': True, '111823': True, '111873': True, '111947': True, '100492': True, '100498': True, '112755': True, '112753': True, '100768': True, '308916': True, '308924': True, '330089': True, '330085': True, '525230': True, '540731': True, '539279': True, '539283': True, '420384': True, '414388': True, '408952': True, '435005': True, '435729': True, '435737': True, '438977': True, '438981': True, '414356': True, '414348': True, '415980': True, '435163': True, '433265': True, '414284': True, '414292': True, '433949': True, '409880': True, '435133': True, '409888': True, '440765': True, '410112': True, '407404': True, '407410': True, '408144': True, '434341': True, '434345': True, '407396': True, '407390': True, '404264': True, '438941': True, '404320': True, '110401': True, '110403': True, '105332': True, '105336': True, '110745': True, '110719': True, '105400': True, '105404': True, '110715': True, '110717': True, '105062': True, '105066': True, '105086': True, '110953': True, '104342': True, '104346': True, '111127': True, '111007': True, '102806': True, '111165': True, '400024': True, '102766': True, '110631': True, '110713': True, '110473': True, '110397': True, '444801': True, '421800': True, '330639': True, '319454': True, '319460': True, '444649': True, '420429': True, '420431': True, '441277': True, '444757': True, '444761': True, '318040': True, '407628': True, '407636': True, '317908': True, '401440': True, '441237': True, '401376': True, '401384': True, '104526': True, '330001': True, '102648': True, '113135': True, '113137': True, '305784': True, '336069': True, '336073': True, '305824': True, '336521': True, '336525': True, '305816': True, '336513': True, '336517': True, '303500': True, '302252': True, '230458': True, '264581': True, '264575': True, '441301': True, '441305': True, '420392': True, '444929': True, '417904': True, '444201': True, '441285': True, '414396': True, '443765': True, '444749': True, '444753': True, '316724': True, '316732': True, '334861': True, '316652': True, '111745': True, '111747': True, '441229': True, '112957': True, '112955': True, '102992': True, '102996': True, '316650': True, '315656': True, '315945': True, '315918': True, '112275': True, '112277': True, '102972': True, '102976': True, '330349': True, '311160': True, '331929': True, '228466': True, '518876': True, '541763': True, '541299': True, '541303': True, '540511': True, '540515': True, '514280': True, '514292': True, '536767': True, '536771': True, '515348': True, '517564': True, '530747': True, '530755': True, '515960': True, '515176': True, '530565': True, '530739': True, '530743': True, '514712': True, '514704': True, '530735': True, '512952': True, '532459': True, '512012': True, '512004': True, '536763': True, '511354': True, '511360': True, '510288': True, '510296': True, '540499': True, '540495': True, '508456': True, '532443': True, '532439': True, '530695': True, '530699': True, '534751': True, '505456': True, '534747': True, '504768': True, '540479': True, '254915': True, '238006': True, '254909': True, '254903': True, '235864': True, '235876': True, '254885': True, '234280': True, '233926': True, '254513': True, '261923': True, '220120': True, '262075': True, '218590': True, '258983': True, '217930': True, '258959': True, '213448': True, '213460': True, '208516': True, '259535': True, '208338': True, '208342': True, '207256': True, '263033': True, '263065': True, '262439': True, '262441': True, '207232': True, '204752': True, '208504': True, '204760': True, '263531': True, '263525': True, '205450': True, '205438': True, '204748': True, '204740': True, '200116': True, '257201': True, '253313': True, '253307': True, '208690': True, '250853': True, '250859': True, '213364': True, '213354': True, '253325': True, '253331': True, '213388': True, '213400': True, '254735': True, '213412': True, '213424': True, '213436': True, '251411': True, '251417': True, '218038': True, '218026': True, '1004777': True, '1005262': True, '1005278': True, '1005286': True, '1006205': True, '1007587': True, '1008096': True, '1011636': True, '1013660': True, '1013711': True, '1013726': True, '1016796': True, '1016800': True, '1016667': True, '1016676': True, '1012625': True, '1012656': True, '1021899': True, '1023786': True, '1026214': True, '1026217': True, '982144': True, '984694': True, '1127352': True, '1124386': True, '1121502': True, '1060999': True, '1060528': True, '1097536': True, '1095321': True, '1095327': True, '1095441': True, '1099111': True, '1099113': True, '1099116': True, '1099120': True, '1101137': True, '1101203': True, '1113318': True, '1114507': True, '1119655': True, '1119698': True, '1119704': True, '1119747': True, '1119751': True, '1119759': True, '1000812': True, '1003043': True, '1004752': True, '1004769': True, '1004773': True, '1004780': True, '1004833': True, '1004857': True, '1004859': True, '1004876': True, '1004884': True, '1004895': True, '1004923': True, '1004933': True, '1004938': True, '1004958': True, '1004961': True, '1005245': True, '1005407': True, '1005409': True, '1005415': True, '1005419': True, '1005420': True, '1005421': True, '1005423': True, '1005623': True, '1005629': True, '1005646': True, '1005678': True, '1011493': True, '1011495': True, '1011525': True, '1007832': True, '1007849': True, '1011531': True, '1011554': True, '1011568': True, '1011626': True, '1026241': True, '1012576': True, '1012612': True, '1013443': True, '1013450': True, '1016359': True, '1016366': True, '1016370': True, '1016373': True, '1016615': True, '1016812': True, '1016816': True, '1017285': True, '1017057': True, '1029418': True, '1017474': True, '1019480': True, '1019462': True, '1013129': True, '1022029': True, '1002142': True, '1125265': True, '1117960': True, '1121507': True, '1061528': True, '1063539': True, '1072212': True, '1095072': True, '1095077': True, '1095334': True, '1095337': True, '1099109': True, '1107903': True, '1107905': True, '1107910': True, '1107914': True, '1119787': True, '1118021': True, '1118045': True, '1118229': True, '1105278': True, '1105596': True, '1000814': True, '1000817': True, '1003025': True, '1004737': True, '1005275': True, '1005290': True, '1005302': True, '1005305': True, '1006053': True, '1004765': True, '1003054': True, '1003060': True, '1003066': True, '1005279': True, '1005281': True, '1005284': True, '1005285': True, '1005292': True, '1005293': True, '1005297': True, '1005329': True, '1005333': True, '1005412': True, '1005425': True, '1005426': True, '1005427': True, '1005428': True, '1005585': True, '1005681': True, '1005686': True, '1005726': True, '1005739': True, '1011492': True, '1006216': True, '1006226': True, '1006229': True, '1007564': True, '1007576': True, '1007611': True, '1007623': True, '1011543': True, '1011562': True, '1012607': True, '1012666': True, '1013388': True, '1013406': True, '1013447': True, '1013639': True, '1013691': True, '1013706': True, '1013770': True, '1016352': True, '1016360': True, '1016396': True, '1016801': True, '1016810': True, '1016401': True, '1016803': True, '1017065': True, '1023782': True, '1011487': True, '1012597': True, '1016346': True, '1017448': True, '1017472': True, '1019484': True, '1020796': True, '1019408': True, '1019454': True, '1019470': True, '1012647': True, '1012652': True, '1012774': True, '1012785': True, '1012811': True, '1013134': True, '1013266': True, '1014039': True, '1025814': True, '1025818': True, '965308': True, '965310': True, '965314': True, '965476': True, '965483': True, '965491': True, '965719': True, '965725': True, '965731': True, '965763': True, '966018': True, '962850': True, '962854': True, '962887': True, '962893': True, '963182': True, '963184': True, '963259': True, '963269': True, '963279': True, '963321': True, '963326': True, '963332': True, '1012508': True, '1012514': True, '1123634': True, '1123639': True, '1123645': True, '1123694': True, '1123704': True, '1125237': True, '1135943': True, '1139399': True, '1139400': True, '1139402': True, '1139461': True, '1141443': True, '1131481': True, '1131470': True, '1131503': True, '1060214': True, '1060218': True, '1060220': True, '1063544': True, '1063548': True, '1072217': True, '1077852': True, '1091636': True, '1091640': True, '1091664': True, '1091670': True, '1093045': True, '1095446': True, '1108360': True, '1105250': True, '1004743': True, '1004854': True, '1004889': True, '1005424': True, '1005431': True, '1005689': True, '1005691': True, '1011516': True, '1013698': True, '1016339': True, '1016785': True, '1016791': True, '1016806': True, '1016639': True, '1016640': True, '1016651': True, '1016687': True, '1019365': True, '1019376': True, '1019388': True, '1019487': True, '1019490': True, '1019493': True, '1019530': True, '1012778': True, '1013260': True, '1012624': True, '1012645': True, '1012653': True, '1014678': True, '1016320': True, '959988': True, '960019': True, '960025': True, '960032': True, '960037': True, '960043': True, '960049': True, '977131': True, '966615': True, '966709': True, '966715': True, '966722': True, '967188': True, '967565': True, '967568': True, '967572': True, '967577': True, '1120986': True, '1123614': True, '1130815': True, '1130705': True, '1061533': True, '1063576': True, '1072734': True, '1072739': True, '1072750': True, '1072752': True, '1072759': True, '1072851': True, '1072856': True, '1072875': True, '1072887': True, '1073858': True, '1072185': True, '1072192': True, '1072199': True, '1073898': True, '1077416': True, '1094455': True, '1094463': True, '1094467': True, '1095313': True, '1095481': True, '1101147': True, '1114474': True, '1114494': True, '1120598': True, '1119767': True, '1005247': True, '1005251': True, '1005256': True, '1005258': True, '1028258': True, '1005430': True, '1005753': True, '1006019': True, '1006022': True, '1006023': True, '1011488': True, '1013704': True, '1013710': True, '1016644': True, '1017469': True, '1017470': True, '1017471': True, '1020799': True, '1019516': True, '1012659': True, '1016327': True, '959977': True, '960054': True, '965172': True, '965178': True, '965214': True, '965241': True, '965516': True, '965522': True, '965527': True, '965531': True, '960449': True, '960450': True, '960451': True, '960454': True, '960455': True, '960493': True, '960497': True, '962711': True, '982828': True, '1123629': True, '1125226': True, '1127371': True, '1060213': True, '1060469': True, '1063554': True, '1073553': True, '1077813': True, '1091028': True, '1095325': True, '1101131': True, '1112010': True}

    
    for dirpath,dirnames,filenames in nested_path_generator:
        for name in filenames:
            file_id = name.split('.')[0]
            if file_id in internal_dict and use_shortened_verion:
                image_id_array.append(file_id)  
        
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
    
    return features, targets.astype(np.float32)


    
if __name__ == "__main__":
    batch_size = 16
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        excel_file_path = "/kaggle/input/coe-cnn-experiment/base_data.xlsx"
        granular_image_path = "/kaggle/input/coe-cnn-experiment/granular images"
    else:
        granular_image_path = './data/granular images'
        excel_file_path = './data/excel_files/base_data.xlsx'
        
    images_ndarray = convert_images_to_numpy(granular_image_path,True)
    parametric_features,targets = generate_target_values_numpy(excel_file_path,granular_image_path,True)
    

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
    