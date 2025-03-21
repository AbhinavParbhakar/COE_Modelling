from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def svm(df:pd.DataFrame)->float:
    """
    Undergoes basic SVM setup to get hands dirty using the dataset
    """
    # Define target and features
    target = df['Volume']
    features = df.drop(['Volume', 'UniqueID',], axis=1)

    # Preprocessing
    transform = ColumnTransformer(transformers=[
        ('One Hot', OneHotEncoder(handle_unknown='ignore'), ["roadclass", "Land Usage"]),
        ("Standard Scale", StandardScaler(), ['Speed (km/h)','Lat','Long'])
    ])

    x = transform.fit_transform(features)
    y = target.to_numpy()

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale target variable
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Train model
    reg = SVR()
    reg.fit(x_train, y_train_scaled)

    # Predict and inverse transform
    pred_scaled = reg.predict(x_test)
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    # Evaluate
    return r2_score(y_test, pred)

def create_graph(features:np.ndarray):
    max_componenets = 30
    scores = []
    x = [i for i in range(1,max_componenets)]
    for n_comp in x:
        gm = GaussianMixture(n_components=n_comp,random_state=0)
        gm.fit(features)
        scores.append(gm.bic(features))
        
    plt.figure(figsize=(10,6))
    plt.plot(x,scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC')
    plt.title("BIC (Lower the better) vs n_clusters")
    plt.savefig('./data/output/bic.png')
        

def EM(df:pd.DataFrame)->pd.DataFrame:
    """
    Given the data frame, perform the EM algorithm using a gaussian mixture model approach
    
    Return a data frame with the cluster for each point attached
    """
    df = df.copy(deep=True)
    features = df.drop(['Volume', 'UniqueID'], axis=1)

    # Preprocessing
    transform = ColumnTransformer(transformers=[
        ('One Hot', OneHotEncoder(handle_unknown='ignore'), ["roadclass", "Land Usage"]),
        ("Standard Scale", StandardScaler(), ['Speed (km/h)', 'Lat', 'Long',])
    ])
    
    x = transform.fit_transform(features)
    create_graph(x)
    gm = GaussianMixture(n_components=16,random_state=0)
    gm.fit(x)
    df['Cluster'] = pd.Series(gm.predict(x))
    
    return df

if __name__ == '__main__':
    df = pd.read_excel('./data/excel_files/features1-31-2025.xlsx')
    cluster_df = EM(df)

