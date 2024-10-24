from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import torch

class autoLabelEncoder:
    """
    Do not think\\
    classic fit, transform \\
    you can get each encoder from get_encoder(feature_name)
    """
    def __init__(self) -> None:
        self.cat_encoders:dict = {}

    def fit(self, data:pd.DataFrame, categories:list[str]) -> None:
        for feat in categories:
            enc = LabelEncoder()
            self.cat_encoders[feat] = enc.fit(data.loc[data[feat].notna(), feat])

    def transform(self, data:pd.DataFrame, categories:list[str]) -> pd.DataFrame:
        for feat in categories:
            if(feat in data.columns):
                enc = self.cat_encoders[feat]
                data.loc[data[feat].notna(), feat] = enc.transform(data.loc[data[feat].notna(), feat]).astype(int)
        return data
    
    def get_encoder(self, category) -> LabelEncoder:
        return self.cat_encoders[category]
    

class wDataset(Dataset):
    """
    Simple dataset for weighted loss\\
    if weights = None than just normal torch dataset
    """
    def __init__(self, data, target, weights = None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.weights = None
        if weights is not None:
            self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.target[idx]
        if self.weights is not None:
            weight = self.weights[idx]
            return item, weight, label
        return item, label
    
    def set_weights(self,weights:np.array):
        self.weights = weights
    

def data_crusher(data:pd.DataFrame, target_name:str, split_size:float, random_state:int = 1, to = "numpy") -> pd.DataFrame:
    """
    Simple function for fast data fragmentation like: \\
    x_train, x_test, y_train, y_test \\
    or \\
    torchDataset_train, torchDataset_test - for torch.
    """
    target = data[target_name]
    data = data.drop(target_name, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=split_size, random_state=random_state)
    if to == "numpy":
        return np.array(X_train).astype(float), np.array(X_test).astype(float), np.array(y_train).astype(float), np.array(y_test).astype(float)
    if to == "torch":
        return wDataset(X_train, y_train), wDataset(X_test, y_test)
    
def cleaner(data:pd.DataFrame, features:list[str], quantile_lower:float, quantile_upper:float):
    """Easy way to destroy your data. Dropping values which are above quantile_upper, and below quantile_lower by creating intersection. """
    condition = pd.Series([True] * len(data), index=data.index)
    for feature in features:
        lower_quantile = data[feature].quantile(quantile_upper)
        upper_quantile = data[feature].quantile(quantile_lower)
        condition &= (data[feature] >= lower_quantile) & (data[feature] <= upper_quantile)
    return data.loc[condition]
    
class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, weights):
        squared_error = (predictions - targets) ** 2
        weighted_error = weights * squared_error
        return torch.mean(weighted_error)