from ast import Num
from multiprocessing.sharedctypes import Value
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Union
import pickle


class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, textual_data: bool=False):
        """torch.Dataset class for using numpy.ndarray.

        Arguments:
            x -- The attributes of the data. (n_samples, n_features)
            y -- The targets of the data. (n_samples, n_classes)

        Keyword arguments:
            textual_data -- Whether the given data in x is tokenized text (default: {False})

        """
        super().__init__()
        
        if len(x) != len(y):
            raise ValueError("The parameters x and y must have the same number of samples.")
        
       
        
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

        if len(y.shape) == 1:
            self.y = torch.unsqueeze(self.y, 1)
        
        
        if textual_data:
            self.__vocab_size = len(np.unique(x))
        else:
            self.__vocab_size = 0

        if len(x.shape) == 1:
            self.__input_dim = 1
        else:
            self.__input_dim = x.shape[1]
        
        self.__output_dim = len(np.unique(y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = self.x[idx]
        target = self.y[idx]
        return data, target
    
    def get_vocab_size(self):
        return self.__vocab_size
    
    def get_input_dim(self):
        return self.__input_dim
    
    def get_output_dim(self):
        return self.__output_dim

class PandasDataset(Dataset):
    def __init__(self, data: Union[pd.DataFrame, str], feature_cols: list, target_cols: list):
        """torch.Dataset class for using pandas.DataFrame.

        Arguments:
            data -- Dataframe that contains the data. Either read from csv file (str) or from RAM (pd.DataFrame)
            The dataframe must only contain numeric values (int, float).
            feature_cols -- List of feature columns. (x)
            target_cols -- List of target columns. (y)
        Raises:
            TypeError or ValueError depending on the argument data.
        """
        super().__init__()
        if type(data) == str:
            if data.endswith(".csv") or data.endswith(".pkl"):
                data_ = pd.read_csv(data)
                self.x = torch.Tensor(data_[feature_cols].values)
                self.y = torch.Tensor(data_[target_cols].values)
                del data_
            elif data.endswith(".pkl"):
                tmp_file = open(data, "rb")
                data_ = pickle.load(tmp_file)
                self.x = torch.Tensor(data_[feature_cols].values)
                self.y = torch.Tensor(data_[target_cols].values)
                tmp_file.close()
            else:
                raise ValueError("A csv or pkl file must be given as string.")
        elif type(data) == pd.DataFrame:
            self.x = torch.Tensor(data[feature_cols].values)
            self.y = torch.Tensor(data[target_cols].values)
        else:
            raise TypeError("The argument data must of type str or pandas.DataFrame")
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = self.x[idx]
        target = self.y[idx]
        return data, target

if __name__ == "__main__":
    
    store_file = open("rt-processed-tokenized-padded", "rb")
    data = pickle.load(store_file)
    store_file.close()
    X = np.stack(data["review_tokenized"].values)
    Y = data["label"].values

    print(f"Shape of X: {X.shape}, shape of Y: {Y.shape}")
    dataset = NumpyDataset(X, Y)

    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    for i, data in enumerate(loader):
        xxx, yyy = data
        print(xxx, yyy)
        break