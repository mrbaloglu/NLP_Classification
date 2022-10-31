import sys
import platform

if platform.system() == "Windows":
    sys.path.append("C:\\Users\\mrbal\\Documents\\NLP\\NLP_Classification")
    print("Running on windows...")
else:
    sys.path.append("/Users/emrebaloglu/Documents/NLP/NLP_Classification")

from sklearn.metrics import classification_report
from preprocessing import *
import pandas as pd
import pickle
from pytorch_datasets import PandasTextDataset
from baseline_models import *
from pytorch_trainer import PytorchModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':

    
    train_dataset = PandasTextDataset("rt-polarity/rt-polarity-train.csv", ["review"], ["label"]) 
    val_dataset = PandasTextDataset("rt-polarity/rt-polarity-val.csv", ["review"], ["label"])
    test_dataset = PandasTextDataset("rt-polarity/rt-polarity-test.csv", ["review"], ["label"]) 
    """VOCAB_SIZE = dataset.get_vocab_size()
    INPUT_DIM = dataset.get_input_dim()
    OUTPUT_DIM = dataset.get_output_dim()

    print(f"Vocabulary size: {VOCAB_SIZE}, input_dim: {INPUT_DIM}, output_dim: {OUTPUT_DIM}")"""
 
    """ train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    val_size = test_size // 2
    test_size = test_size - val_size
    print(len(dataset))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], 
                                                                            generator=torch.Generator().manual_seed(42))
    """
    """train_dataset, val_dataset = dataset.split_dataset(test_size=0.2, random_state=42)
    val_dataset, test_dataset = dataset.split_dataset(test_size=0.5, random_state=42)"""

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BERT_Baseline_Classifier(dropout=0.3)
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()
    trainer = PytorchModelTrainer()
    trainer.train_pytorch_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, use_bert_tokens=True, n_epochs=1)

    
    # TODO try and fix the dataset splitting methods
    
    
    
    

    



