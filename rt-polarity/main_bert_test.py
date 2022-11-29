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
from pytorch_datasets import PandasTextDataset, BertDataset
from baseline_models import *
from pytorch_trainer import PytorchModelTrainer, BertModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
import mlflow
from datetime import datetime

if __name__ == '__main__':

    train_data = openDfFromPickle("rt-polarity/rt-polarity-train-bert.pkl")
    val_data = openDfFromPickle("rt-polarity/rt-polarity-val-bert.pkl")
    test_data = openDfFromPickle("rt-polarity/rt-polarity-test-bert.pkl")
    
    train_dataset = BertDataset(train_data, "review_bert_attention_mask", "review_bert_input_ids", "label")
    train_dataloader = DataLoader(train_dataset, 64, shuffle=True)

    val_dataset = BertDataset(val_data, "review_bert_attention_mask", "review_bert_input_ids", "label")
    val_dataloader = DataLoader(val_dataset, 32, shuffle=False)

    test_dataset = BertDataset(test_data, "review_bert_attention_mask", "review_bert_input_ids", "label")
    test_dataloader = DataLoader(test_dataset, 32, shuffle=False)

    model = BERT_Baseline_Classifier(output_dim=1)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()
    trainer = BertModelTrainer()
    trainer.train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, n_epochs=1)
    print(trainer.test_model_w_loader(model, val_dataloader, "classification"))    

    raise NotImplementedError
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

    mlflow.set_experiment("rt-polarity-w-feed-forward-nns")
    now = str(datetime.now())
    mlflow.start_run(run_name="rnn_run_" + now)

    model = BERT_Baseline_Classifier(dropout=0.3)
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()
    trainer = PytorchModelTrainer()
    trainer.train_pytorch_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, use_bert_tokens=True, n_epochs=1)

    acc, prec, recall, f1 = trainer.test_pytorch_model_np(model, val_dataset.get_x(), val_dataset.get_y(), "classification", "binary")
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    mlflow.pytorch.log_model(model, "bert-model_"+now)
    mlflow.end_run()

    
    # TODO try and fix the dataset splitting methods
    
    
    
    

    



