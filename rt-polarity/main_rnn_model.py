import sys
import platform
import mlflow

if platform.system() == "Windows":
    sys.path.append("C:\\Users\\mrbal\\Documents\\NLP\\NLP_Classification")
    print("Running on windows...")
else:
    sys.path.append("/Users/emrebaloglu/Documents/NLP/NLP_Classification")

from sklearn.metrics import classification_report
from preprocessing import *
import pandas as pd
import pickle
from pytorch_datasets import NumpyDataset
from baseline_models import Transformer_Baseline_Classifier, RNN_Baseline_Classifier
from pytorch_trainer import PytorchModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from datetime import datetime

if __name__ == '__main__':

    
    """ preprocessing the data """
    
    data = pd.read_csv("rt-polarity/rt-polarity-full.csv")
    data.columns = ['label', 'review']
    data_prc = process_df_texts(data, ["review"])
    data_tkn = tokenize_data(data, ["review"], preprocess=True)
    data_sqn = pad_tokenized_data(data_tkn, ["review_tokenized"])
    

    """ save the processed data in a pickle file """
    
    store_file = open("rt-processed-tokenized-padded.pkl", "ab")
    pickle.dump(data_sqn, store_file)
    store_file.close()
    
    """ read processed data from pickled file"""
    """store_file = open("./rt-polarity/rt-processed-tokenized-padded.pkl", "rb")
    data = pickle.load(store_file)
    store_file.close()"""

    X = np.stack(data_sqn["review_tokenized"].values)
    Y = data_sqn["label"].values

    
    dataset = NumpyDataset(X, Y, textual_data=True)
    VOCAB_SIZE = len(np.unique(X)) # dataset.get_vocab_size()
    INPUT_DIM = dataset.get_input_dim()
    OUTPUT_DIM = dataset.get_output_dim()

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    xval, xtest, yval, ytest = train_test_split(xtest, ytest, test_size=0.5, random_state=42)
    
    print(f"Vocabulary size: {VOCAB_SIZE}, input_dim: {INPUT_DIM}, output_dim: {OUTPUT_DIM}")

    """train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    val_size = test_size // 2
    test_size = test_size - val_size
    print(len(dataset))"""
    train_dataset = NumpyDataset(xtrain, ytrain, textual_data=True)
    val_dataset = NumpyDataset(xval, yval, textual_data=True)
    test_dataset = NumpyDataset(xtest, ytest, textual_data=True)
    # torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    print(type(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    now = str(datetime.now())
    mlflow.set_experiment("rt-polarity-w-feed-forward-nns")
    
    mlflow.start_run(run_name="rnn_run_" + now)

    rnn_params = {
    "embed_dim": 10,
    "rnn_type": "gru",
    "rnn_hidden_size": 4,
    "rnn_hidden_out": 2,
    "rnn_bidirectional": True,
    "units": 32
    }

    for name, val in rnn_params.items():
        mlflow.log_param(name, val)
    

    
    model = RNN_Baseline_Classifier(VOCAB_SIZE, INPUT_DIM, OUTPUT_DIM, **rnn_params)
    """
    5 - 250 
    6 - 300
    8 - 400
    """
    learning_rate = 0.001
    mlflow.log_param("learning rate", learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()
    trainer = PytorchModelTrainer()

    N_EPOCHS = 1
    mlflow.log_param("n epochs", N_EPOCHS)

    print(trainer.test_pytorch_model_np(model, val_dataset.get_x(), val_dataset.get_y(), "classification", "binary"))    
    trainer.train_pytorch_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, n_epochs=N_EPOCHS, use_mlflow=True)

    acc, prec, recall, f1 = trainer.test_pytorch_model_np(model, val_dataset.get_x(), val_dataset.get_y(), "classification", "binary")
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    mlflow.pytorch.log_model(model, "rnn-model_"+now)
    # ypred = np.array(model(X).detach().cpu().numpy() > 0.5).astype(int)
    # print(classification_report(Y, ypred))


    

    
    
    
    
    

    



