import sys

sys.path.append("/Users/emrebaloglu/Documents/NLP/NLP_Classification")

from sklearn.metrics import classification_report
from preprocessing import *
import pandas as pd
import pickle
from pytorch_datasets import *
from baseline_models import *
from pytorch_trainer import PytorchModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch

if __name__ == '__main__':

    
    """ preprocessing the data """
    
    """data = pd.read_csv("rt-polarity-full.csv")
    data.columns = ['label', 'review']
    data_prc = process_df_texts(data, ["review"])
    data_tkn = tokenize_data(data, ["review"], preprocess=True)
    data_sqn = pad_tokenized_data(data_tkn, ["review_tokenized"])
    """

    """ save the processed data in a pickle file """
    
    """store_file = open("rt-processed-tokenized-padded.pkl", "ab")
    pickle.dump(data_sqn, store_file)
    store_file.close()"""
    

    # data_sqn.to_csv("rt-polarity-processed.csv", index=False)
    
    """ read processed data from pickled file"""
    # data = pd.read_csv("rt-polarity-processed.csv")
    store_file = open("./rt-polarity/rt-processed-tokenized-padded.pkl", "rb")
    data = pickle.load(store_file)
    store_file.close()

    X = np.stack(data["review_tokenized"].values)
    Y = data["label"].values

    
    dataset = NumpyDataset(X, Y, textual_data=True)
    VOCAB_SIZE = dataset.get_vocab_size()
    INPUT_DIM = dataset.get_input_dim()
    OUTPUT_DIM = dataset.get_output_dim()

    print(f"Vocabulary size: {VOCAB_SIZE}, input_dim: {INPUT_DIM}, output_dim: {OUTPUT_DIM}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    val_size = test_size // 2
    test_size = test_size - val_size
    print(len(dataset))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], 
                                                                            generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Transformer_Baseline_Classifier(VOCAB_SIZE, INPUT_DIM, OUTPUT_DIM, 5, 5, embed_dim=5)
    """
    5 - 250 
    6 - 300
    8 - 400
    """
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()
    trainer = PytorchModelTrainer()
    trainer.train_pytorch_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, n_epochs=3)

    ypred = np.array(model(X).detach().cpu().numpy() > 0.5).astype(int)
    print(classification_report(Y, ypred))

    

    
    
    
    
    

    



