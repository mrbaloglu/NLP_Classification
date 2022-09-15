import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
from torchtext.vocab import GloVe



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if device == torch.device("cpu"):
    print("Using cpu.")
else:
    print("Using gpu.")

class RNN_Baseline_Classifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 input_dim: int,
                 out_num_classes: int,
                 embed_dim: int = 5,
                 rnn_type: str = "gru",
                 rnn_hidden_size: int = 2,
                 rnn_hidden_out: int = 2,
                 rnn_bidirectional: bool = True,
                 units: int = 50):
        """
        RNN Class for text classification.
        Arguments:
            vocab_size -- specifies the size of the vocabulary to be used in word embeddings.
            input_dim -- specifies the dimension of the features (n_samples, max_sentence_length (input_dim)).
            out_num_classes -- specifies the number of classes for the output. 
            (for binary classification this argument must be 2)

        Keyword Arguments:
            embed_dim -- embed_dim specifies the embedding dimension for the categorical part of the input. (default: {5})
            rnn_type -- specifies the type of the recurrent layer for word embeddings. (default: {"gru"})
            rnn_hidden_size -- specifies the number of stacked recurrent layers. (default: {2})
            rnn_hidden_out -- specifies number of features in the hidden state h of recurrent layer. (default: {2})
            rnn_bidirectional -- specifies whether the recurrent layers be bidirectional. (default: {True})
            units -- specifies the number of neurons in the hidden layers. (default: {50})

        """
        super(RNN_Baseline_Classifier, self).__init__()
        self.embed_dim = embed_dim
        self.out_num_classes = out_num_classes

        self.embed_enc = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_out, rnn_hidden_size,
                          bidirectional=rnn_bidirectional, batch_first=True)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_out, rnn_hidden_size,
                               bidirectional=rnn_bidirectional, batch_first=True)
        elif rnn_type != "gru":
            raise ValueError("The argument rnn_type must be 'gru' or 'lstm'!")

        rnn_out_dim = rnn_hidden_out * input_dim
        if rnn_bidirectional:
            rnn_out_dim *= 2

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(rnn_out_dim, units)
        self.fc2 = nn.Linear(units, units)
        if self.out_num_classes == 2:
            self.fc3 = nn.Linear(units, 1)
        else:
            self.fc3 = nn.Linear(units, out_num_classes)

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Function for model architecture.
        :param x: np.ndarray or torch.Tensor (n_samples, max_sentence_length)
        :return prediction for class (n_samples, out_num_classes)
        """

        # type check
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).to(device).int()
        # reshape when there is only one sample
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        
        x_c = self.embed_enc(x.int())
        x_c, _ = self.rnn(x_c)
        x_c = self.flat(x_c)

        x = F.relu(self.fc1(x_c))
        x = F.relu(self.fc2(x))
        if self.out_num_classes == 2:
            x = F.sigmoid(self.fc3(x))
        else:
            x = F.softmax(self.fc3(x))

        return x

def create_emb_layer(weights_matrix, freeze = True):
    num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(weights_matrix), freeze = freeze, sparse = True)
    #emb_layer.load_state_dict({'weight': weights_matrix})

    return emb_layer, num_embeddings, embedding_dim
 
class Transformer_Baseline_Model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 input_shape: tuple,
                 out_num_classes: int,
                 pretrained: str = "None",
                 freeze_emb: bool = True,
                 embed_dim: int = 5):
        """
        Transformer Encoder Class for text classification.
        :param vocab_size specifies the size of the vocabulary to be used in word embeddings.
        :param input_shape specifies the shape of the features (n_samples, max_sentence_length?). TODO check
        :param out_num_classes specifies the number of classes to be predicted.
        :param pretrained specifies the type of word-embeddings,
        "None": Use no pretrained word embeddings.
        "glove": Use GloVe embeddings.
        Throws an error when an other input is given.
        :param freeze_emb specifies whether to freeze the embeddings when it is pretrained. 
        Will not affect the embedding if it is not pretrained.
        :param embed_dim specifies the embedding dimension for the categorical part of the input, 
        when it is not pretrained.
        """
        super(Transformer_Baseline_Model, self).__init__()

        self.embed_dim = embed_dim
        self.embedding = None
        if pretrained == "None":
            self.embedding = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        elif pretrained == "glove":
            """
            glove = GloVe(name='6B', dim=300)
            weights_matrix = np.zeros((vocab_size, 300))
            words_found = 0
            for i, word in enumerate(vocab.get_itos()):
                try: 
                    weights_matrix[i] = glove.get_vecs_by_tokens(word)
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = glove.get_vecs_by_tokens("<unk>")

            self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, freeze_emb)
            """
            raise NotImplementedError()
        else:
            raise ValueError("The argument pretrained must be either 'None' or 'glove'.")
        # self.embed_enc = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        
        # print(num_embeddings, embedding_dim)
        trans_enc_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = 5 ,batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_enc_layer, num_layers=16)
        self.fc1 = nn.Linear(18600, out_num_classes) # TODO find out dimensions
        self.flatten = nn.Flatten()

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Function for model architecture.
        :param x: np.ndarray or torch.Tensor (n_samples, max_sentence_length)
        :return prediction for class (n_samples, out_num_classes)
        """

        # type check
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).to(device).int()
        # reshape when there is only one sample
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        
        x = self.embedding(x)
        #print(x.shape)
        x = self.transformer(x)
        #print(x.shape)
        #print(cn, hn.shape, 2)
        x = self.flatten(x)
        #print(x.shape)
        x = F.softmax(self.fc1(x))
        return x

    def embout(self, x):
        return self.embedding(x)

    # freeze the layers until given index
    def freeze_layers(self, index):
        cnt = 0
        for p in self.parameters():
            p.requires_grad = False
            cnt += 1
            if(cnt == index):
              break
