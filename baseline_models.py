import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class RNN_Baseline_Model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 input_shape: tuple,
                 out_num_classes: int,
                 embed_dim: int = 5,
                 rnn_type: str = "gru",
                 rnn_hidden_size: int = 2,
                 rnn_hidden_out: int = 2,
                 rnn_bidirectional: bool = True,
                 units: int = 50):
        """
        RNN Class for text classification.
        :param vocab_size specifies the size of the vocabulary to be used in word embeddings.
        :param input_shape specifies the shape of the features (n_samples, max_sentence_length?). TODO check
        :param embed_dim specifies the embedding dimension for the categorical part of the input.
        :param rnn_type specifies the type of the recurrent layer for word embeddings.
        MUST be "gru" or "lstm".
        :param rnn_hidden_size specifies the number of stacked recurrent layers.
        :param rnn_hidden_out specifies number of features in the hidden state h of recurrent layer.
        :param rnn_bidirectional specifies whether the recurrent layers be bidirectional.
        :param units specifies the number of neurons in the hidden layers.
        """
        super(RNN_Baseline_Model, self).__init__()
        self.embed_dim = embed_dim

        self.embed_enc = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_out, rnn_hidden_size,
                          bidirectional=rnn_bidirectional, batch_first=True)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_out, rnn_hidden_size,
                               bidirectional=rnn_bidirectional, batch_first=True)
        elif rnn_type != "gru":
            raise ValueError("The argument rnn_type must be 'gru' or 'lstm'!")

        rnn_out_dim = rnn_hidden_out * input_shape[1]
        if rnn_bidirectional:
            rnn_out_dim *= 2

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(rnn_out_dim, units)
        self.fc2 = nn.Linear(units, units)
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
        
        x_c = self.embed_enc(x)
        x_c, _ = self.rnn(x_c)
        x_c = self.flat(x_c)

        x = torch.cat((x_c, x_num), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        # TODO figure out the embedding transformations
        return x


 
class Transformer_Baseline_Model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 input_shape: tuple,
                 out_num_classes: int,
                 embed_dim: int = 5):
        """
        Transformer Encoder Class for text classification.
        :param vocab_size specifies the size of the vocabulary to be used in word embeddings.
        :param input_shape specifies the shape of the features (n_samples, max_sentence_length?). TODO check
        :param embed_dim specifies the embedding dimension for the categorical part of the input.
        """
        super(Transformer_Baseline_Model, self).__init__()

        self.embed_dim = embed_dim
        self.embed_enc = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, freeze_emb)
        print(num_embeddings, embedding_dim)
        trans_enc_layer = nn.TransformerEncoderLayer(d_model = embedding_dim, nhead = 5 ,batch_first=True)
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
