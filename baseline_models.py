import torch
import torch.nn as nn
import torch.nn.functional as F

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
        The argument "vocab_size" specifies the size of the vocabulary to be used in word embeddings.
        The argument "input_shape" specifies the shape of the features (n_samples, max_sentence_length?). TODO check
        The argument "embed_dim" specifies the embedding dimension for the categorical part of the input.
        The argument "rnn_type" specifies the type of the recurrent layer for word embeddings.
        MUST be "gru" or "lstm".
        The argument "rnn_hidden_size" specifies the number of stacked recurrent layers.
        The argument "rnn_hidden_out" specifies number of features in the hidden state h of recurrent layer.
        The argument "rnn_bidirectional" specifies whether the recurrent layers be bidirectional.
        The argument "units" specifies the number of neurons in the hidden layers.
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
        # print(f"After embedding x_cat shape: {x_c.shape}")
        x_c, _ = self.rnn(x_c)
        x_c = self.flat(x_c)

        x = torch.cat((x_c, x_num), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # TODO figure out the embedding transformationskkkk
        return x