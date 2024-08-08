
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvLSTM(nn.Module):
    def __init__(self, class_amount: int=0, embedding_size: int=64, hidden_size: int=10, layers: int=1, dropout_chance: float=0.5, kernel_size: int=3, cnn_out_dim: int=64):
        super(ConvLSTM, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance

        self.kernel_size = kernel_size
        self.cnn_out_dim = cnn_out_dim

        self.embedder = nn.Embedding(29, self.embedding_size)

        self.conv1 = nn.Sequential(nn.Conv1d(self.embedding_size, self.cnn_out_dim, kernel_size=self.kernel_size),
                                   nn.ReLU())
        
        self.lstm = nn.LSTM(input_size=self.cnn_out_dim, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True)
        
        self.dropout = nn.Dropout2d(p=self.dropout_chance)
        self.linear1 = nn.Linear(self.hidden_size, class_amount)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedder(x.type(torch.LongTensor).to(device=device))
        x = x.squeeze(2).transpose(1, 2)
        
        x = self.conv1(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = x[:, -1]

        x = self.dropout(x)

        x = self.linear1(x)
        x = self.logSoftmax(x)

        return x



