import torch.nn as nn


class GuitarLSTM(nn.Module):
    def __init__(self, conv1d_filters, conv1d_strides, input_size, hidden_units):
        super(GuitarLSTM, self).__init__()
        self.kernel_size = 12
        self.conv1d_layer1 = nn.Conv1d(in_channels=input_size, out_channels=conv1d_filters, kernel_size=self.kernel_size, stride=conv1d_strides, padding=6)

        self.conv1d_layer2 = nn.Conv1d(in_channels=conv1d_filters, out_channels=conv1d_filters, kernel_size=self.kernel_size, stride=conv1d_strides, padding=6)

        self.lstm_layer = nn.LSTM(input_size=conv1d_filters, hidden_size=hidden_units, batch_first=True)
        self.dense_layer = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.conv1d_layer1(x)
        x = self.conv1d_layer2(x)
        x, _ = self.lstm_layer(x.transpose(1, 2))
        x = self.dense_layer(x[:, -1, :])
        return x



# class GuitarLSTM(nn.Module):
#     def __init__(self, conv1d_filters, conv1d_strides, input_size, hidden_units):
#         super(GuitarLSTM, self).__init__()
#         self.kernel_size = 12
#         self.conv1d_layer1 = nn.Conv1d(in_channels=input_size, out_channels=conv1d_filters, kernel_size=self.kernel_size, stride=conv1d_strides, padding=6)
#
#         self.conv1d_layer2 = nn.Conv1d(in_channels=conv1d_filters, out_channels=conv1d_filters, kernel_size=self.kernel_size, stride=conv1d_strides, padding=6)
#
#         self.lstm_layer = nn.LSTM(input_size=conv1d_filters, hidden_size=hidden_units, batch_first=True)
#         self.dense_layer = nn.Linear(hidden_units, 1)
#
#     def forward(self, x):
#         x = self.conv1d_layer1(x)
#         x = self.conv1d_layer2(x)
#         x, _ = self.lstm_layer(x.transpose(1, 2))
#         x = self.dense_layer(x[:, -1, :])
#         return x
#


