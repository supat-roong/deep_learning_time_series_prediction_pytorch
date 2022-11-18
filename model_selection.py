import torch
import torch.nn as nn
from torch.autograd import Variable


class ModelSelector:
    def __init__(self, model_config_map):
        """
        Model selector class object to select model from specify config

        Args:
            model_config_map (dict)
        """
        self.model = model_config_map["model"]
        self.input_size = model_config_map["input_size"]
        self.output_size = model_config_map["output_size"]
        self.hidden_size = model_config_map["hidden_size"]
        self.num_layers = model_config_map["num_layers"]

    def create_model(self):
        """
        Create and return deep learning model from specify parameters

        Returns:
            model: deep learning model class object
        """
        if self.model == "rnn":
            return RNN(
                self.input_size, self.output_size, self.hidden_size, self.num_layers
            )
        elif self.model == "lstm":
            return LSTM(
                self.input_size, self.output_size, self.hidden_size, self.num_layers
            )
        elif self.model == "gru":
            return GRU(
                self.input_size, self.output_size, self.hidden_size, self.num_layers
            )


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        # Defining some parameters
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Defining the layers
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out


class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(GRU, self).__init__()

        # Defining some parameters
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Defining the layers
        # GRU Layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        ula, h_out = self.gru(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = h_out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        ula, h_out = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = h_out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden
