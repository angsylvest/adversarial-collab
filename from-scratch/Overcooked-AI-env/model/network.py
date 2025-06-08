import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.model import fanin_init


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, activation=F.relu,
                 constrain_out=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.activation = activation

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        if constrain_out:
            self.fc3.weight.data.uniform_(-0.003, 0.003)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # X = self.norm1(X)
        h1 = self.activation(self.fc1(X))
        h2 = self.activation(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, lstm_hidden_dim=128, activation=F.relu, constrain_out=False):
        super(LSTMEncoder, self).__init__()

        self.activation = activation
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        if constrain_out:
            self.fc3.weight.data.uniform_(-0.003, 0.003)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    # TODO: need to init hidden as well s
    def init_hidden(self):
        # For LSTM: hidden_state = (h_0, c_0)
        batch_size = 1
        hidden_dim = 128
        return (torch.zeros(1, batch_size, hidden_dim),
                torch.zeros(1, batch_size, hidden_dim))

    def forward(self, X, hidden_state=None):
        """
        Inputs:
            X (torch.Tensor): (batch_size, seq_len, input_dim)
            hidden_state (tuple): (h_0, c_0) for LSTM, optional
        Outputs:
            out (torch.Tensor): (batch_size, out_dim)
        """
        
        # if passed per observation 
        if X.dim() == 2:
            X = X.unsqueeze(1)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(X, hidden_state)  # lstm_out: (batch_size, seq_len, lstm_hidden_dim)

        # Take only the last time step’s output
        last_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)

        # MLP head
        h1 = self.activation(self.fc1(last_out))
        h2 = self.activation(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out, hidden
    

class LSTMEncoderKL(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, lstm_hidden_dim=128, activation=F.relu):
        super(LSTMEncoderKL, self).__init__()

        self.activation = activation
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

        self.fc_mu = nn.Linear(lstm_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(lstm_hidden_dim, latent_dim)

        self.hidden_dim = hidden_dim

    def forward(self, X, hidden_state=None):
        if X.dim() == 2:
            X = X.unsqueeze(1)

        lstm_out, hidden = self.lstm(X, hidden_state)
        last_out = lstm_out[:, -1, :]  # (batch, lstm_hidden_dim)

        mu = self.fc_mu(last_out)
        logvar = self.fc_logvar(last_out)
        return mu, logvar, hidden

    
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, lstm_hidden_dim=128, activation=F.relu, constrain_out=False):
        super(LSTMNetwork, self).__init__()

        self.activation = activation
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        if constrain_out:
            self.fc3.weight.data.uniform_(-0.003, 0.003)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    # TODO: need to init hidden as well s
    def init_hidden(self):
        # For LSTM: hidden_state = (h_0, c_0)
        batch_size = 1
        hidden_dim = 64
        return (torch.zeros(1, batch_size, hidden_dim),
                torch.zeros(1, batch_size, hidden_dim))

    def forward(self, X, hidden_state=None):
        """
        Inputs:
            X (torch.Tensor): (batch_size, seq_len, input_dim)
            hidden_state (tuple): (h_0, c_0) for LSTM, optional
        Outputs:
            out (torch.Tensor): (batch_size, out_dim)
        """
        
        # if passed per observation 
        if X.dim() == 2:
            X = X.unsqueeze(1)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(X, hidden_state)  # lstm_out: (batch_size, seq_len, lstm_hidden_dim)

        # Take only the last time step’s output
        last_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)

        # MLP head
        h1 = self.activation(self.fc1(last_out))
        h2 = self.activation(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out, hidden
    
class BCQActorNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, activation=F.relu,
                 constrain_out=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BCQActorNetwork, self).__init__()

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.activation = activation

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        if constrain_out:
            self.fc3.weight.data.uniform_(-0.003, 0.003)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, state, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # X = self.norm1(X)
        X = torch.cat([state, action], 1)
        h1 = self.activation(self.fc1(X))
        h2 = self.activation(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

      
class VAENetwork(nn.Module):
    """
    VAE network
    """
    def __init__(self, state_dim, action_dim, latent_dim):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(VAENetwork, self).__init__()

        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.latent_dim = latent_dim

    def forward(self, state, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).clamp(-0.5, 0.5)
            z = z.to(state.device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))

        return torch.tanh(self.d3(a))

      
class BCNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, activation=F.relu):
        super(BCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.activation = activation

    def forward(self, X):
        h1 = self.activation(self.fc1(X))
        out = self.fc2(h1)
        return out

