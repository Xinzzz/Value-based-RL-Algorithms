import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class nn_DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_dim_2, noisy=False):
        super(nn_DQN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noisy = noisy
        self.feature = nn.Linear(input_dim[0], hidden_dim)
        
        if not self.noisy:
            self.hidden = nn.Linear(hidden_dim, hidden_dim_2)
            self.out = nn.Linear(hidden_dim_2, output_dim)
        elif self.noisy:
            self.hidden = NoisyLinear(hidden_dim, hidden_dim_2)
            self.out = NoisyLinear(hidden_dim_2, output_dim)

    def forward(self, x):
        x = F.relu(self.feature(x))
        x = F.relu(self.hidden(x))
        x = self.out(x)

        return x

    def reset_noise(self):
        self.hidden.reset_noise()
        self.out.reset_noise()

class nn_Dueling(nn.Module):
    def __init__(self, input_dim, output_dim,  hidden_dim, hidden_dim_2, noisy=False):
        super(nn_Dueling, self).__init__()

        # common feature layer
        self.feature = nn.Sequential(
            nn.Linear(input_dim[0], hidden_dim),
            nn.ReLU(),
        )

        # adv layer
        if not noisy:
            self.adv_hidden_layer = nn.Linear(hidden_dim, hidden_dim_2)
            self.adv_layer = nn.Linear(hidden_dim_2, output_dim)
        elif noisy:
            self.adv_hidden_layer = NoisyLinear(hidden_dim, hidden_dim_2)
            self.adv_layer = NoisyLinear(hidden_dim_2, output_dim)

        # val layer
        if not noisy:
            self.val_hidden_layer = nn.Linear(hidden_dim, hidden_dim_2)
            self.val_layer = nn.Linear(hidden_dim_2, 1)
        elif noisy:
            self.val_hidden_layer = NoisyLinear(hidden_dim, hidden_dim_2)
            self.val_layer = NoisyLinear(hidden_dim_2, 1)

    def forward(self, x):
        feature = self.feature(x)

        val_hidden = F.relu(self.val_hidden_layer(feature))
        val = self.val_layer(val_hidden)
        
        adv_hidden = F.relu(self.adv_hidden_layer(feature))
        adv = self.adv_layer(adv_hidden)

        q = val + adv - adv.mean(dim=-1, keepdim=True)

        return q

    def reset_noise(self):
        self.adv_layer.reset_noise()
        self.adv_hidden_layer.reset_noise()
        self.val_layer.reset_noise()
        self.val_hidden_layer.reset_noise()

class cnn_DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hid1_dim, hid2_dim): #32 64
        super(cnn_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], hid1_dim, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(hid1_dim, hid2_dim, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(hid2_dim, hid2_dim, kernel_size=3, stride=1),
            nn.ReLU()
        )
 
        conv_out_size = self._get_conv_out(input_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)

class cnn_Dueling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cnn_Dueling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
 
        conv_out_size = self._get_conv_out(input_dim)
        
        self.adv_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.val_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        adv = self.adv_fc(conv_out)
        val = self.val_fc(conv_out)

        q = val + adv - adv.mean(dim=-1, keepdim=True)

        return q



class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        """Make new noise."""
        self.epsilon_in = self.scale_noise(self.in_features) 
        self.epsilon_out = self.scale_noise(self.out_features) 
        self.weight_epsilon.copy_(self.epsilon_out.ger(self.epsilon_in))
        self.bias_epsilon.copy_(self.epsilon_out)


        self.reset_parameters()
        self.reset_noise()


    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        self.epsilon_in = self.scale_noise(self.in_features)
        self.epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(self.epsilon_out.ger(self.epsilon_in))
        self.bias_epsilon.copy_(self.epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())