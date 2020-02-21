import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class nn_DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(nn_DQN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature = nn.Linear(input_dim[0], 128)
        self.hidden = nn.Linear(128, 256)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.feature(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.out(x))

        return x

class cnn_DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cnn_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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

