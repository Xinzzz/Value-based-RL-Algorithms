import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        conv_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_dim[1])))
        self.fc1 = nn.Linear(conv_out * conv_out * 64, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

