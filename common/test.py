import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('LunarLander-v2').unwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    env.reset()
    for _ in range(20):
        env.render()
        env.step(env.action_space.sample())
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # # Cart is in the lower half, so strip off the top and bottom of the screen
    # _, screen_height, screen_width = screen.shape
    # screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # view_width = int(screen_width * 0.6)
    # cart_location = get_cart_location(screen_width)
    # if cart_location < view_width // 2:
    #     slice_range = slice(view_width)
    # elif cart_location > (screen_width - view_width // 2):
    #     slice_range = slice(-view_width, None)
    # else:
    #     slice_range = slice(cart_location - view_width // 2,
    #                         cart_location + view_width // 2)
    # # Strip off the edges, so that we have a square image centered on a cart
    # screen = screen[:, :, slice_range]
    # # Convert to float, rescale, convert to torch tensor
    # # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return screen.unsqueeze(0).to(device)


env.reset()
plt.tight_layout()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.axis('off')
plt.savefig('test.svg', bbox_inches='tight', facecolor='none', edgecolor='none', dpi=300)
plt.show()