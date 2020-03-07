import gym
from gym.utils.play import play
env = gym.make("FreewayNoFrameskip-v4")
play(env, zoom=4)