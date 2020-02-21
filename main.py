import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))

from algos.DQN import *
from common.plot import *
from common.wrappers import *
from common.logger import *
from common.hyperparameters import *

env_run_name = 'pong'
params = HYPERPARAMS[env_run_name]

seed = params['seed']
env = gym.make(params['env_name'])
env = wrap_env(env)
seed_torch(seed)
env.seed(seed)

agent = DQNAgent(env, params)
agent.train()
# agent.load_saved_model()
# agent.dqn.eval()
# agent.test()
load_data(agent.name)
