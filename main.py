import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))

from algos.DQN import *
from common.plot import *
from common.wrappers import *
from common.logger import *
from common.hyperparameters import *

env_run_name = 'cartpole'
params = HYPERPARAMS[env_run_name]

seed = params['seed']
env = gym.make(params['env_name'])
env = wrap_env(env)
seed_torch(seed)
env.seed(seed)

agent_1 = DQNAgent(env, params, dueling=True, noisy=True)
agent_2 = DQNAgent(env, params, dueling=True, noisy=True, mod=True)
agent_3 = DQNAgent(env, params, noisy=True, mod=True)
agent_4 = DQNAgent(env, params)
agent_5 = DQNAgent(env, params, noisy=True)

training = False
if training:
    #agent_1.train()
    #agent_2.train()
    # agent_3.train()
    #agent_4.train()
    agent_5.train()
   

loading = True
if loading:
    data_to_plot = []

    #data_to_plot.append((load_data(agent_1.env_name + agent_1.model_name),'darkgrey'))
    #data_to_plot.append((load_data(agent_2.env_name + agent_2.model_name),'deepskyblue'))
    # data_to_plot.append((load_data(agent_3.env_name + agent_3.model_name),'salmon'))
    #data_to_plot.append((load_data(agent_4.env_name + agent_4.model_name),'green'))
    data_to_plot.append((load_data(agent_5.env_name + agent_5.model_name),'salmon'))
    plot(data_to_plot)