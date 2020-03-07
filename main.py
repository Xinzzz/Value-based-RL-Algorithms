import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))

from algos.DQN import *
from common.plot import *
from common.wrappers import *
from common.logger import *
from common.hyperparameters import *

def ready_env():
    env_id='freeway'
    env_run_name = env_id
    global params
    params = HYPERPARAMS[env_run_name]
    seed = params['seed']
    global env
    env =  gym.make(params['env_name'])
    env = wrap_env(env)
    seed_torch(seed)
    env.seed(seed)
    return env,params

env, params = ready_env()
agent_1 = DQNAgent(env, params, 32,64, PER=True, RND_use=True, double=True)
agent_2 = DQNAgent(env, params, 32,64, PER=True, double=True)

# training = False
# if training:
#     agent_1.train()
#     agent_2.train()
#     agent_3.train()
#     agent_4.train()
#     agent_5.train()
#     agent_6.train()


loading = True
if loading:
    data_to_plot = []
    
    #data_to_plot.append((load_data(agent_1.env_name + agent_1.model_name),'black', 'DQN Noisy'))
    #data_to_plot.append((load_data(agent_2.env_name + agent_2.model_name),'orange', 'Duling Noisy'))
    data_to_plot.append((load_data(agent_1.env_name + agent_1.model_name),'orangered','Dueling-DN')) #29
    data_to_plot.append((load_data(agent_2.env_name + agent_2.model_name),'salmon', 'DQN-DN')) #31 33
    #data_to_plot.append((load_data(agent_2.env_name + agent_5.model_name),'gold', 'Dueling Noisy')) # 20
    #data_to_plot.append((load_data(agent_6.env_name + agent_6.model_name),'darkgrey', 'Noisy DQN')) #9
    #data_to_plot.append((load_data(agent_2.env_name + agent_1.model_name),'blue', 'DQN DN'))

    plot(data_to_plot)

