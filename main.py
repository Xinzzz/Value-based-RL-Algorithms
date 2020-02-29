import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))

from algos.DQN import *
from common.plot import *
from common.wrappers import *
from common.logger import *
from common.hyperparameters import *


def ready_env(env_id):
    env_run_name = env_id
    global params
    params = HYPERPARAMS[env_run_name]
    seed = params['seed']
    global env
    env =  gym.make(params['env_name'])
    env = wrap_env(env)
    seed_torch(seed)
    env.seed(seed)

ready_env('cartpole')
agent_1 = DQNAgent(env, params, 64,64)
agent_2 = DQNAgent(env, params, 64,128)
agent_3 = DQNAgent(env, params, 128,128)
agent_4 = DQNAgent(env, params, 128,256)
agent_5 = DQNAgent(env, params, 256,256)
agent_6 = DQNAgent(env, params, 256,512)

agent_7 = DQNAgent(env, params, 64,64, noisy=True)
agent_8 = DQNAgent(env, params, 64,128, noisy=True)
agent_9 = DQNAgent(env, params, 128,128, noisy=True)
agent_10 = DQNAgent(env, params, 128,256, noisy=True)
agent_11 = DQNAgent(env, params, 256,256, noisy=True)
agent_12 = DQNAgent(env, params, 256,512, noisy=True)

agent_13 = DQNAgent(env, params, 64,64, dueling=True)
agent_14 = DQNAgent(env, params, 64,128, dueling=True)
agent_15 = DQNAgent(env, params, 128,128, dueling=True)
agent_16 = DQNAgent(env, params, 128,256, dueling=True)
agent_17 = DQNAgent(env, params, 256,256, dueling=True)
agent_18 = DQNAgent(env, params, 256,512, dueling=True)

agent_19 = DQNAgent(env, params, 64,64, noisy=True, dueling=True)
agent_20 = DQNAgent(env, params, 64,128, noisy=True, dueling=True)
agent_21 = DQNAgent(env, params, 128,128, noisy=True, dueling=True)
agent_22 = DQNAgent(env, params, 128,256, noisy=True, dueling=True)
agent_23 = DQNAgent(env, params, 256,256, noisy=True, dueling=True)
agent_24 = DQNAgent(env, params, 256,512, noisy=True, dueling=True)

agent_25 = DQNAgent(env, params, 64,64, noisy=True, dueling=True, mod=True)
agent_26 = DQNAgent(env, params, 64,128, noisy=True, dueling=True, mod=True)
agent_27 = DQNAgent(env, params, 128,128, noisy=True, dueling=True, mod=True)
agent_28 = DQNAgent(env, params, 128,256, noisy=True, dueling=True, mod=True)
agent_29 = DQNAgent(env, params, 256,256, noisy=True, dueling=True, mod=True)
agent_30 = DQNAgent(env, params, 256,512, noisy=True, dueling=True, mod=True)

agent_31 = DQNAgent(env, params, 64,64, noisy=True, mod=True)
agent_32 = DQNAgent(env, params, 64,128, noisy=True, mod=True)
agent_33 = DQNAgent(env, params, 128,128, noisy=True, mod=True)
agent_34 = DQNAgent(env, params, 128,256, noisy=True, mod=True)
agent_35 = DQNAgent(env, params, 256,256, noisy=True, mod=True)
agent_36 = DQNAgent(env, params, 256,512, noisy=True, mod=True)



training = False
if training:
    agent_1.train()
    # agent_2.train()
    # agent_3.train()
    # agent_4.train()
    # agent_5.train()
    # agent_6.train()

    # agent_7.train()
    # agent_8.train()
    # agent_9.train()
    # agent_10.train()
    # agent_11.train()
    # agent_12.train()

    # agent_13.train()
    # agent_14.train()
    # agent_15.train()
    # agent_16.train()
    # agent_17.train()
    # agent_18.train()

    # agent_19.train()
    # agent_20.train()
    # agent_21.train()
    # agent_22.train()
    # agent_23.train()
    # agent_24.train()
    # agent_25.train()
    # agent_26.train()
    # agent_27.train()
    # agent_28.train()
    # agent_29.train()
    # agent_30.train()
    # agent_31.train()
    # agent_32.train()
    # agent_33.train()
    # agent_34.train()
    # agent_35.train()
    # agent_36.train()
    


loading = True
if loading:
    data_to_plot = []
    
    # data_to_plot.append((load_data(agent_11.env_name + agent_11.model_name),'darkgrey', 'DQN Noisy'))
    # data_to_plot.append((load_data(agent_27.env_name + agent_24.model_name),'orange', 'Duling Noisy'))
    data_to_plot.append((load_data(agent_25.env_name + agent_25.model_name),'red','1'))
    data_to_plot.append((load_data(agent_25.env_name + agent_28.model_name),'green', '2'))
    data_to_plot.append((load_data(agent_25.env_name + agent_19.model_name),'salmon', '3'))
    # data_to_plot.append((load_data(agent_22.env_name + agent_29.model_name),'gold', 'DQN DN'))
    # data_to_plot.append((load_data(agent_33.env_name + agent_30.model_name),'salmon', 'DQN DN'))
    plot(data_to_plot)