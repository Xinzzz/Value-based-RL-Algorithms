'''
target_update_freq: update model step interval
'''

HYPERPARAMS = {
    'pong': {
        'env_name': "PongNoFrameskip-v4",
        'stop_reward': 18.0,
        'run_name': 'pong',
        'replay_size': 10000,
        'replay_warmup': 4000,
        'target_update_freq': 1000,
        'epsilon_decay': 50000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64,
        'episode': 200,
        'seed': 666
    },
    'monte': {
        'env_name': "MontezumaRevengeNoFrameskip-v4",
        'stop_reward': 18.0,
        'run_name': 'monte',
        'replay_size': 50000,
        'replay_warmup': 4000,
        'target_update_freq': 1000,
        'epsilon_decay': 50000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64,
        'episode': 200,
        'seed': 666
    },
    'lunarlander': {
        'env_name': "LunarLander-v2",
        'stop_reward': 200.0,
        'run_name': 'lunarlander',
        'replay_size': 5000,
        'replay_warmup': 256,
        'target_update_freq': 400,
        'epsilon_decay': 4000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 32,
        'episode': 200,
        'seed': 777
    },
    'acrobot': {
        'env_name': "Acrobot-v1",
        'stop_reward': 200.0,
        'run_name': 'acrobot',
        'replay_size': 2000,
        'replay_warmup': 32,
        'target_update_freq': 100,
        'epsilon_decay': 5000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 32,
        'episode': 300,
        'seed': 777
    },

    'cartpole':{
        'env_name': "CartPole-v0",
        'stop_reward': 200.0,
        'run_name': 'cartpole',
        'replay_size': 2000,
        'replay_warmup': 32,
        'target_update_freq': 100,
        'epsilon_decay': 2000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 32,
        'episode': 300,
        'seed': 777
    },

    'mountaincar':{
        'env_name': "MountainCar-v0",
        'stop_reward': 200.0,
        'run_name': 'mountaincar',
        'replay_size': 2000, 
        'replay_warmup': 32,
        'target_update_freq': 200,
        'epsilon_decay': 70000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'batch_size': 32,
        'episode': 10,
        'seed': 777
    },
}