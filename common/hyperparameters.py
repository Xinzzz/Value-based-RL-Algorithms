'''
target_update_freq: update model step interval
'''

HYPERPARAMS = {
    'pong': {
        'env_name': "BreakoutNoFrameskip-v4",
        'stop_reward': 18.0,
        'run_name': 'pong',
        'replay_size': 10000,
        'replay_warmup': 2000,
        'target_update_freq': 1000,
        'epsilon_decay': 10000,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64,
        'episode': 5000,
        'seed': 666
    },

    'cartpole':{
        'env_name': "CartPole-v0",
        'stop_reward': 200.0,
        'run_name': 'cartpole',
        'replay_size': 1000,
        'replay_warmup': 32,
        'target_update_freq': 100,
        'epsilon_decay': 3000,
        'epsilon_start': 0.9,
        'epsilon_final': 0.02,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 32,
        'episode': 200,
        'seed': 6
    },
}