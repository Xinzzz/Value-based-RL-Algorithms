import argparse

def get_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('-seed', type=int, default=777, help='the random seeds')
    parse.add_argument('-num_frames', type=int, default=1000, help='frams to train nn')
    parse.add_argument('-memory_size', type=int, default=10000, help='length of replay memory to store transitions')
    parse.add_argument('-batch_size', type=int, default=32, help='batch size for sampling')
    parse.add_argument('-target_update', type=int, default=100, help='period for target model hard update')
    parse.add_argument('-epsilon_decay', type=float, default=1 / 2000, help='step size to decrease epsilon')
    parse.add_argument('-max_epsilon', type=float, default=1.0, help='max value of epsilon')
    parse.add_argument('-min_epsilon', type=float, default=0.1, help='min value of epsilon')
    parse.add_argument('-gamma', type=float, default=0.99, help='discount factor')
    # PER
    parse.add_argument('-alpha', type=float, default=0.2)
    parse.add_argument('-beta', type=float, default=0.6, help='determines how much importance sampling is used')
    parse.add_argument('-prior_eps', type=float, default=1e-6, help='guarantees every transition can be sampled')

    args = parse.parse_args()

    return args