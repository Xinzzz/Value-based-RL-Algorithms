import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--seed', dest='seed', type=int, default=6)
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--atari_env', dest='atari_env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--episodes', dest='episodes', type=int, default=500)
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=10000)
    parser.add_argument('--warmup_memory_size', dest='warmup_memory_size', type=int, default=1000)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99)
    parser.add_argument('--eps_start', dest='eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', dest='eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', dest='eps_decay', type=int, default=3000)
    parser.add_argument('--target_update_freq', dest='target_update_freq', type=int, default=1000)

    return parser.parse_args()
