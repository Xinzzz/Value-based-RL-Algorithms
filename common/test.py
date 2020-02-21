import retro
import gym
def main():
    env = gym.make('Copy-v0')
    print(env.observation_space)
    print(env.action_space.n)
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        #env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
