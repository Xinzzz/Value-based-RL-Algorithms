import os
from typing import Dict, List, Tuple

import gym
import numpy as np
import torch


from tqdm import tqdm

def train(Agent, env, num_frames: int, plotting_interval: int = 200):
    """Train the agent."""
    Agent.is_test = False
    
    state = Agent.env.reset()
    update_cnt = 0
    epsilons = []
    losses = []
    scores = []
    score = 0

    for frame_idx in tqdm(range(1, num_frames + 1)):
        action = Agent.select_action(state)
        next_state, reward, done = Agent.step(action)

        state = next_state
        score += reward

        # if episode ends
        if done:
            state = env.reset()
            scores.append(score)
            score = 0

        # if training is ready
        if len(Agent.memory) >= Agent.batch_size:
            loss = Agent.update_model()
            losses.append(loss)
            update_cnt += 1
            
            # linearly decrease epsilon
            Agent.epsilon = max(
                Agent.min_epsilon, Agent.epsilon - (
                    Agent.max_epsilon - Agent.min_epsilon
                ) * Agent.epsilon_decay
            )
            epsilons.append(Agent.epsilon)
            
            # if hard update is needed
            if update_cnt % Agent.target_update == 0:
                Agent._target_hard_update()
            
    Agent.env.close()
    return frame_idx, scores, losses, epsilons

def test(Agent) -> None:
    """Test the agent."""
    Agent.is_test = True
    
    state = Agent.env.reset()
    done = False
    score = 0
    
    while not done:
        Agent.env.render()
        action = Agent.select_action(state)
        next_state, reward, done = Agent.step(action)

        state = next_state
        score += reward
    
    print("score: ", score)
    Agent.env.close()

