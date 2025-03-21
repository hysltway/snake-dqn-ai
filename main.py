from Agent import Agent
from env import Snake
import pygame
import numpy as np


def testAgent(test_env, agent, episode):
    ep_reward = 0
    o = test_env.reset()
    for _ in range(650):
        if episode % 100 == 0:
            test_env.render()
        for event in pygame.event.get():  # 不加这句render要卡，不清楚原因
            pass
        a_int = agent.select_action(o)
        o2, reward, done, _ = test_env.step(a_int)
        ep_reward += reward
        if done:
            break
        o = o2
    return ep_reward


if __name__ == "__main__":
    env = Snake()
    test_env = Snake()
    act_dim = 4
    state_dim = 10
    agent = Agent(act_dim, state_dim)
    agent.state = env.reset()
    MAX_EPISODE = 2000
    maxReward = -np.inf

    for episode in range(MAX_EPISODE):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)

            state = next_state

            agent.train_step()

        if episode % agent.target_update == 0:
            agent.update_target_network()

        ep_reward = testAgent(test_env, agent, episode)
        print('Episode:', episode, 'Reward:%f' % ep_reward, 'Max Reward:%f' % maxReward)

        if episode > MAX_EPISODE / 100 and ep_reward > maxReward:
            maxReward = ep_reward
            print('当前最大奖赏值更新！')

    pygame.quit()
