#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: daisuke
"""
import sys
sys.path.append("/Users/daisuke/.pyenv/versions/anaconda3-2.4.0/lib/python3.5/site-packages")
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

def Grad_estimate(Z, q):
    Znorm_inv = np.linalg.inv(np.dot(Z.T, Z))
    w = np.dot(np.dot(Znorm_inv, Z.T), q)
    return w

def Pendulum_run(L, M, param):
    env = gym.make("Pendulum-v0")

    #mu = np.array([[random.uniform(-2, 2)] for _ in range(3)]) # 平均 μ
    mu = np.zeros([2,1])
    sigma = random.uniform(-2, 2)
    Z_mu = np.zeros([M,2])
    Z_sigma = np.zeros([M,1])
    q = np.zeros([M,1])
    w_mu = np.zeros([M,2])
    w_sigma = np.zeros([M,1])

    print("init sigma:",sigma)

    rewards_list = []

    for l in range(L):
        state1_value = 0

        for i_episode in range(M):
            observation = env.reset()
            state = np.array([observation[0:2]]).T
            step = 0
            rewards = 0

            while True:
                step += 1

                #action = random.randn() * sigma + np.dot(mu.T, state)[0]
                action = random.gauss(0, max(-sigma*0.1,sigma*0.1)) + np.dot(mu.T, state)[0]
                action = max([-2.0], action)
                action = min([2.0], action)


                observation, reward, done, _ = env.step(action)
                state = np.array([observation[0:2]]).T

                rewards += reward
                state1_value += param["gamma"]**(step - 1) * reward

                #print(Z_mu[i_episode], Z_sigma[i_episode])
                Z_mu[i_episode] += param["gamma"]**(step - 1) * ((action - np.dot(mu.T, state)) * state / (sigma**2)).T[0]
                Z_sigma[i_episode] += param["gamma"]**(step - 1) * (((action - np.dot(mu.T, state))**2 - sigma**2) / (sigma**3))[0]

                q[i_episode] += param["gamma"]**(step - 1) * reward

                if i_episode == M-1:
                    rewards_list.append(rewards)
                    env.render()

                if done:
                    #print("Episode {0} finished after {1} timesteps, rewards {2}".format(i_episode+1, step, rewards))
                    break

        q -= state1_value / M

        w_mu = Grad_estimate(Z_mu, q)
        w_sigma = Grad_estimate(Z_sigma, q)[0][0]

        print("-"*20)
        print("Repitation {0}, rewards {1}".format(l, state1_value))

        print("mu_grad ",w_mu)
        print("sigma_grad ",w_sigma)

        mu += param["alpha"] * w_mu
        sigma += param["alpha"] * w_sigma

        print("mu ",mu)
        print("sigma ",sigma)
                      
    return rewards_list

if __name__ == '__main__':
    num_episode = 100
    repetition = 2000

    param = { "gamma":0.8, "alpha":0.05}
    rewards_list = Pendulum_run(repetition, num_episode, param)

    plt.clf()
    plt.plot(range(len(rewards_list)), rewards_list)
    plt.savefig('pendulum_reward.png')
