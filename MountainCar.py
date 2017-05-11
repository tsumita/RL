#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:37:15 2017

@author: daisuke
"""
import sys
sys.path.append("/Users/daisuke/.pyenv/versions/anaconda3-2.4.0/lib/python3.5/site-packages")
import gym

#env = gym.envs.make("MountainCarContinuous-v0")
env = gym.make("CartPole-v0")
"""
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
"""
   
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break