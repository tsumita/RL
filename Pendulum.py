#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: daisuke
"""
import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import operator


def Policy_function(action, state, mu, sigma):
    """ 政策を表す関数 """
    factor1 = 1 / (sigma * math.sqrt(2 * math.pi))
    factor2 = - (action - np.dot(mu.T, state))**2/ (2 * sigma**2)
    return factor1 * math.exp(factor2) 

def Policy_function_diff(action, state, mu, sigma):
    """ 政策関数の導関数 """
    factor1 = 1 / (sigma * math.sqrt(2 * math.pi))
    factor2 = - (action - np.dot(mu.T, state))**2 / (2 * sigma**2)
    factor3 = - (action - np.dot(mu.T, state)) / sigma**2
    return factor1 * math.exp(factor2) * factor3
    
def gradient_descent(state, mu, sigma, learning_rate = 0.5, steps = 50):
    """ 導関数から勾配法により政策が最大となる行動を求める """
    max_p_dict = {}
    actions = [-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0]
    for action in actions:
        #action = random.uniform(-2.0, 2.0)
        for _ in range(steps):
            grad = Policy_function_diff(action, state, mu, sigma)
            action += learning_rate * grad[0]
            if action >= 2.0 or action <= -2.0:
                break
        max_p_dict.update({action[0]: Policy_function(action, state, mu, sigma)})
        
    bestAction = max(max_p_dict.items(), key=operator.itemgetter(1))
    #print(bestAction)
    a = max(-2.0, bestAction[0])
    a = min(2.0, a)
    return [a]
    

def Pendulum_run(L, M, param):
    env = gym.make("Pendulum-v0")
    
    #mu = np.array([[random.uniform(-2, 2)] for _ in range(3)]) # 平均 μ
    mu = np.zeros([2,1])
    sigma = random.uniform(-2, 2)
    print("init sigma:",sigma)
    
    rewards_list = []
    
    for l in range(L):
        r = 0
        rewards_disc_list = np.array([])
        mu_numer = np.empty((0, 2))
        sigma_numer = np.array([])
        mu_denom = np.empty((0, 2))
        sigma_denom = np.array([])
        
        for i_episode in range(M):
            observation = env.reset()
            state = np.array([observation[0:2]]).T
            step = 0
            rewards = 0
            rewards_disc = 0
            mu_grad = np.zeros([2,1])
            sigma_grad = 0
            
        
            while True:
                step += 1
                
                #action = gradient_descent(state, mu, sigma)
                action = random.gauss(0, max(-sigma*0.1,sigma*0.1)) + np.dot(mu.T, state)[0]
                action = max([-2.0], action)
                action = min([2.0], action)
                
                
                observation, reward, done, _ = env.step(action)
                print(observation[0:2], reward)
                state = np.array([observation[0:2]]).T
            
                r += reward
                rewards += reward
                rewards_disc += param["gamma"]**(step - 1) * reward
                                
                mu_grad += (action - np.dot(mu.T, state)) * state / (sigma**2)
                sigma_grad += ((action - np.dot(mu.T, state))**2 - sigma**2) / (sigma**3)
            
                if i_episode == M-1:
                    rewards_list.append(rewards)
                    env.render()
            
                if done:
                    #print("Episode {0} finished after {1} timesteps, rewards {2}".format(i_episode+1, step, rewards))
                    break
                
            rewards_disc_list = np.append(rewards_disc_list, [rewards_disc], axis=0)
            
            mu_numer = np.append(mu_numer, (rewards_disc * mu_grad ** 2).T, axis=0)
            sigma_numer = np.append(sigma_numer, (rewards_disc * sigma_grad ** 2)[0], axis=0)
            mu_denom = np.append(mu_denom, mu_grad.T, axis=0)
            sigma_denom = np.append(sigma_denom, sigma_grad[0], axis=0)
         
        print("Repitation {0}, rewards {1}".format(l, r))
            
        b_mu = sum(mu_numer) / sum(mu_denom**2)
        b_sigma = sum(sigma_numer) / sum(sigma_denom**2)
        J_mu_grad = sum((np.array([rewards_disc_list]).T - np.array([b_mu])) * mu_denom) / M
        J_sigma_grad = sum((rewards_disc_list - b_sigma) * sigma_denom) / M
                                                               
        print("-"*20)
           
        """
        J_mu_grad = np.array([min(J_mu_grad[i], 20) for i in range(3)])
        J_mu_grad = np.array([max(J_mu_grad[i], -20) for i in range(3)])
        J_sigma_grad = min(J_sigma_grad, 2)
        J_sigma_grad = max(J_sigma_grad, -2)
        """
        
        print("mu_grad",J_mu_grad)
        print("sigma_grad",J_sigma_grad)
        
        mu += np.array([param["alpha"] * J_mu_grad]).T
        sigma += param["alpha"] * J_sigma_grad
                      
        print("mu",mu)
        print("sigma",sigma)
        
        
            
    return rewards_list
    
if __name__ == '__main__':
    num_episode = 100
    repetition = 2000
    
    param = { "gamma":0.8, "alpha":0.05}
    rewards_list = Pendulum_run(repetition, num_episode, param)
    
    plt.clf()
    plt.plot(range(len(rewards_list)), rewards_list)
    plt.savefig('pendulum_reward.png')
    