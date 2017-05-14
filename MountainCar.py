# -*- coding: utf-8 -*-
import gym
import numpy as np
import math
import random

def Average_calculate(posi_grid, velo_grid):   
    C = np.empty((0,2), float)
    for i in posi_grid:
        for j in velo_grid:
            C = np.append(C, np.array([[i, j]]), axis=0)
    return C
            
def Phi_calculate(states, C, SD):  
    Phi = np.zeros([len(C), 1])
    for i in range(len(C)):
        diff = states - C[i]
        square_diff = diff[0]**2 + diff[1]**2
        Phi[i] = math.exp(- square_diff / (2 * (SD ** 2)))
    return Phi

def Policy_decision(Q, actions, param):
    #（ε-greedy）
    action = np.where(Q==max(Q))[0][0]
    policy = np.ones([actions]) * param["epsilon"] / actions
    policy[action] = 1 - param["epsilon"] + param["epsilon"] / actions
    return policy

def Action_decision(policy):
    sum_policy = 0
    ran = random.random()
    for a in range(len(policy)):
        sum_policy += policy[a]
        if sum_policy > ran:
            return a

def Least_square_estimate(Phi, Phi_pre, reward, param):
    X = Phi_pre - param["gamma"] * Phi
    Xnorm_inv = np.linalg.inv(np.dot(X.T, X))
    Theta_td = np.dot(Xnorm_inv, X.T) * reward
    return Theta_td[0]
    

def MountainCar_run(M, param):
    env = gym.make('MountainCar-v0')

    actions = 3

    posi_grid = [-1.2, -0.3, 0.6]
    velo_grid = [-0.07, -0.02, 0.02, 0.07]
    
    C = Average_calculate(posi_grid, velo_grid)

    Phi = np.zeros([len(C), 1])
    Theta = np.ones([actions, len(C)])
    
    for i_episode in range(M):
        
        observation = env.reset()
        Phi = Phi_calculate(observation, C, param["SD"])
        
        rewards = 0
        step = 0
        
        while True:
            step += 1
            env.render()
        
            Q = np.dot(Theta, Phi)
            
            policy = Policy_decision(Q, actions, param)
            
            action = Action_decision(policy)
            
            Phi_pre = Phi
            
            observation, reward, done, _ = env.step(action)
            
            rewards += reward
            
            if done:
                print("Episode finished after {} timesteps".format(step))
                r_list.append(rewards)
                break
            
            Phi = Phi_calculate(observation, C, param["SD"])
            
            # θ更新
            Theta[action] += Least_square_estimate(Phi, Phi_pre, reward, param)
    
        print(rewards)


if __name__=='__main__':
    episode = 1000
    
    param = {"SD":0.5, "gamma":0.95, "epsilon":0.1}
    MountainCar_run(episode, param)
