#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt


def Average_calculate(posi_grid, velo_grid):   
    C = np.empty((0,2), float) # 状態の平均[座標,速度]

    for i in posi_grid:
        for j in velo_grid:
            C = np.append(C, np.array([[i, j]]), axis=0)
            
    return C
            
def Phi_calculate(states, action, C, SD):  
    Phi = np.zeros([1, 3*len(C)])
    
    for i in range(action*len(C), action*len(C)+12):
        diff = states - C[i%12]
        square_diff = diff[0]**2 + diff[1]**2
        Phi[0][i] = math.exp(- square_diff / (2 * (SD ** 2)))
        
    return Phi

def Policy_decision(Q, param):
    # 政策の生成（ε-greedy）
    action = np.where(Q==max(Q))[0][0]
    policy = np.ones([3]) * param["epsilon"] / 3
    policy[action] = 1 - param["epsilon"] + param["epsilon"] / 3
          
    return policy

def Action_decision(policy):
    sum_policy = 0
    ran = random.random()
    for a in range(len(policy)):
        sum_policy += policy[a]
        if sum_policy > ran:
            return a

def Least_square_estimate(X, reward):
    Xnorm_inv = np.linalg.inv(np.dot(X.T, X))
    Xnorm_inv_dot = np.dot(Xnorm_inv, X.T)
    Theta_td =  np.dot(Xnorm_inv_dot, reward)
    print(Theta_td)
    
    return Theta_td.T


def MountainCar_run(L, M, param):
    env = gym.make('MountainCar-v0')

    posi_grid = [-1.2, -0.3, 0.6]
    velo_grid = [-0.07, -0.02, 0.02, 0.07]
    
    C = Average_calculate(posi_grid, velo_grid)

    Phi = np.zeros([1, len(C)])
    Theta = np.ones([1, 3*len(C)])
    
    rewards_list = []
    
    for l in range(L):
        
        
        X = np.empty((0,3*len(C)))
        r = np.empty((0,1))
    
        for i_episode in range(M):
        
            observation = env.reset()
            Phi = Phi_calculate(observation, 0, C, param["SD"])
        
            rewards = 0
            step = 0
        
            while True:
                step += 1
                env.render() # 環境を与える
                Phi_a = np.reshape(Phi, (3, len(C)))
                
                if step == 1:
                    action = 0
                    
                Q = np.dot(np.reshape(Theta, (3, len(C))), Phi_a[action].T)
                # 行動価値出力
                #print(Q.T, end = ' ')
                policy = Policy_decision(Q, param)
                
                action = Action_decision(policy)
                
                # 行動前に基底関数保持              
                Phi_pre = Phi
                
                # 行動から観測を得る
                # observation: [座標(-1.2,0.6),速度(-0.07,0.07)]、 reward: [各タイムステップ -1, ゴール 0.5]、 done: [True,False]
                observation, reward, done, _ = env.step(action)
                
                rewards += reward
                
                if done:
                    print("Episode {0} finished after {1} timesteps".format(i_episode+1, step))
                    rewards_list.append(rewards)
                    break
                
                Phi = Phi_calculate(observation, action, C, param["SD"])
                
                X_line = Phi_pre - param["gamma"] * Phi
                X = np.append(X, X_line, axis=0)
                
                r = np.append(r, [[reward]], axis=0)
            
        
        Theta = Least_square_estimate(X, r)
        #print("X: {0}".format(X))
            
    return rewards_list 
        

if __name__=='__main__':
    episode = 10
    repetition = 50
    
    param = {"SD":0.2, "gamma":0.9, "epsilon":0.1} # 標準偏差 max 1.8くらい, 割引率
    rewards_list = MountainCar_run(repetition, episode, param)
    
    plt.clf()
    plt.plot(range(len(rewards_list)), rewards_list)
    plt.savefig('mountaincar_reward.png')