#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def Average_calculate(posi_grid, velo_grid):
    """ ガウス関数の中心を求める """
    C = np.empty((0,2), float) # 状態の平均[座標,速度]
    for i in posi_grid:
        for j in velo_grid:
            C = np.append(C, np.array([[i, j]]), axis=0)
            
    return C
            
def Phi_calculate(states, C, param):
    """ 線形近似モデルの基底関数Φを求める """
    Phi = np.zeros([len(C), 1])    
    for i in range(len(C)):
        diff = states - C[i]
        square_diff = diff[0]**2 + diff[1]**2
        Phi[i] = math.exp(- square_diff / (2 * (param["SD"] ** 2)))
        
    return Phi

def Policy_decision(Q, actions, param):
    """ 政策の生成（ε-greedy） """
    action = np.where(Q==max(Q))[0][0]
    policy = np.ones([actions]) * param["epsilon"] / actions
    policy[action] = 1 - param["epsilon"] + param["epsilon"] / actions
          
    return policy

def Action_decision(policy):
    """ 行動選択 """
    sum_policy = 0
    ran = random.random()
    for a in range(len(policy)):
        sum_policy += policy[a]
        if sum_policy > ran:
            return a

def Theta_update(theta_one, Phi, Phi_pre, reward, e, param):
    """ モデルパラメータの更新値をを求める """
    Delta = reward + param["gamma"] * np.dot(theta_one, Phi) - np.dot(theta_one, Phi_pre)
    #e = param["gamma"] * param["lambda"] * e + Phi_pre
    e += Phi_pre
    theta_one += (param["alpha"] * Delta * e).T[0]
    e = param["gamma"] * Delta * e
    
    return theta_one
        

def MountainCar_run(M, param):
    """ エージェントにMエピソード分の訓練をさせる """
    env = gym.make('MountainCar-v0')

    actions = 3

    posi_grid = [-1.2, -0.3, 0.6]
    velo_grid = [-0.07, -0.02, 0.02, 0.07]
    
    C = Average_calculate(posi_grid, velo_grid)

    Phi = np.zeros([1, len(C)])
    Theta = np.ones([actions, len(C)])
    
    rewards_list = []
    
    for i_episode in range(M):
        
        observation = env.reset()
        Phi = Phi_calculate(observation, C, param)
        
        rewards = 0
        step = 0
        e = 0
        
        while True:
            step += 1
            #env.render() # 環境を与える          
            Q = np.dot(Theta, Phi)
            # 行動価値出力
            #print(Q.T, end = ' ')
            policy = Policy_decision(Q, actions, param)            
            action = Action_decision(policy)            
            # 行動前に基底関数保持              
            Phi_pre = Phi
            
            # 行動から観測を得る
            # observation: [座標(-1.2,0.6),速度(-0.07,0.07)]、 reward: [各タイムステップ -1, ゴール 0.5]、 done: [True,False]
            observation, reward, done, _ = env.step(action)
            
            rewards += reward
            
            Phi = Phi_calculate(observation, C, param)
            # θ更新
            Theta[action] = Theta_update(Theta[action], Phi, Phi_pre, reward, e, param)
            
            if done:
                print("Episode {0} finished after {1} timesteps".format(i_episode+1, step))
                rewards_list.append(rewards)
                env.render()
                break
            
        param["epsilon"] *= 0.999
        if np.mean(rewards_list[-100:]) > -110:
            print("Complete training after {} episodes".format(i_episode))
            break
            
            
    return rewards_list 
        

if __name__=='__main__':
    episode = 5000
    
    param = {"SD":0.1, "gamma":0.2, "epsilon":0.1, "lambda": 0.01, "alpha":0.1}
    rewards_list = MountainCar_run(episode, param)
    
    plt.clf()
    plt.plot(range(len(rewards_list)), rewards_list)
    plt.savefig('mountaincar_reward.png')
