# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:29:23 2020

@author: Shreyas Dharanesh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn
from mpl_toolkits.mplot3d import Axes3D

class HMM(object):
     
    def __init__(self, transition_matrix_matrix,current_state):
        self.transition_matrix = transition_matrix
        self.current_state = current_state
      
    def filtering(self,observation_matrix):
        new_state = np.dot(observation_matrix,np.dot(self.transition_matrix,self.current_state))
        new_state_normalized = new_state/np.sum(new_state)
        self.current_state = new_state_normalized
        return new_state_normalized
    
    def prediction(self):
        new_state = np.dot(self.transition_matrix,self.current_state)
        new_state_normalized = new_state/np.sum(new_state)
        self.current_state=new_state_normalized
        return new_state_normalized

    def plot_state(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        xpos = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
        ypos = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        zpos = np.zeros(len(initial_state.shape))
        dx = np.ones(len(initial_state.shape))
        dy = np.ones(len(initial_state.shape))
        dz = self.current_state      
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce8900')
        ax1.set_xticks([0., 1., 2., 3.])   
        ax1.set_yticks([0., 1., 2., 3.])   
        plt.show()
        
    def create_observation_matrix(self,error_rate, no_discrepancies):
        sensor_list=[]
        for number in no_discrepancies:
            probability=(1-error_rate)**(4-number)*error_rate**number
            sensor_list.append(probability)
            observation_matrix = np.zeros((len(sensor_list),len(sensor_list)))
            np.fill_diagonal(observation_matrix,sensor_list)
        return observation_matrix

################## MyCode to generate taransition_matrix for my map(map_8.png)
temp1=np.zeros((16,16))
for i in range(0,16):
    temp1[i,i]=0.2
    if(i<=11):
        temp1[i+4,i]=1
    if(i>=4):
        temp1[i-4,i]=1
    if(i<=14):
        temp1[i+1,i]=1
    if(i>=1):
        temp1[i-1,i]=1
temp1[1,0]=temp1[0,1]=temp1[7,3]=temp1[3,7]==0

count=0
for j in range(0,16):
    temp1[j,j]=0.2
    numofones=np.count_nonzero(temp1[:,j]==1)
    if numofones==1:
        temp1[:,j]=np.where(temp1[:,j]==1,0.8,temp1[:,j])
    elif numofones==2:
        temp1[:,j]=np.where(temp1[:,j]==1,0.4,temp1[:,j])
    elif numofones==3:
        temp1[:,j]=np.where(temp1[:,j]==1,0.267,temp1[:,j])
    elif numofones==4:
        temp1[:,j]=np.where(temp1[:,j]==1,0.2,temp1[:,j])

#Print the below lint to view my transition matrix if required for your verification by uncommenting
print (temp1)  
##################

#   define two models
transition_matrix = temp1
    
initial_state= np.array([1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16])

Model = HMM(transition_matrix,initial_state)
Model2 = HMM(transition_matrix,initial_state)

#   create observation matrices(10 sensor reading as required- SWE, NW, N, NE, S, SW, SE, E, W, NWE)
observation_matrix_SWE = Model.create_observation_matrix(0.25,[2,3,2,0,4,3,3,1,3,3,3,2,3,2,2,2])
observation_matrix_NW = Model.create_observation_matrix(0.25,[1,0,1,3,1,2,2,2,1,2,2,3,2,3,3,4])
observation_matrix_N = Model.create_observation_matrix(0.25, [2,1,2,4,0,1,1,3,0,1,1,2,1,2,2,3])
observation_matrix_NE = Model.create_observation_matrix(0.25, [3,2,3,2,1,2,2,4,1,2,2,3,0,1,1,2])
observation_matrix_S = Model.create_observation_matrix(0.25, [2,3,2,2,2,1,1,1,2,1,1,0,3,2,2,1])
observation_matrix_SW = Model.create_observation_matrix(0.25, [1,2,1,1,3,2,2,0,3,2,2,1,4,3,3,2])
observation_matrix_SE = Model.create_observation_matrix(0.25, [3,4,3,1,3,2,2,2,3,2,2,1,2,1,1,0])
observation_matrix_E = Model.create_observation_matrix(0.25, [4,2,1,2,2,1,1,3,2,1,1,2,1,0,0,1])
observation_matrix_W = Model.create_observation_matrix(0.25, [2,1,0,2,2,1,1,1,2,1,1,2,3,2,2,3])
observation_matrix_NWE = Model.create_observation_matrix(0.25, [2,1,2,2,2,3,3,3,2,3,3,4,1,2,2,3])

#   localize of the robot using filtering (Ten first timesteps)
print("**localize of the robot using filtering-Model (Ten first timesteps)**")
state_1 = Model.filtering(observation_matrix_SWE)
Model.plot_state()
state_2 = Model.filtering(observation_matrix_NW)
Model.plot_state()
state_3 = Model.filtering(observation_matrix_N)
Model.plot_state()
state_4 = Model.filtering(observation_matrix_NE)
Model.plot_state()
state_5 = Model.filtering(observation_matrix_S)
Model.plot_state()

state_6 = Model.filtering(observation_matrix_SW)
Model.plot_state()
state_7 = Model.filtering(observation_matrix_SE)
Model.plot_state()
state_8 = Model.filtering(observation_matrix_E)
Model.plot_state()
state_9 = Model.filtering(observation_matrix_W)
Model.plot_state()
state_10 = Model.filtering(observation_matrix_NWE)
Model.plot_state()


print("**Localize of the robot using filtering-Model2 (Five first timesteps) and prediction (Ten last timesteps)**")
#   localize of the robot using filtering (Five first timesteps) and prediction (Ten last timesteps)
state_11 = Model2.filtering(observation_matrix_SWE)
Model2.plot_state()
state_12 = Model2.filtering(observation_matrix_NW)
Model2.plot_state()
state_13 = Model2.filtering(observation_matrix_N)
Model2.plot_state()
state_14 = Model2.filtering(observation_matrix_NE)
Model2.plot_state()
state_15 = Model2.filtering(observation_matrix_S)
Model2.plot_state()
#print("Next 10 steps prediction")
prediction_1 = Model2.prediction()
Model2.plot_state()
prediction_2 = Model2.prediction()
Model2.plot_state()
prediction_3 = Model2.prediction()
Model2.plot_state()
prediction_4 = Model2.prediction()
Model2.plot_state()
prediction_5 = Model2.prediction()
Model2.plot_state()
prediction_6 = Model2.prediction()
Model2.plot_state()
prediction_7 = Model2.prediction()
Model2.plot_state()
prediction_8 = Model2.prediction()
Model2.plot_state()
prediction_9 = Model2.prediction()
Model2.plot_state()
prediction_10 = Model2.prediction()
Model2.plot_state()
