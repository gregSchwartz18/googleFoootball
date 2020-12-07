from kaggle_environments.envs.football.helpers import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import math

import tensorflow as tf

# load the model from disk
filename = '/kaggle_simulations/agent/saved_model/model1.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename1 = '/kaggle_simulations/agent/saved_model/model.sav'
loaded_model1 = pickle.load(open(filename1, 'rb'))
#result = loaded_model.score(X_test, y_test)
#print(result)

filename = '/kaggle_simulations/agent/saved_model/defensive-movement.h5'
defensive_movement = tf.keras.models.load_model(filename)
defensive_movement.compile(optimizer='adam',
        loss=['mse'],
        metrics=['mae'])

directions = [[Action.TopLeft, Action.Top, Action.TopRight],
[Action.Left, Action.Idle, Action.Right],
[Action.BottomLeft, Action.Bottom, Action.BottomRight]]

#track raw data to troubleshoot...
track_raw_data=[]

perfectRange = [[0.7, 0.95], [-0.12, 0.12]]

def inside(pos, area):
    return area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]

def get_distance(pos1,pos2):
    return ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5

def get_heading(pos1,pos2):
    raw_head=math.atan2(pos1[0]-pos2[0],pos1[1]-pos2[1])/math.pi*180

    if raw_head<0:
        head=360+raw_head
    else:
        head=raw_head
    return head

def get_action(action_num):
    if action_num==0:
        return Action.Idle
    if action_num==1:
        return Action.Left
    if action_num==2:
        return Action.TopLeft
    if action_num==3:
        return Action.Top
    if action_num==4:
        return Action.TopRight
    if action_num==5:
        return Action.Right
    if action_num==6:
        return Action.BottomRight
    if action_num==7:
        return Action.Bottom
    if action_num==8:
        return Action.BottomLeft
    if action_num==9:
        return Action.LongPass
    if action_num==10:
        return Action.HighPass
    if action_num==11:
        return Action.ShortPass
    if action_num==12:
        return Action.Shot
    if action_num==13:
        return Action.Sprint
    if action_num==14:
        return Action.ReleaseDirection
    if action_num==15:
        return Action.ReleaseSprint
    if action_num==16:
        #return Action.Sliding
        return Action.Idle
    if action_num==17:
        return Action.Dribble
    if action_num==18:
        #return Action.ReleaseDribble
        return Action.Idle
    return Action.Right

@human_readable_agent
def agent(obs):

    
    controlled_player_pos = obs['left_team'][obs['active']]
    x = controlled_player_pos[0]
    y = controlled_player_pos[1]
    pactive=obs['active']
    
    if obs["game_mode"] == GameMode.Penalty:
        return Action.Shot
    if obs["game_mode"] == GameMode.Corner:
        if controlled_player_pos[0] > 0:
            return Action.Shot
    if obs["game_mode"] == GameMode.FreeKick:
        return Action.Shot
    
    # Make sure player is running.
    if  0 < controlled_player_pos[0] < 0.6 and Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    elif 0.6 < controlled_player_pos[0] and Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint
    
    #if we have the ball
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        dat=[]
        
        to_append=[]
        to_append1= []
        #return Action.Right
        #get controller player pos
        controlled_player_pos = obs['left_team'][obs['active']]

        if inside(controlled_player_pos, perfectRange) and controlled_player_pos[0] < obs['ball'][0]:
            return Action.Shot
        
        goalx=0.0
        goaly=0.0

        sidelinex=0.0
        sideliney=0.42

        to_append.append(x)
        to_append.append(y)
        
        goal_dist=get_distance((x,y),(goalx,goaly))
        sideline_dist=get_distance((x,y),(sidelinex,sideliney))
        to_append.append(goal_dist)
        to_append.append(sideline_dist)
        to_append1.append(goal_dist)
        to_append1.append(sideline_dist)
        
        for i in range(len(obs['left_team'])):
            dist=get_distance((x,y),(obs['left_team'][i][0],obs['left_team'][i][1]))
            head=get_heading((x,y),(obs['left_team'][i][0],obs['left_team'][i][1]))
            to_append.append(dist)
            to_append.append(head)
            to_append1.append(dist)
            to_append1.append(head)
        
        for i in range(len(obs['right_team'])):
            dist=get_distance((x,y),(obs['right_team'][i][0],obs['right_team'][i][1]))
            head=get_heading((x,y),(obs['right_team'][i][0],obs['right_team'][i][1]))
            to_append.append(dist)
            to_append.append(head)
            to_append1.append(dist)
            to_append1.append(head)
        
        
        if (len(obs['sticky_actions']) != 10):
            dat1 = []
            dat1.append(to_append1)
            predicted1=loaded_model1.predict(dat1)
            do=get_action(predicted1)
            if do == None:
                return Action.Right
            else:
                return do


        for i in range(10):
            to_append.append(obs['sticky_actions'][i])
        
        dat.append(to_append)
        dat1 = []
        dat1.append(to_append1)
        
        predicted=loaded_model.predict(dat)
        predicted1=loaded_model1.predict(dat1)

        if (predicted >= 9 and predicted <= 12):
            predicted1 = predicted
        
        do=get_action(predicted1)
        
        if do == None:
            return Action.Right
        else:
            return do
    
    # if we don't have ball run to ball
    else:

        to_append = []
        dat = []

        controlled_player_pos = obs['left_team'][obs['active']]
        x = controlled_player_pos[0]
        y = controlled_player_pos[1]
        # controlled_player_dir = obs['left_team_direction'][obs['active']]

        to_append.append(x)
        to_append.append(y)
        # to_append.append(controlled_player_dir[0])
        # to_append.append(controlled_player_dir[1])

        ballpos = obs['ball']

        to_append.append(ballpos[0])
        to_append.append(ballpos[1])
        # to_append.append(ballpos[2])

        balldir = obs['ball_direction']

        to_append.append(balldir[0])
        to_append.append(balldir[1])
        # to_append.append(balldir[2])


        if (obs['ball_owned_team'] == 1):
            to_append.append(1)
        else:
            to_append.append(0)


        if Action.Sprint in obs['sticky_actions']:
            to_append.append(1)
        else:
            to_append.append(0)

        if Action.Dribble in obs['sticky_actions']:
            return Action.ReleaseDribble
        
        
        dirsign = lambda x: 1 if abs(x) < 0.01 else (0 if x < 0 else 2)
        #where ball is going
        ball_targetx=obs['ball'][0]+(obs['ball_direction'][0]*5)
        ball_targety=obs['ball'][1]+(obs['ball_direction'][1]*5)
        e_dist=get_distance(obs['left_team'][obs['active']],obs['ball'])
        if e_dist > 1:
            if e_dist >.01:
                # Run where ball will be
                xdir = dirsign(ball_targetx - controlled_player_pos[0])
                ydir = dirsign(ball_targety - controlled_player_pos[1])
                return directions[ydir][xdir]
            else:
                # Run towards the ball.
                xdir = dirsign(obs['ball'][0] - controlled_player_pos[0])
                ydir = dirsign(obs['ball'][1] - controlled_player_pos[1])
                return directions[ydir][xdir]

        dat.append(to_append)
        predicted = defensive_movement.predict(dat)

        ydir = int(round(predicted[0][0]))
        xdir = int(round(predicted[0][1]))

        return directions[ydir][xdir]

