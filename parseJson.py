import json
import csv
import math

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

# Opening JSON file 
f = open('1.json',) 
  
# returns JSON object as  
# a dictionary 

# configuration
# description
# id
# info
# name
# rewards
# schema_version
# specification
# statuses
# steps
# title
# version

data = json.load(f)
csv_data = []

# print(data['steps'][1506])
totalSteps = len(data['steps'])
for i in range(totalSteps):
    states_actions = data['steps'][i]
    for j in range(2):
        state_action = states_actions[j]
        observation = state_action.get('observation')
        if (observation != None):
            players_raw = observation.get('players_raw')
            if (players_raw != None):
                p = players_raw[0]
                if (p.get("ball_owned_team") == 0 and p.get('game_mode') == 0):
                    to_append = []

                    goalx=0.0
                    goaly=0.0
                    sidelinex=0.0
                    sideliney=0.42

                    controlled_player_pos = p['left_team'][p['active']]
                    x = controlled_player_pos[0]
                    y = controlled_player_pos[1]

                    goal_dist=get_distance((x,y),(goalx,goaly))
                    sideline_dist=get_distance((x,y),(sidelinex,sideliney))
                    to_append.append(goal_dist)
                    to_append.append(sideline_dist)
                    for k in range(len(p['left_team'])):

                        

                        dist=get_distance((x,y),(p['left_team'][k][0],p['left_team'][k][1]))
                        head=get_heading((x,y),(p['left_team'][k][0],p['left_team'][k][1]))
                        to_append.append(dist)
                        to_append.append(head)
                        
                    for k in range(len(p['right_team'])):
                        dist=get_distance((x,y),(p['right_team'][k][0],p['right_team'][k][1]))
                        head=get_heading((x,y),(p['right_team'][k][0],p['right_team'][k][1]))
                        to_append.append(dist)
                        to_append.append(head)
                    
                    to_append.append(state_action['action'][0])

                    print(to_append)
                    csv_data.append(to_append)
                    
                else:
                    continue
# Closing file 
f.close()
with open('plays_offense_expert.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(['ballx','bally','l0x','l0y','l1x','l1y','l2x','l2y','l3x','l3y','l4x','l4y','l5x','l5y','l6x','l6y','l7x','l7y','l8x','l8y','l9x','l9y','l10x','l10y','r0x','r0y','r1x','r1y','r2x','r2y','r3x','r3y','r4x','r4y','r5x','r5y','r6x','r6y','r7x','r7y','r8x','r8y','r9x','r9y','r10x','r10y','action'])
    for row in csv_data:
        writer.writerow(row)

    