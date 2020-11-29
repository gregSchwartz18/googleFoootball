import json
import csv
import math
import numpy as np

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


def closest_opponent_to_object(obs, o):
    """For a given object returns the closest opponent.
    Args:
      o: Source object.

    Returns:
      Closest opponent."""
    min_d = None
    closest = None
    for p in obs['right_team']:
      d = get_distance(o, p)
      if min_d is None or d < min_d:
        min_d = d
        closest = p
    assert closest is not None
    return closest

def closest_front_opponent(obs, o, target):
    """For an object and its movement direction returns the closest opponent.

    Args:
        o: Source object.
        target: Movement direction.
    Returns:
        Closest front opponent."""
    delta = target - o
    min_d = None
    closest = None
    for p in obs['right_team']:
        delta_opp = p - o
        if np.dot(delta, delta_opp) <= 0:
            continue
        d = get_distance(o, p)
        if min_d is None or d < min_d:
            min_d = d
        closest = p

    # May return None!
    return closest

def score_pass_target(obs, active, player):
    """Computes score of the pass between players.

    Args:
        active: Player doing the pass.
        player: Player receiving the pass.

    Returns:
        Score of the pass.
    """
    opponent = closest_opponent_to_object(obs, player)
    dist = get_distance(player, opponent)
    
    trajectory = np.array(player) - np.array(active)
    dist_closest_traj = None
    for i in range(10):
        position = active + (i + 1) / 10.0 * trajectory
        opp_traj = closest_opponent_to_object(obs, position)
        dist_traj = get_distance(position, opp_traj)
        if dist_closest_traj is None or dist_traj < dist_closest_traj:
            dist_closest_traj = dist_traj
    return -dist_closest_traj

def best_pass_target(obs, active):
    """Computes best pass a given player can do.

    Args:
        active: Player doing the pass.

    Returns:
        Best target player receiving the pass.
    """
    best_score = None
    best_target = None
    for player in obs['left_team']:
        if get_distance(player, active) > 0.3:
            continue
        score = score_pass_target(obs, active, player)
        if best_score is None or score > best_score:
            best_score = score
        best_target = player
    return best_target, best_score

# Opening JSON file 
  
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

csv_data = []
direction = 0
slide = 0
for f_count in range(71):
    f = open('jsonFiles/' + str(f_count) + '.json',) 
    data = json.load(f)
    # if (f_count == 0):
    #     print(data['steps'][1506])
    totalSteps = len(data['steps'])
    for i in range(totalSteps):
        states_actions = data['steps'][i]
        for j in range(2):
            state_action = states_actions[j]
            if (len(state_action['action']) == 0):
                continue
            observation = state_action.get('observation')
            if (observation != None):
                players_raw = observation.get('players_raw')
                if (players_raw != None):
                    obs = players_raw[0]
                    
                    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
                        continue
                    else:
                        
                        if (obs['game_mode'] != 0):
                            continue

                        to_append = []

                        goalx=0.0
                        goaly=0.0
                        sidelinex=0.0
                        sideliney=0.42

                        controlled_player_pos = obs['left_team'][obs['active']]
                        x = controlled_player_pos[0]
                        y = controlled_player_pos[1]
                        controlled_player_dir = obs['left_team_direction'][obs['active']]
                        

                        to_append.append(x)
                        to_append.append(y)

                        to_append.append(controlled_player_dir[0])
                        to_append.append(controlled_player_dir[1])

                        ballpos = obs['ball']

                        

                        
                        dist = get_distance(controlled_player_pos, ballpos)
                        if (dist < 0.02 or dist > 0.04 or controlled_player_pos[0] < -0.5 or controlled_player_pos[0] > 0):
                            continue
                        
                        to_append.append(ballpos[0])
                        to_append.append(ballpos[1])
                        to_append.append(ballpos[2])

                        balldir = obs['ball_direction']

                        to_append.append(balldir[0])
                        to_append.append(balldir[1])
                        to_append.append(balldir[2])

                        
                        

                        if (obs['left_team_yellow_card'][obs['active']]):
                            to_append.append(1)
                        else:
                            to_append.append(0)

                        
                        if (obs['ball_owned_team'] == 1):
                            
                            rightplayer = obs['right_team'][obs['ball_owned_player']]
                            
                            
                            to_append.append(rightplayer[0])
                            to_append.append(rightplayer[1])
                            rightplayer_dir = obs['right_team_direction'][obs['ball_owned_player']]
                            to_append.append(rightplayer_dir[0])
                            to_append.append(rightplayer_dir[1])

                        else:
                            continue

                        
                        for k in range(10):
                            to_append.append(obs['sticky_actions'][k])
                        
                        new_dist = get_distance(controlled_player_pos, rightplayer)
                        if (new_dist < 0.008 or new_dist > 0.016):
                            continue

                        if (state_action['action'][0] == 16):
                            slide += 1
                            
                            to_append.append(0)
                        elif (state_action['action'][0] >= 0 and state_action['action'][0] <= 8):
                            direction += 1
                            to_append.append(1)
                        
                        #to_append.append(state_action['action'][0])

                        # print(to_append)
                        print(slide, direction)
                        csv_data.append(to_append)
    # Closing file 
    f.close()
with open('slide.csv', 'w') as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(['x','y', 'dir_x', 'dir_y', 'ball_x', 'ball_y', 'ball_z', 'ball_dir_x', 'ball_dir_y', 'ball_dir_z', 'yellow_card', 'right_pos_x', 'right_pos_y', 'right_dir_x', 'right_dir_y', 's1','s2','s3','s4','s5','s6','s7','s8','s9','s10','action'])
    for row in csv_data:
        writer.writerow(row)

    