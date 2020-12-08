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
for f_count in range(51):
    f = open('jsonFiles/' + str(f_count) + '.json',) 
    data = json.load(f)
    if (f_count == 0):
        print(data['steps'][1506])
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

                        to_append.append(x)
                        to_append.append(y)

                        goal_dist=get_distance((x,y),(goalx,goaly))
                        sideline_dist=get_distance((x,y),(sidelinex,sideliney))

                        best_target, best_score = best_pass_target(p, [x, y])
                        to_append.append(best_target[0])
                        to_append.append(best_target[1])
                        to_append.append(best_score)

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
                        
                        for k in range(10):
                            to_append.append(p['sticky_actions'][k])
                        
                        to_append.append(state_action['action'][0])

                        # print(to_append)
                        csv_data.append(to_append)
                        
                    else:
                        continue
    # Closing file 
    f.close()
with open('plays_offense_expert.csv', 'w') as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(['x','y', 'best_pass_target_x', 'best_pass_target_y', 'passScore', 'ballx','bally','l0x','l0y','l1x','l1y','l2x','l2y','l3x','l3y','l4x','l4y','l5x','l5y','l6x','l6y','l7x','l7y','l8x','l8y','l9x','l9y','l10x','l10y','r0x','r0y','r1x','r1y','r2x','r2y','r3x','r3y','r4x','r4y','r5x','r5y','r6x','r6y','r7x','r7y','r8x','r8y','r9x','r9y','r10x','r10y','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','action'])
    for row in csv_data:
        writer.writerow(row)

    