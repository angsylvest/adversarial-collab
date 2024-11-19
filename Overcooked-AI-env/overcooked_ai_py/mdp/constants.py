import math 

lazy_agent = True
adv_agent = False 
advers_prob = 0.5
lazy_prob = 0.5

# Trust related params 
# Set both to False for general DDPG RL agent 
include_trust = True # Only TRAVOS
multi_dim_trust = True # For multi-TRAVOS

def is_close_enough(rl_pos, adv_pos, rl_orient=None, adv_orient=None):
    distance = math.sqrt((rl_pos[0] - adv_pos[0])**2 + (rl_pos[1] - adv_pos[1])**2)
    if distance < 3.0: 
        return True 
    return False 