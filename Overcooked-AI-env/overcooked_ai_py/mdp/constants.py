import math 

lazy_agent = True
adv_agent = False 
advers_prob = 0.5
lazy_prob = 0.5
include_trust = True 
multi_dim_trust = True 

def is_close_enough(rl_pos, adv_pos, rl_orient=None, adv_orient=None):
    distance = math.sqrt((rl_pos[0] - adv_pos[0])**2 + (rl_pos[1] - adv_pos[1])**2)
    if distance < 3.0: 
        return True 
    return False 