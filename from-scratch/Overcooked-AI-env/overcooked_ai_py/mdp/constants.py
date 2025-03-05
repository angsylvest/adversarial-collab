import math 

alg_type = "ShannonEntropy" # "multiTravos" # [multiTravos, baseline, Travos]
lazy_agent = True
adv_agent = False 
both = False 
if lazy_agent and adv_agent:
    both = True 
advers_prob = 0.5
lazy_prob = 0.5
discretize_trust = True 
adaptive_discretize = False
include_in = [True, True, True, True]
one_hot_encode = True 
include_thres = True 
sliding_window = True 

# Trust related params 
# Set both to False for general DDPG RL agent 
if alg_type == "baseline": 
    include_trust = False # True # Only TRAVOS
    multi_dim_trust = False # True # For multi-TRAVOS
elif alg_type == "Vulnerability" or alg_type == "ShannonEntropy" or alg_type == "Vulnerability+Freq":
    include_trust = True 
    multi_dim_trust = False
elif alg_type == "Travos": 
    include_thres = False 
    alg_type += f"_disc_{discretize_trust}_adapt_{adaptive_discretize}_window_{sliding_window}"
    include_trust = True # True # Only TRAVOS
    multi_dim_trust = False # True # For multi-TRAVOS    
else: 
    alg_type += f"_disc_{discretize_trust}_adapt_{adaptive_discretize}_int_count_{include_thres}_window_{sliding_window}"
    include_trust = True # True # Only TRAVOS
    multi_dim_trust = True # True # For multi-TRAVOS    

if lazy_agent: 
    save_path_include = f"_lazy_{lazy_prob}"

if adv_agent: 
    save_path_include = f"_adv_{advers_prob}"

else: 
    save_path_include = f"_coop"


def is_close_enough(rl_pos, adv_pos, rl_orient=None, adv_orient=None):
    distance = math.sqrt((rl_pos[0] - adv_pos[0])**2 + (rl_pos[1] - adv_pos[1])**2)
    if distance < 3.0: 
        return True 
    return False 
