from collections import deque
import numpy as np 
from overcooked_ai_py.mdp.constants import *

class Influence: 
    # update to include params relevant to keep track of 

    def __init__(self, include_num_interactions=False, sliding_window=False, total_window=None, potential_window=None,interaction=None ):

        self.include_num_interactions = include_num_interactions 

        self.sliding_window = sliding_window 
        self.num_collected = 0 

        self.influence_calc = 0 
        self.reward_loss = 0 
        # if not using sliding window, will just average everything 
        # if using sliding window, will only average most recent 100

        if self.sliding_window: 
            if not total_window and not potential_window and not interaction: 
                self.total_window = deque(maxlen=100)
                self.potential_window = deque(maxlen=100)
                self.interaction = deque(maxlen=100)
            else: 
                self.total_window = total_window
                self.potential_window = potential_window
                self.interaction = interaction



    def update(self, interact_info):
        if self.sliding_window: 
            total_rw_gained = interact_info["total_reward_gained"]
            total_rw_potential = interact_info["total_reward_potential"]

            self.total_window.append(total_rw_gained)
            self.potential_window.append(total_rw_potential)
            
            interaction_occured = interact_info["interaction_occured"]
            if interaction_occured == True: 
                self.interaction.append(1)
            else: 
                self.interaction.append(0)

    
    def one_hot(self, index, num_bins):
        """
        Converts an index value to a one-hot encoded vector.
        """
        one_hot = np.zeros(num_bins)
        one_hot[index] = 1
        return one_hot
    

    def adaptively_discretize_trust(self, trust_score, adapt=False):

        if trust_score > 0.98 and adapt:
            # Focus on fine-grained changes for high trust
            bins = [0.99, 0.995, 0.998, 1.0]
        else:
            # Broader bins for lower trust
            # bins = [0, 0.5, 0.8, 0.95, 0.99]
            bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        index = np.digitize([trust_score], bins, right=True)[0] 
        if one_hot_encode:
            return self.one_hot(index-1, len(bins))
            
        else: 
            return index
         

    def calc_influence(self): 
        # include interaction info if necessary 
        pass 

    def to_dict(self, pos, orientation, held_obj):
        # update obs space 
        pass 