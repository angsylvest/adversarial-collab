from collections import deque
import numpy as np 
from overcooked_ai_py.mdp.constants import *

class Influence: 

    # TODO: only using sliding window ..
    def __init__(self, include_num_interactions=False, sliding_window=True, total_window=None, potential_window=None,interaction=None, influence_calc=None, reward_loss=None, num_collected=None ):

        self.include_num_interactions = include_num_interactions 
        self.sliding_window = sliding_window 
        
        if not num_collected: 
            self.num_collected = 0 
        else: 
            self.num_collected = num_collected

        if not influence_calc and not reward_loss: 
            self.influence_calc = 0 
            self.reward_loss = 0 
        else: 
            self.influence_calc = influence_calc
            self.reward_loss = reward_loss
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

        # influence relevant data .. 
        self.influence_score = self.adaptively_discretize_trust(self.calc_influence())
        self.interaction_score = self.adaptively_discretize_trust(self.calc_influence())


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

            # influence relevant data .. 
            self.influence_score = self.adaptively_discretize_trust(self.calc_influence())
            self.interaction_score = self.adaptively_discretize_trust(self.calc_influence())

            # print(f'influence score: {len(self.total_window)}')
    
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
        if not self.total_window or not self.potential_window:
            return 0  # Avoid division by zero if no data is present

        if self.sliding_window: 
            # include interaction info if necessary 
            avg_total = sum(self.total_window) / len(self.total_window)
            avg_potential = sum(self.potential_window) / len(self.potential_window)
            
            if avg_total + avg_potential == 0:
                return 0  # Avoid division by zero if both averages are zero
            
            percent_diff = abs(avg_total - avg_potential) / ((avg_total + avg_potential) / 2) * 100
            return percent_diff
        

    def calc_interaction(self): 
        count_stable = self.interaction.count_instances(1) 
        return count_stable / len(count_stable)

    def to_dict(self, pos, orientation, held_obj):
        # update obs space 
        pass 