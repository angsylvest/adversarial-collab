class Influence: 

    def __init__(self, include_num_interactions=False, sliding_window=False):

        self.include_num_interactions = include_num_interactions 

        self.sliding_window = sliding_window 
        self.num_collected = 0 
        # if not using sliding window, will just average everything 
        # if using sliding window, will only average most recent 100

    def update(self, interact_info):
        pass 

    def to_dict(self, pos, orientation, held_obj):
        # update obs space 
        pass 