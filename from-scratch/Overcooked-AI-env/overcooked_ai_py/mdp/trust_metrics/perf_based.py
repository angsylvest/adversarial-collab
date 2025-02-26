class PerfBased: 

    def __init__(self, multi=False, sliding_window=False):
        self.sliding_window = sliding_window 
        self.num_collected = 0 
        # if not using sliding window, will just average everything 
        # if using sliding window, will only average most recent 100 

    def update(self, interact_info):
        pass 

    def to_dict(self, pos, orientation, held_obj):
        # update obs space 
        pass 