import numpy as np

Action = namedtuple('action', ['number'])
State = namedtuple('state', ['processing', 'memory', 'bandwidth'])

class networkenv():

    actions = [
        Action('local', 0),
        Action('offload_edge', 1),
        Action('offload_adjedge', 2),
        Action('offload_cloud', 3)
    ]



    def __init__(self):

        self.states = self.make_state_space()

    def make_state_space(self):
        


        return states




   

