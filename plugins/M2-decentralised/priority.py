import bluesky as bs
import numpy as np
from bluesky import stack
from bluesky.core import Entity, trafficarrays

def init_plugin():
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'PRIORITY',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        'update':          update
    }

    return config

def update():
    priority.update()

class Priority(Entity):
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.priority = np.array([], dtype=int)
        
        bs.traf.priority = self.priority

    def create(self, n=1):
        super().create(n)
        self.priority[-n:] = 1
        bs.traf.priority = self.priority
        
    def update(self):
        bs.traf.priority = self.priority

# Addtional initilisation code
priority = Priority()

@stack.command
def getprio(acid: 'acid'):
    stack.stack(f'ECHO The priority of {bs.traf.id[acid]} is {bs.traf.priority[acid]}.')
    return