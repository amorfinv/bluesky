''' Bird traffic simulation plugin '''
from click import progressbar
import numpy as np

import bluesky as bs
from bluesky import stack
from bluesky.core.walltime import Timer
from bluesky.core import timed_function


def init_plugin():

    config = {
        # The name of your plugin
        'plugin_name'      : 'progressbar',
        'plugin_type'      : 'sim',
                }

    return config
        
@timed_function(dt=10)
def update():
    ''' Periodic update function for metrics calculation. '''
    
    data = dict()
    data['scenario_name'] = bs.stack.get_scenname()
    data['scenario_time'] = bs.sim.simt
    
    # send bird data
    bs.net.send_event(b'PROGRESS', (bs.stack.get_scenname(), bs.sim.simt))