import bluesky as bs
from bluesky import stack
from bluesky.core import Entity, trafficarrays

def init_plugin():
    # Addtional initilisation code
    priority = Priority()
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'PRIORITY',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config

class Priority(Entity):
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.priority = []
        bs.traf.priority = self.priority

@stack.command
def crem2(acid: 'txt', actype: 'txt', lat: 'lat', lon: 'lon',
          hdg: 'hdg', alt: 'alt', spd: 'spd', prio: int = 0):
    """CREM2 acid, type, [latlon], [hdg], [alt], [spd], prio"""
    # Creates an aircrft, but also assigns priority
    # Convert stuff for bs.traf.cre
    
    # First create the aircraft
    bs.traf.cre(acid, actype, lat, lon, hdg, alt, spd)
    # Then assign its priority
    bs.traf.priority[-1] = prio
    return

@stack.command
def getprio(acid: 'acid'):
    stack.stack(f'ECHO The priority of {bs.traf.id[acid]} is {bs.traf.priority[acid]}.')
    return