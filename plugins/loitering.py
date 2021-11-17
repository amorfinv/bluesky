import bluesky as bs
from bluesky import stack
from bluesky.core import Entity
from bluesky.core.simtime import timed_function
from plugins.geofence import Geofence

def init_plugin():
    ''' Plugin initialisation function. '''
    loiter = Loitering()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'Loitering',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

check_dt = 10
class Loitering(Entity):
    def __init__(self):
        super().__init__()
        self.loitergeofences = dict()
        with self.settrafarrays():
            self.futuregeofences = []
            self.geodurations = []
        bs.traf.loiter = self
    
    @staticmethod
    @stack.command
    def creloiter(acid:'txt', actype:'txt', aclat:'lat', aclon:'lon', achdg:'hdg', acalt:'alt', acspd:'spd', geodur:float, *geocoords:float):
        '''Create a loitering aircraft.'''
        # First, create aircraft
        bs.traf.cre(acid, actype, aclat, aclon, achdg, acalt, acspd)
        
        # Store the geofence data in the array until it needs to be enacted
        bs.traf.loiter.futuregeofences[acid] = geocoords
        bs.traf.loiter.geodurations[acid] = geodur
    
    @staticmethod
    @stack.command
    def delloiter(acidx:'acid'):
        '''Delete loitering aircraft, add its geofence.'''
        acid = bs.traf.id[acidx]

        # First of all, add the geofence
        geofence = Geofence(f'LOITER{acid}', bs.traf.loiter.futuregeofences[acidx])

        bs.traf.loiter.loitergeofences[acid] = {'geofence':geofence, 
                                                'time_left':bs.traf.loiter.geodurations[acidx]}
        
        # add constrained nodes inside this geofence to Geofence.nodes_in_loiter_geofence
        geofence.update_nodes_in_loitering_geofences(f'LOITER{acid}', update='add')

        # Then delete the aircraft
        bs.traf.delete(acidx)
        
    @staticmethod   
    @timed_function(dt = check_dt)
    def keep_track_loitering():
        '''Keep track of loiter geofences, and delete them when they have expired.'''
        # iterate through dictionary entries
        # This shouldn't take too long, there won't be many entries in this dictionary
        print(bs.traf.loiter.loitergeofences)
        temp = bs.traf.loiter.loitergeofences.copy()
        for acid in temp:
            # Decrement time
            bs.traf.loiter.loitergeofences[acid]['time_left'] -= check_dt
            # Check if time is negative
            if bs.traf.loiter.loitergeofences[acid]['time_left'] < 0:
                # remove nodes from nodes_in_geofence
                Geofence.update_nodes_in_loitering_geofences(f'LOITER{acid}', update='del')

                # Delete geofence
                Geofence.delete(f'LOITER{acid}')
                bs.traf.loiter.loitergeofences.pop(acid)
                bs.traf.loiter.futuregeofences.pop(acid)
                bs.traf.loiter.geodurations.pop(acid)
                         
            
            
            
        
    