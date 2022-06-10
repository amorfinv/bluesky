"A simple metrics plugin for Bluesky."
import numpy as np
from bluesky import stack, traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.core import Entity, timed_function
from bluesky.tools.aero import nm, ft

def init_plugin():
    # Addtional initilisation code
    global metrics
    metrics = SimpleMetrics()
    # Configuration parameters
    config = {
        'plugin_name': 'SIMPLEMETRICS',
        'plugin_type': 'sim'
        }

    return config

# We need to create a class that inherits from the Entity class, as it
# contains a lot of useful functionality, such as the ability to use
# trafarrays, the create and delete functions, etc.
class SimpleMetrics(Entity):
    def __init__(self):
        super().__init__()
        # Initialise the conflict pairs and loss of separation pairs
        self.prevconfpairs = set()
        self.prevlospairs = set()
        
        # Here we initialise the metrics we want to keep track of globally,
        # in this case number of conflicts and losses of separation.
        self.conflict_no = 0
        self.los_no = 0
        
        # By default, the plugin is disabled.
        self.enabled = False
        
        # Metrics that are aircraft specific can be handled in trafarrays
        with self.settrafarrays():
            self.distance_horizontal = np.array([])
            self.distance_vertical = np.array([])
            self.spawntime = np.array([])
            
        return
            
    # Whenever we use trafarrays, we need to define a create function to set the initial values
    # This function is always called when an aircraft is created
    def create(self, n = 1):
        # The latest created aircraft is always at the end of the traf arrays
        self.distance_horizontal[-n:] = 0
        self.distance_vertical[-n:] = 0
        self.spawntime[-n:] = sim.simt
        
        return
        
    # Just as the create function, the delete function is called when an aircraft is deleted
    # In this case, we want to log the data when an aircraft is deleted
    def delete(self, acidxs):
        # Only run the function the plugin is set as enabled
        if not self.enabled:
            return
        
        # We print stuff here, but we can also save all this data to a file. 
        # The delete function can be called for several aircraft, so we need to
        # for loop through the received aircraft indexes.
        for acidx in acidxs:
            print('------------------------------------------------------')
            print(f'Aircraft {traf.id[acidx]} was created at {self.spawntime[acidx]}.')
            print(f'Aircraft {traf.id[acidx]} was deleted at {sim.simt}.')
            print(f'Aircraft {traf.id[acidx]} flew {self.distance_horizontal[acidx]} m horizontally.')
            print(f'Aircraft {traf.id[acidx]} flew {self.distance_vertical[acidx]} m vertically.')
            
        return
        
    
    # The update function is a timed function, meaning it is run once every "dt" seconds.
    @timed_function(dt = 1)
    def update_metrics(self):
        # Only run the function the plugin is set as enabled
        if not self.enabled:
            return
        
        # First we do the conflicts. We only want unique conflicts, so we check the traffic
        # object for new conflicts that have appeared since the last update.
        confpairs_new = list(set(traf.cd.confpairs) - self.prevconfpairs)
        
        if confpairs_new:
            # Add the number of new conflicts to the global conflict counter
            self.conflict_no += len(confpairs_new)
            
        # Same for the loss of separation pairs.
        lospairs_new = list(set(traf.cd.lospairs) - self.prevlospairs)
        
        if lospairs_new:
            # Add the number of new losses of separation to the global loss of separation counter
            self.lospairs_no += len(lospairs_new)
        
        # Now we can also update the travelled distances
        self.distance_horizontal += sim.simdt * abs(traf.gs)
        self.distance_vertical += sim.simdt * abs(traf.vs)
        
        return
        
    # We want the plugin to run only when it is enabled via a stack command. 
    # This is done by adding a command to the stack using the @stack.command decorator.
    # The name of the function will then be the name of the command (or not, if it is specificed
    # in the decorator arguments).
    @stack.command()
    def simplemetricsON(self):
        self.enabled = True
        
        return
    

