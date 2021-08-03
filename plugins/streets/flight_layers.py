from bluesky.tools.misc import lat2txt
import json
import numpy as np
from numpy import *
from collections import Counter
from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning
import bluesky as bs
from bluesky import core, stack, traf, scr, sim  #settings, navdb, tools
# from bluesky.tools.aero import ft, kts, nm
# from bluesky.tools import geo

from bluesky.core import Entity, Replaceable


def init_plugin():

    config = {
        # The name of your plugin
        'plugin_name'      : 'layers',
        'plugin_type'      : 'sim',
        # 'update_interval'  :  1.0,
        'update':          update

        }

    return config

def update():
    pass

class FlightLayers():
    def __init__(self):
        self.layer_dict = {}
    

# layer dictionary



