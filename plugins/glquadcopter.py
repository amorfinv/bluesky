""" Bird traffic gui plugin """
import numpy as np
from os import path

import bluesky as bs
from bluesky import ui 
from bluesky.ui import palette
from bluesky import settings
import bluesky.ui.qtgl.glhelpers as glh
from bluesky.tools.aero import ft
from bluesky.ui.qtgl.guiclient import UPDATE_ALL
from bluesky.ui.qtgl.gltraffic import Traffic

# Register settings defaults
settings.set_variable_defaults(
    text_size=13, ac_size=16,
    asas_vmin=200.0, asas_vmax=500.0)

# Static defines
MAX_NAIRCRAFT = 10000
MAX_NCONFLICTS = 25000
MAX_ROUTE_LENGTH = 500
ROUTE_SIZE = 500
TRAILS_SIZE = 1000000

### Initialization function of your plugin.
def init_plugin():
    config = {
        'plugin_name':     'QUADGUI',
        'plugin_type':     'gui',
        }

    return config

# Bird traffic class
class QuadTraffic(Traffic):
    # TODO: FIX labels
    def __init__(self):
        super().__init__()
    
    def create(self):
        super().create()

        # get aircraft size
        ac_size = settings.ac_size

        # set square in aircraft vertices
        acvertices = np.array([(-1.0*ac_size, -1.0 * ac_size), (-1.0 * ac_size, 1.0 * ac_size),
                               (1.0*ac_size, 1.0 * ac_size), (1.0 * ac_size, -1.0 * ac_size)],
                              dtype=np.float32)

        # texture coordinates
        texcoords = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=np.float32)

        # filepath of the texture
        fname = path.join(settings.gfx_path, 'quadcopter.png')

        # create the vertex array object. NOTE: will get a warning. But ignore it.
        self.ac_symbol.create(vertex=acvertices, texcoords=texcoords, texture=fname)

