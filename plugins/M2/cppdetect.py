''' CPP conflict detection. '''
from bluesky.traffic.asas import ConflictDetection
import casas

def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'CPPSTATEBASED',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config

class CPPStateBased(ConflictDetection):
    def __init__(self):
        super().__init__()
        self.detect = casas.detect
        return