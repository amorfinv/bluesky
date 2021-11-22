from shapely.geometry import Polygon

import bluesky as bs
from bluesky import stack
from plugins.geofence import Geofence

def init_plugin():
    # Configuration parameters
    config = {
        'plugin_name': 'CITYJSON',
        'plugin_type': 'sim',
    }
    return config


@stack.command()
def load_cityjson(filename):
    """LOAD CITY JSON FILE"""

    # Create geofences from polygons
    # Geofence(geofence['name'], geofence['coordinates'], geofence['top'], geofence['bottom'])
    pass

@stack.command()
def delete_cityjson(filename):
    """DELETE CITY JSON FILE"""
    pass

# Simplification of geometry
