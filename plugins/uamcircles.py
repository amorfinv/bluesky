from logging import warning
import numpy as np
from os import path

import bluesky as bs
from bluesky import settings, stack
from bluesky.ui import palette
from bluesky.ui.qtgl.gltraffic import Traffic
from bluesky.ui.qtgl import glhelpers as glh

def init_plugin():
    ''' Plugin initialisation function. '''
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'UAMGUI',

        # The type of this plugin.
        'plugin_type':     'gui',
        }

    # init_plugin() should always return a configuration dict.
    return config

palette.set_default_colours(
    caution=(255,255,0),
    warning=(255,165,0),
    collision=(255,0,0),
)

MAX_NAIRCRAFT = 10000

# UAM traffic class
class UAMTraffic(Traffic):
    def __init__(self):
        super().__init__()

        # create the buffers
        # These store the radius of the circles in GPU
        self.caution_buffer = glh.GLBuffer()
        self.warning_buffer = glh.GLBuffer()
        self.collision_buffer = glh.GLBuffer()

        # color buffers
        self.caution_color = glh.GLBuffer()
        self.warning_color = glh.GLBuffer()
        self.collision_color = glh.GLBuffer()

        # create circle buffers
        self.caution_circle = glh.Circle()
        self.warning_circle = glh.Circle()
        self.collision_circle = glh.Circle()

        # initialize values for radius
        self.caution_radius = 60.0
        self.warning_radius = 40.0
        self.collision_radius = 30.0

    def create(self):
        super().create()

        # create buffers
        self.caution_buffer.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.warning_buffer.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.collision_buffer.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)

        self.caution_color.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.warning_color.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.collision_color.create(MAX_NAIRCRAFT * 4, glh.GLBuffer.UsagePattern.StreamDraw)

        # link the circle buffers to the color and radius buffers

        self.caution_circle.create(radius=1.0)
        self.caution_circle.set_attribs(lat=self.lat, lon=self.lon, scale=self.caution_buffer,
                                       color=self.caution_color, instance_divisor=1)

        self.warning_circle.create(radius=1.0)
        self.warning_circle.set_attribs(lat=self.lat, lon=self.lon, scale=self.warning_buffer,
                                       color=self.warning_color, instance_divisor=1)

        self.collision_circle.create(radius=1.0)
        self.collision_circle.set_attribs(lat=self.lat, lon=self.lon, scale=self.collision_buffer,
                                        color=self.collision_color, instance_divisor=1)


    def draw(self):

        # draw them in the gui
        super().draw()

        actdata = bs.net.get_nodedata()
        #  circles only when they are bigger than the A/C symbols
        if actdata.zoom >= 0.15:
            self.shaderset.set_vertex_scale_type(
                self.shaderset.VERTEX_IS_METERS)

            # draw the circles    
            self.caution_circle.draw(n_instances=actdata.naircraft)
            self.warning_circle.draw(n_instances=actdata.naircraft)
            self.collision_circle.draw(n_instances=actdata.naircraft)


    def update_aircraft_data(self, data):
        super().update_aircraft_data(data)

        naircraft = len(data.lat)
        if naircraft != 0:

            # Here we assume that all circles are the same.
            # if different then we need to make a sim plugin to communicate this info
            caution_array = np.ones(naircraft, dtype=np.float32)*self.caution_radius
            warning_array = np.ones(naircraft, dtype=np.float32)*self.warning_radius
            collision_array = np.ones(naircraft, dtype=np.float32)*self.collision_radius

            self.caution_buffer.update(np.array(caution_array, dtype=np.float32))
            self.warning_buffer.update(np.array(warning_array, dtype=np.float32))
            self.collision_buffer.update(np.array(collision_array, dtype=np.float32))
            
            
            # set colors
            caution_color = np.empty((min(naircraft, MAX_NAIRCRAFT), 4), dtype=np.uint8)
            warning_color = np.empty((min(naircraft, MAX_NAIRCRAFT), 4), dtype=np.uint8)
            collision_color = np.empty((min(naircraft, MAX_NAIRCRAFT), 4), dtype=np.uint8)

            # TODO: no for loop needed probably
            for i in range(naircraft):

                caution_color[i, :] = tuple(palette.caution) + (255,)
                warning_color[i, :] = tuple(palette.warning) + (255,)
                collision_color[i, :] = tuple(palette.collision) + (255,)

            self.caution_color.update(caution_color)
            self.warning_color.update(warning_color)
            self.collision_color.update(collision_color)

    @stack.command
    def uamcircles(self, caution_radius: float, warning_radius: float, collision_radius: float):
        ''' Set the size of the circles. '''

        # create a numpy array that is self.ntraf size long and stores the radius
        self.caution_radius = caution_radius
        self.warning_radius = warning_radius
        self.collision_radius = collision_radius