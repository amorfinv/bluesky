import numpy as np
from os import path

import bluesky as bs
from bluesky import settings, stack
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
        self.caution_radius = 0.0
        self.warning_radius = 0.0
        self.collision_radius = 0.0



    def create(self):
        super().create()

        # create the objects
        # link the circle buffers to the color and radius buffers

        self.caution_circle.create(radius=1.0)
        self.caution_circle.set_attribs(lat=self.lat, lon=self.lon, scale=self.caution_buffer,
                                       color=self.caution_color, instance_divisor=1)

        self.warning_circle.create(radius=1.0)
        self.warning_circle.set_attribs(lat=self.lat, lon=self.lon, scale=self.warning_circle,
                                       color=self.warning_color, instance_divisor=1)

        self.collision_circle.create(radius=1.0)
        self.collision_circle.set_attribs(lat=self.lat, lon=self.lon, scale=self.collision_circle,
                                        color=self.collision_color, instance_divisor=1)


    def draw(self):

        # draw them in the gui
        super().draw()

        actdata = bs.net.get_nodedata()

        #  circles only when they are bigger than the A/C symbols
        if actdata.show_pz and actdata.zoom >= 0.15:
            self.shaderset.set_vertex_scale_type(
                self.shaderset.VERTEX_IS_METERS)

            # draw the circles    
            self.collision_circle.draw(n_instances=actdata.naircraft)
            self.warning_circle.draw(n_instances=actdata.naircraft)
            self.caution_circle.draw(n_instances=actdata.naircraft)


    @stack.command
    def uamcircles(self, caution_radius: float, warning_radius: float, collision_radius: float):
        ''' Set the size of the circles. '''

        # create a numpy array that is self.ntraf size long and stores the radius
        self.caution_radius = caution_radius
        self.warning_radius = warning_radius
        self.collision_radius = collision_radius


    def update_aircraft_data(self, data):
        super().update_aircraft_data(data)


        # set the color of the circles
        # TODO: make it dynamic
        # set rgb red color tuple
        rgb_red = (1.0, 0.0, 0.0)
        rgb_orange = (1.0, 0.5, 0.0)
        rgb_yellow = (1.0, 1.0, 0.0)

        self.caution_color.update(rgb_red)
        self.warning_color.update(rgb_orange)
        self.collision_color.update(rgb_yellow)


        # create a numpy array that is self.ntraf size long and stores the radius
        caution_array = np.ones(self.ntraf, dtype=np.float32)*self.caution_radius
        warning_array = np.ones(self.ntraf, dtype=np.float32)*self.warning_radius
        collision_array = np.ones(self.ntraf, dtype=np.float32)*self.collision_radius
        
        # set the color of the circles
        self.caution_buffer.update(np.array(caution_array, dtype=np.float32))
        self.warning_buffer.update(np.array(warning_array, dtype=np.float32))
        self.collision_buffer.update(np.array(collision_array, dtype=np.float32))