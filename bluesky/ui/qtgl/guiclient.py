''' I/O Client implementation for the QtGL gui. '''
try:
    from PyQt5.QtCore import QTimer
except ImportError:
    from PyQt4.QtCore import QTimer

import numpy as np

from bluesky.network import Client
from bluesky.tools import Signal
from bluesky.tools.aero import ft

# Globals
UPDATE_ALL = ['SHAPE', 'TRAILS', 'CUSTWPT', 'PANZOOM', 'ECHOTEXT']
ACTNODE_TOPICS = [b'ACDATA']


class GuiClient(Client):
    def __init__(self):
        super(GuiClient, self).__init__(ACTNODE_TOPICS)
        self.nodedata = dict()
        self.timer = None
        self.ref_nodedata = nodeData()

        self.timer = QTimer()
        self.timer.timeout.connect(self.receive)
        self.timer.start(20)
        self.subscribe(b'SIMINFO')

        # Signals
        self.actnodedata_changed = Signal()
        self.event_received      = Signal()
        self.stream_received     = Signal()

    def event(self, name, data, sender_id):
        sender_data = self.get_nodedata(sender_id)
        data_changed = []
        if name == b'RESET':
            sender_data.clear_scen_data()
            data_changed = list(UPDATE_ALL)
        elif name == b'SHAPE':
            sender_data.update_poly_data(**data)
            data_changed.append('SHAPE')
        elif name == b'DEFWPT':
            sender_data.defwpt(**data)
            data_changed.append('CUSTWPT')
        elif name == b'DISPLAYFLAG':
            sender_data.setflag(**data)
        elif name == b'ECHO':
            sender_data.echo(**data)
            data_changed.append('ECHOTEXT')
        elif name == b'PANZOOM':
            sender_data.panzoom(**data)
            data_changed.append('PANZOOM')
        elif name == b'SIMSTATE':
            sender_data.siminit(**data)
            data_changed = list(UPDATE_ALL)
        else:
            self.event_received.emit(name, data, sender_id)

        if sender_id == self.act and data_changed:
            self.actnodedata_changed.emit(sender_id, sender_data, data_changed)

    def stream(self, name, data, sender_id):
        self.stream_received.emit(name, data, sender_id)

    def actnode_changed(self, newact):
        self.actnodedata_changed.emit(newact, self.get_nodedata(newact), UPDATE_ALL)

    def get_nodedata(self, nodeid=None):
        nodeid = nodeid or self.act
        if not nodeid:
            return self.ref_nodedata

        data = self.nodedata.get(nodeid)
        if not data:
            # If this is a node we haven't addressed yet: create dataset and
            # request node settings
            self.nodedata[nodeid] = data = nodeData()
            self.send_event(b'GETSIMSTATE', target=nodeid)

        return data

    # def connect(self, hostname='localhost', event_port=0, stream_port=0, protocol='tcp'):
    #     super(GuiClient, self).connect(hostname, event_port, stream_port, protocol)
    #

    def sender(self):
        return self.sender_id


class nodeData(object):
    def __init__(self, route=None):
        # Stack window
        self.echo_text  = ''
        self.stackcmds  = dict()

        # Display pan and zoom
        self.pan       = (0.0, 0.0)
        self.zoom      = 1.0

        # Per-scenario data
        self.clear_scen_data()

        # Network route to this node
        self._route    = route

    def clear_scen_data(self):
        # Clear all scenario-specific data for sender node
        self.polynames = dict()
        self.polydata  = np.array([], dtype=np.float32)
        self.custwplbl = ''
        self.custwplat = np.array([], dtype=np.float32)
        self.custwplon = np.array([], dtype=np.float32)

        # Filteralt settings
        self.filteralt = False

        # Create trail data
        self.traillat0 = []
        self.traillon0 = []
        self.traillat1 = []
        self.traillon1 = []

        # Reset transition level
        self.translvl = 4500.*ft

        # Display flags
        self.show_map      = True
        self.show_coast    = True
        self.show_traf     = True
        self.show_pz       = False
        self.show_fir      = True
        self.show_lbl      = 2
        self.show_wpt      = 1
        self.show_apt      = 1
        self.ssd_all       = False
        self.ssd_conflicts = False
        self.ssd_ownship   = set()


    def siminit(self, shapes, **kwargs):
        self.__dict__.update(kwargs)
        for shape in shapes:
            self.update_poly_data(**shape)

    def panzoom(self, pan=None, zoom=None, absolute=True):
        if pan:
            if absolute:
                self.pan  = pan
            else:
                self.pan[0] += pan[0]
                self.pan[1] += pan[1]
        if zoom:
            self.zoom = zoom * (1.0 if absolute else self.zoom)

    def update_poly_data(self, name, shape='', coordinates=None):
        if name in self.polynames:
            # We're either updating a polygon, or deleting it. In both cases
            # we remove the current one.
            self.polydata = np.delete(self.polydata, list(range(*self.polynames[name])))
            del self.polynames[name]

        # Break up polyline list of (lat,lon)s into separate line segments
        if coordinates is not None:
            if shape == 'LINE' or shape[:4] == 'POLY':
                # Input data is list or array: [lat0,lon0,lat1,lon1,lat2,lon2,lat3,lon3,..]
                newdata = np.array(coordinates, dtype=np.float32)

            elif shape == 'BOX':
                # Convert box coordinates into polyline list
                # BOX: 0 = lat0, 1 = lon0, 2 = lat1, 3 = lon1 , use bounding box
                newdata = np.array([coordinates[0], coordinates[1],
                                 coordinates[0], coordinates[3],
                                 coordinates[2], coordinates[3],
                                 coordinates[2], coordinates[1]], dtype=np.float32)

            elif shape == 'CIRCLE':
                # Input data is latctr,lonctr,radius[nm]
                # Convert circle into polyline list

                # Circle parameters
                Rearth = 6371000.0             # radius of the Earth [m]
                numPoints = 72                 # number of straight line segments that make up the circrle

                # Inputs
                lat0 = coordinates[0]              # latitude of the center of the circle [deg]
                lon0 = coordinates[1]              # longitude of the center of the circle [deg]
                Rcircle = coordinates[2] * 1852.0  # radius of circle [NM]

                # Compute flat Earth correction at the center of the experiment circle
                coslatinv = 1.0 / np.cos(np.deg2rad(lat0))

                # compute the x and y coordinates of the circle
                angles    = np.linspace(0.0, 2.0 * np.pi, numPoints)   # ,endpoint=True) # [rad]

                # Calculate the circle coordinates in lat/lon degrees.
                # Use flat-earth approximation to convert from cartesian to lat/lon.
                latCircle = lat0 + np.rad2deg(Rcircle * np.sin(angles) / Rearth)  # [deg]
                lonCircle = lon0 + np.rad2deg(Rcircle * np.cos(angles) * coslatinv / Rearth)  # [deg]

                # make the data array in the format needed to plot circle
                newdata = np.empty(2 * numPoints, dtype=np.float32)  # Create empty array
                newdata[0::2] = latCircle  # Fill array lat0,lon0,lat1,lon1....
                newdata[1::2] = lonCircle

            self.polynames[name] = (len(self.polydata), 2 * len(newdata))
            newbuf = np.empty(2 * len(newdata), dtype=np.float32)
            newbuf[0::4]   = newdata[0::2]  # lat
            newbuf[1::4]   = newdata[1::2]  # lon
            newbuf[2:-2:4] = newdata[2::2]  # lat
            newbuf[3:-3:4] = newdata[3::2]  # lon
            newbuf[-2:]    = newdata[0:2]
            self.polydata  = np.append(self.polydata, newbuf)

    def defwpt(self, name, lat, lon):
        self.custwplbl += name.ljust(5)
        self.custwplat = np.append(self.custwplat, np.float32(lat))
        self.custwplon = np.append(self.custwplon, np.float32(lon))

    def setflag(self, flag, args):
        # Switch/toggle/cycle radar screen features e.g. from SWRAD command
        if flag == 'SYM':
            # For now only toggle PZ
            self.show_pz = not self.show_pz
        # Coastlines
        elif flag == 'GEO':
            self.show_coast = not self.show_coast

        # FIR boundaries
        elif flag == 'FIR':
            self.show_fir = not self.show_fir

        # Airport: 0 = None, 1 = Large, 2= All
        elif flag == 'APT':
            self.show_apt = not self.show_apt

        # Waypoint: 0 = None, 1 = VOR, 2 = also WPT, 3 = Also terminal area wpts
        elif flag == 'VOR' or flag == 'WPT' or flag == 'WP' or flag == 'NAV':
            self.show_wpt = not self.show_wpt

        # Satellite image background on/off
        elif flag == 'SAT':
            self.show_map = not self.show_map

        # Satellite image background on/off
        elif flag == 'TRAF':
            self.show_traf = not self.show_traf

        elif flag == 'SSD':
            self.show_ssd(args)

        elif flag == 'FILTERALT':
            # First argument is an on/off flag
            if args[0]:
                self.filteralt = args[1:]
            else:
                self.filteralt = False

    def echo(self, text='', flags=0):
        if text:
            self.echo_text += ('\n' + text)

    def show_ssd(self, arg):
        if 'ALL' in arg:
            self.ssd_all      = True
            self.ssd_conflicts = False
        elif 'CONFLICTS' in arg:
            self.ssd_all      = False
            self.ssd_conflicts = True
        elif 'OFF' in arg:
            self.ssd_all      = False
            self.ssd_conflicts = False
            self.ssd_ownship = set()
        else:
            remove = self.ssd_ownship.intersection(arg)
            self.ssd_ownship = self.ssd_ownship.union(arg) - remove