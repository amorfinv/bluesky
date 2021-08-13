"""
Airspace edge information. Copies traffic, autopilot, route, activewaypoint
"""
# TODO: check if deleting routes/aircraft/waypoints works

from bluesky.tools.misc import lat2txt
import json
import numpy as np
from numpy import *
from collections import Counter

import bluesky as bs
from bluesky import core, stack, traf, scr, sim  #settings, navdb, tools
from bluesky.tools.aero import ft, kts, nm
from bluesky.tools import geo
from bluesky.core import Entity, Replaceable

streets_bool = False

def init_plugin():

    config = {
        # The name of your plugin
        'plugin_name'      : 'streets',
        'plugin_type'      : 'sim',
        # 'update_interval'  :  1.0,
        'update':          update,
        'reset':           reset
        }
    
    return config

# TODO: 
#   - add stack commands to get closest cruise/turn/free layers 

######################## UPDATE FUNCTION  ##########################

def update():
    global streets_bool

    # Only update if strets_bool is enabled
    if streets_bool:
        # Main update function for streets plugin. updates edge and flight layer tracking
        # Update edege autopilot
        edge_traffic.edgeap.update()

        # update layer tracking
        flight_layers.layer_tracking()

######################## RESET FUNCTION  ##########################
def reset():
    # when reseting bluesky turn off streets
    global streets_bool

    streets_bool = False

######################## TIMED FUNCTION  ##########################
# @core.timed_function(dt=10)
# def get_count():
#     # tells you how many aircraft in an edge group
#     edge_count_dict = dict(Counter(edge_traffic.actedge.wpedgeid))
#     group_count_dict = dict(Counter(edge_traffic.actedge.group_number))
#     # print(edge_count_dict)

######################## STACK COMMANDS ##########################
@stack.command
def addwpt2(acid: 'acid', lat: float, lon: float, alt: float = -999, spd: float = -999, wpedgeid: 'txt'="",  group_number: 'txt' = ""):
    
    """ADDWPT2 acid, (lat,lon),[alt],[spd],[edgeid],[group_number]"""
    # edgeid comes from graph
    # M2 wpt command
    # corrected arguments
    latlon = f'{lat},{lon}'
    spd *= kts
    alt *= ft

    # get group number
    group_number = edge_traffic.edge_dict[wpedgeid]['stroke_group']
    
    # get layer type
    edge_layer_type = edge_traffic.edge_dict[wpedgeid] ['layer_height']
    
    # dictionary of layers
    edge_layer_dict = flight_layers.layer_dict["config"][edge_layer_type]['levels']

    # add edge info to stack
    edge_traffic.edgeap.edge_rou[acid].addwptedgeStack(acid, latlon, alt, spd, wpedgeid, group_number, edge_layer_dict)

@stack.command
def streetsenable():
    """streetsenable"""

    # Turns on streets for scenario
    global streets_bool

    streets_bool = True

@stack.command
def edgeid(acid: 'txt'):
    """EDGEID, acid"""
    # check edge id of aircraft
    idx = traf.id2idx(acid)
    bs.scr.echo(f'{acid} flying over {edge_traffic.actedge.wpedgeid[idx]}')

@stack.command
def dis2int(acid: 'txt'):
    """dis2int acid"""
    # distance to next intersection
    idx = traf.id2idx(acid)

    current_edge = edge_traffic.actedge.wpedgeid[idx]

    node_id = int(current_edge.split("-",1)[1])

    node_lat, node_lon = osmid_to_latlon(current_edge, 1)

    _, d = geo.qdrdist(traf.lat[idx], traf.lon[idx], node_lat, node_lon)
    
    bs.scr.echo(f'{acid} is {d*nm} meters from node {node_id}')

######################## WAYPOINT TRAFFIC TRACKING  ##########################

# "traffic" class. Contains edge "autopilot" and "activedge"
class EdgeTraffic(Entity):

    def __init__(self, dict_file_path):
        super().__init__()

        with self.settrafarrays():

            self.edgeap   = EdgesAp()
            self.actedge  = ActiveEdge()

        # make variables available in bs.traf
        bs.traf.edgeap = self.edgeap
        bs.traf.actedge = self.actedge

        # initialize edge and nodes dictionaries
        
        # Opening edges.JSON as a dictionary
        with open(f'{dict_file_path}edges.json', 'r') as filename:
            self.edge_dict = json.load(filename)

        # Opening nodes.JSON as a dictionary
        with open(f'{dict_file_path}nodes.json', 'r') as filename:
            node_dict = json.load(filename)

        # reverse dictionary
        self.node_dict = {v: k for k, v in node_dict.items()}
# "autopilot"
class EdgesAp(Entity):
    def __init__(self):
        super().__init__()

        with self.settrafarrays():
            self.edge_rou = []
    
    def create(self, n=1):
        super().create(n)

        for ridx, acid in enumerate(bs.traf.id[-n:]):
            self.edge_rou[ridx - n] = Route_edge(acid)


    def update(self):
        qdr, distinnm = geo.qdrdist(bs.traf.lat, bs.traf.lon,
                                    bs.traf.actwp.lat, bs.traf.actwp.lon)  # [deg][nm])
        dist2wp = distinnm*nm  # Conversion to meters
        # print(edge_traffic.actedge.group_number)
        # See if waypoints have reached their destinations
        for i in bs.traf.actwp.Reached(qdr, dist2wp, bs.traf.actwp.flyby,
                                       bs.traf.actwp.flyturn,bs.traf.actwp.turnrad,bs.traf.actwp.swlastwp):

            # get next wpedgeid for aircraft and lat lon of next intersection
            edge_traffic.actedge.wpedgeid[i], \
            edge_traffic.actedge.intersection_lat[i] , edge_traffic.actedge.intersection_lon[i], \
            edge_traffic.actedge.group_number[i], edge_traffic.actedge.edge_layer_dict[i] \
                 = self.edge_rou[i].getnextwp()
        
        # get distance of drones to next intersection intersection
        _, dis_to_int = geo.qdrdist_matrix(traf.lat, traf.lon, edge_traffic.actedge.intersection_lat,
                                                                edge_traffic.actedge.intersection_lon)
        edge_traffic.actedge.dis_to_int = np.asarray(dis_to_int).flatten()

        # update variables available in bs.traf
        bs.traf.edgeap = edge_traffic.edgeap
        bs.traf.actedge = edge_traffic.actedge

        return
# active edge class "the active waypoint"
class ActiveEdge(Entity):
    def __init__(self):
        super().__init__()

        with self.settrafarrays():
            self.wpedgeid = np.array([], dtype=str)
            self.nextwpedgeid = np.array([], dtype=str)

            self.intersection_lat = np.array([])
            self.intersection_lon = np.array([])

            self.dis_to_int = np.array([])

            self.group_number = np.array([], dtype=int)
            self.edge_layer_dict = np.array([], dtype=object)

    
    def create(self, n=1):
        super().create(n)

        self.wpedgeid[-n:]                  = ""
        self.nextwpedgeid[-n:]              = ""

        self.intersection_lat[-n:]          = 89.99
        self.intersection_lon[-n:]          = 89.99

        self.dis_to_int[-n:]                = 9999.9

        self.group_number[-n:]              = 999
        self.edge_layer_dict[-n:]           = {}

# route_edge class. keeps track of when aircraft move to new edges and adds edges to stack
class Route_edge(Replaceable):

    def __init__(self, acid):
        # Aircraft id (callsign) of the aircraft to which this route belongs
        self.acid = acid
        self.nwp = 0

        # Waypoint data
        self.wpname = []

        # Current actual waypoint
        self.iactwp = -1
   
        # initialize edge id list. osmids of edge
        self.wpedgeid = []

        # initialize group_number
        self.group_number = []

        # initialize edge_layer_dict
        self.edge_layer_dict = []

    def addwptedgeStack(self, idx, latlon, alt, spd, wpedgeid, group_number, edge_layer_dict): 

        # send command to bluesky waypoint stack
        traf.ap.route[idx].addwptStack(idx, latlon, alt, spd)

        # Get name
        name    = bs.traf.id[idx]
        
        # Add waypoint
        wpidx = self.addwpt(idx, name, wpedgeid, group_number, edge_layer_dict)

        # Check for success by checking inserted location in flight plan >= 0
        if wpidx < 0:
            return False, "Waypoint " + name + " not added."

        # check for presence of orig/dest
        norig = int(bs.traf.ap.orig[idx] != "") # 1 if orig is present in route
        ndest = int(bs.traf.ap.dest[idx] != "") # 1 if dest is present in route

        # Check whether this is first 'real' waypoint (not orig & dest),
        # And if so, make active
        if self.nwp - norig - ndest == 1:  # first waypoint: make active
            self.direct(idx, self.wpname[norig])  # 0 if no orig

        return True

    def overwrite_wpt_data(self, wpidx, wpname, wpedgeid, group_number, edge_layer_dict):
        """
        Overwrites information for a waypoint, via addwpt_data/9
        """
        # TODO: check if it works

        self.addwpt_data(True, wpidx, wpname, wpedgeid, group_number, edge_layer_dict)

    def insert_wpt_data(self, wpidx, wpname, wpedgeid, group_number, edge_layer_dict):
        """
        Inserts information for a waypoint, via addwpt_data/9
        """
        # TODO: check if it works

        self.addwpt_data(False, wpidx, wpname, wpedgeid, group_number, edge_layer_dict)

    def addwpt_data(self, overwrt, wpidx, wpname, wpedgeid, group_number, edge_layer_dict):
        """
        Overwrites or inserts information for a waypoint
        """
        # TODO: check if it works

        if overwrt:
            self.wpname[wpidx]  = wpname
            self.wpedgeid[wpidx] = wpedgeid
            self.group_number[wpidx] = group_number
            self.edge_layer_dict[wpidx] = edge_layer_dict

        else:
            self.wpname.insert(wpidx, wpname)
            self.wpedgeid.insert(wpidx, wpedgeid)
            self.group_number.insert(wpidx, group_number)
            self.edge_layer_dict.insert(wpidx, edge_layer_dict)

    def addwpt(self, iac, name, wpedgeid ="", group_number="", edge_layer_dict =""):
        """Adds waypoint an returns index of waypoint, lat/lon [deg], alt[m]"""

        # For safety
        self.nwp = len(self.wpedgeid)

        name = name.upper().strip()

        newname = Route_edge.get_available_name(
            self.wpname, name, 3)

        wpidx = self.nwp

        self.addwpt_data(False, wpidx, newname, wpedgeid, group_number, edge_layer_dict)

        idx = wpidx
        self.nwp += 1

        return idx

    def delwpt(self,delwpname,iac=None):
        """Delete waypoint"""
        # TODO: check if it works

        # Delete complete route?
        if delwpname =="*":
            return self.delrte(iac)

        # Look up waypoint
        idx = -1
        i = len(self.wpname)
        while idx == -1 and i > 0:
            i -= 1
            if self.wpname[i].upper() == delwpname.upper():
                idx = i

        # check if active way point is the one being deleted and that it is not the last wpt.
        # If active wpt is deleted then change path of aircraft
        if self.iactwp == idx and not idx == self.nwp - 1:
            self.direct(iac, self.wpname[idx + 1])

        # Delete waypoint
        if idx == -1:
            return False, "Waypoint " + delwpname + " not found"

        self.nwp =self.nwp - 1
        del self.wpname[idx]
        del self.wpedgeid[idx]
        if self.iactwp > idx:
            self.iactwp = max(0, self.iactwp - 1)

        self.iactwp = min(self.iactwp, self.nwp - 1)

        return True

    def direct(self, idx, wpnam):
        #print("Hello from direct")
        """Set active point to a waypoint by name"""
        name = wpnam.upper().strip()
        if name != "" and self.wpname.count(name) > 0:
            wpidx = self.wpname.index(name)
            self.iactwp = wpidx

            # set edge id and intersection lon lat for actedge
            edge_traffic.actedge.wpedgeid[idx] = self.wpedgeid[wpidx]
            
            edge_traffic.actedge.intersection_lat[idx], edge_traffic.actedge.intersection_lon[idx] \
                = osmid_to_latlon(self.wpedgeid[wpidx], 1)
            
            # set group_number and edge layer_dict
            edge_traffic.actedge.group_number[idx] = self.group_number[wpidx]
            edge_traffic.actedge.edge_layer_dict[idx] = self.edge_layer_dict[wpidx]

            return True
        else:
            return False, "Waypoint " + wpnam + " not found"

    def getnextwp(self):
        
        if self.iactwp < len(self.wpedgeid) - 1:
           self.iactwp += 1

        wpedgeid = self.wpedgeid[self.iactwp]

        intersection_lat ,intersection_lon = osmid_to_latlon(wpedgeid, 1)

        # Updata group number and edge_layer_dict
        group_number = self.group_number[self.iactwp]
        edge_layer_dict = self.edge_layer_dict[self.iactwp]

        return wpedgeid, intersection_lat, intersection_lon, group_number, edge_layer_dict
    
    def delrte(self,iac=None):
        """Delete complete route"""
        # Simple re-initialize this route as empty
        self.__init__(edge_traffic.id[iac])

        return True

    @staticmethod
    def get_available_name(data, name_, len_=2):
        """
        Check if name already exists, if so add integer 01, 02, 03 etc.
        """
        appi = 0  # appended integer to name starts at zero (=nothing)
        # Use Python 3 formatting syntax: "{:03d}".format(7) => "007"
        fmt_ = "{:0" + str(len_) + "d}"

        # Avoid using call sign without number
        if bs.traf.id.count(name_) > 0:
            appi = 1
            name_ = name_+fmt_.format(appi)

        while data.count(name_) > 0 :
            appi += 1
            name_ = name_[:-len_]+fmt_.format(appi)
        return name_

def osmid_to_latlon(osmid , i=2):

    # input an edge and get the lat lon of one of the nodes
    # i = 0 gets nodeid of first node of edges
    # i = 1 gets nodeid of second node of edge

    if not i == 2:
        # if given an edge
        node_id = int(osmid.split("-",1)[i])
    else:
        # if given a node
        node_id = int(osmid)

    node_latlon = edge_traffic.node_dict[node_id]

    node_lat = float(node_latlon.split("-",1)[0])
    node_lon = float(node_latlon.split("-",1)[1])

    return node_lat, node_lon

######################## FLIGHT LAYER TRACKING ############################
class FlightLayers(Entity):
    def __init__(self, dict_file_path):
        super().__init__()

        with self.settrafarrays():
            self.flight_levels                  = np.array([], dtype=int)
            self.flight_layer_type               = np.array([], dtype=str)

            self.closest_cruise_layer_bottom    = np.array([], dtype=int)
            self.closest_cruise_layer_top       = np.array([], dtype=int)

            self.closest_turn_layer_bottom      = np.array([], dtype=int)
            self.closest_turn_layer_top         = np.array([], dtype=int)

        # Assign info to bs.trafic
        bs.traf.flight_levels = self.flight_levels
        bs.traf.flight_layer_type = self.flight_layer_type
        bs.traf.closest_cruise_layer_bottom = self.closest_cruise_layer_bottom
        bs.traf.closest_cruise_layer_top = self.closest_cruise_layer_top
        bs.traf.closest_turn_layer_bottom = self.closest_turn_layer_bottom
        bs.traf.closest_turn_layer_top = self.closest_turn_layer_top

        self.layer_dict = {}
        # Opening edges.JSON as a dictionary
        with open(f'{dict_file_path}layers.json', 'r') as filename:
            self.layer_dict = json.load(filename)
        
        self.layer_spacing = self.layer_dict['info']['spacing']
        self.layer_levels = np.array(self.layer_dict['info']['levels'])
        self.layer_ranges = self.layer_levels - self.layer_dict['info']['spacing']/2

        # vectorize function to extract layer_info
        self.layer_info = np.vectorize(self.get_layer_type)
    
    def create(self, n=1):
        super().create(n)

        self.flight_levels[-n:]                 = 0
        self.flight_layer_type[-n:]             = ""

        self.closest_cruise_layer_top[-n:]      = 0
        self.closest_cruise_layer_bottom[-n:]   = 0

        self.closest_turn_layer_top[-n:]        = 0
        self.closest_turn_layer_bottom[-n:]     = 0

    def layer_tracking(self):
        # update flight levels
        self.flight_levels = np.array((np.round((bs.traf.alt/ft) / self.layer_spacing))*self.layer_spacing, dtype=int)

        # update flight layer type
        edge_layer_dicts = edge_traffic.actedge.edge_layer_dict

        # only go into vectorized function if there is traffic. otherwise it fails
        if bs.traf.ntraf > 0:
            self.flight_layer_type, self.closest_cruise_layer_bottom, self.closest_cruise_layer_top, \
                self.closest_turn_layer_bottom, self.closest_turn_layer_top = self.layer_info(self.flight_levels, edge_layer_dicts)
        else:
            self.flight_layer_type, self.closest_cruise_layer_bottom, self.closest_cruise_layer_top, \
                self.closest_turn_layer_bottom, self.closest_turn_layer_top = "", "", "", "", ""
        # update bs.traf
        bs.traf.flight_levels = self.flight_levels
        bs.traf.flight_layer_type = self.flight_layer_type
        bs.traf.closest_cruise_layer_bottom = self.closest_cruise_layer_bottom
        bs.traf.closest_cruise_layer_top = self.closest_cruise_layer_top
        bs.traf.closest_turn_layer_bottom = self.closest_turn_layer_bottom
        bs.traf.closest_turn_layer_top = self.closest_turn_layer_top

        return

    @staticmethod
    def get_layer_type(flight_level, edge_layer_dict):

        # vectorized function to process edge layer dictionary
        # get correct layer_info_list based on your flight level
        layer_list = edge_layer_dict[f'{flight_level}']
        
        # get layer_type
        layer_type = layer_list[0]

        # get closest cruise layers
        closest_cruise_layer_bottom = layer_list[1]
        closest_cruise_layer_top = layer_list[2]

        if closest_cruise_layer_bottom == '':
            closest_cruise_layer_bottom = 0

        if closest_cruise_layer_top =='':
            closest_cruise_layer_top = 0
        
        # get closest turnlayers
        closest_turn_layer_bottom = layer_list[3]
        closest_turn_layer_top = layer_list[4]

        if closest_turn_layer_bottom == '':
            closest_turn_layer_bottom = 0

        if closest_turn_layer_top =='':
            closest_turn_layer_top = 0


        return layer_type, closest_cruise_layer_bottom, closest_cruise_layer_top, \
            closest_turn_layer_bottom, closest_turn_layer_top


######################## EDGE AND FLIGHT LAYER INITIALIZATION##########################

# load street_graphs
dict_file_path = 'plugins/streets/'

# Initialize EdgeTraffic class
edge_traffic = EdgeTraffic(dict_file_path)

# # Initialize Flight Layers
flight_layers = FlightLayers(dict_file_path)

