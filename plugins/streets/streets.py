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

def init_plugin():

    config = {
        # The name of your plugin
        'plugin_name'      : 'streets',
        'plugin_type'      : 'sim',
        # 'update_interval'  :  1.0,
        'update':          update

        }

    return config

######################## UPDATE FUNCTION  ##########################

def update():
    # Update edege autopilot
    edge_traffic.edgeap.update()

    # update variables available in bs.traf
    bs.traf.edgeap = edge_traffic.edgeap
    bs.traf.actedge = edge_traffic.actedge

    # get distance of drones to next intersection/turn intersection
    _, dis_to_int = geo.qdrdist_matrix(traf.lat, traf.lon, edge_traffic.actedge.intersection_lat, edge_traffic.actedge.intersection_lon)
    _, dis_to_turn = geo.qdrdist_matrix(traf.lat, traf.lon, edge_traffic.actedge.turn_intersection_lat, edge_traffic.actedge.turn_intersection_lon)

######################## TIMED FUNCTION  ##########################
@core.timed_function(dt=30)
def do_flowcontrol():
    # tells you how many aircraft in an edge
    # TODO: perhaps only useful for stroke_groups
    edge_count_dict = dict(Counter(edge_traffic.actedge.wpedgeid))
    #print(edge_count_dict)

######################## STACK COMMANDS ##########################
@stack.command
def addwptm2(acid: 'acid', lat: float, lon: float, alt: float, spd: float, wpedgeid: 'txt',  turn_node: 'txt' = ""):
    """ADDWPTM2 acid, (lat,lon),[alt],[spd],[edgeid],[turn_node]"""
    # edgeid comes from graph
    # M2 wpt command
    # corrected arguments
    latlon = f'{lat},{lon}'
    spd *= kts
    alt *= ft

    # add edge info to stack
    edge_traffic.edgeap.edge_rou[acid].addwptedgeStack(acid, latlon, alt, spd, wpedgeid, turn_node)


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

    def __init__(self):
        super().__init__()

        with self.settrafarrays():

            self.edgeap   = EdgesAp()
            self.actedge  = ActiveEdge()

        # make variables available in bs.traf
        bs.traf.edgeap = self.edgeap
        bs.traf.actedge = self.actedge

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

        # See if waypoints have reached their destinations
        for i in bs.traf.actwp.Reached(qdr, dist2wp, bs.traf.actwp.flyby,
                                       bs.traf.actwp.flyturn,bs.traf.actwp.turnrad,bs.traf.actwp.swlastwp):

            # get next wpedgeid for aircraft and lat lon of next intersection/turn
            edge_traffic.actedge.wpedgeid[i], edge_traffic.actedge.nextturnnode[i], \
            edge_traffic.actedge.intersection_lat[i] , edge_traffic.actedge.intersection_lon[i], \
            edge_traffic.actedge.turn_intersection_lat[i], edge_traffic.actedge.turn_intersection_lon[i] \
                     = self.edge_rou[i].getnextwp()
        return

# active edge class "the active waypoint"
class ActiveEdge(Entity):
    def __init__(self):
        super().__init__()

        with self.settrafarrays():
            self.wpedgeid = np.array([], dtype=str)
            self.nextwpedgeid = np.array([], dtype=str)
            self.nextturnnode = np.array([], dtype=str)

            self.intersection_lat = np.array([])
            self.intersection_lon = np.array([])
            self.turn_intersection_lat = np.array([])
            self.turn_intersection_lon = np.array([])

    
    def create(self, n=1):
        super().create(n)

        self.wpedgeid[-n:]                  = ""
        self.nextwpedgeid[-n:]              = ""
        self.nextturnnode[-n:]              = ""

        self.intersection_lat[-n:]          = 89.99
        self.intersection_lon[-n:]          = 89.99
        self.turn_intersection_lat[-n:]     = 89.99
        self.turn_intersection_lon[-n:]     = 89.99

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

        # initialize turn_node
        self.turn_node = []

    def addwptedgeStack(self, idx, latlon, alt, spd, wpedgeid, turn_node):  # args: all arguments of addwpt
        """ADDWPT acid, (wpname/lat,lon),[alt],[spd],[wpedgeid]"""

        # send command to bluesky waypoint stack
        traf.ap.route[idx].addwptStack(idx, latlon, alt, spd)

        # Get name
        name    = bs.traf.id[idx]
        
        # Add waypoint
        wpidx = self.addwpt(idx, name, wpedgeid, turn_node)

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

    def overwrite_wpt_data(self, wpidx, wpname, wpedgeid, turn_node):
        """
        Overwrites information for a waypoint, via addwpt_data/9
        """
        # TODO: check if it works

        self.addwpt_data(True, wpidx, wpname, wpedgeid, turn_node)

    def insert_wpt_data(self, wpidx, wpname, wpedgeid, turn_node):
        """
        Inserts information for a waypoint, via addwpt_data/9
        """
        # TODO: check if it works

        self.addwpt_data(False, wpidx, wpname, wpedgeid, turn_node)

    def addwpt_data(self, overwrt, wpidx, wpname, wpedgeid, turn_node):
        """
        Overwrites or inserts information for a waypoint
        """
        # TODO: check if it works

        if overwrt:
            self.wpname[wpidx]  = wpname
            self.wpedgeid[wpidx] = wpedgeid
            self.turn_node[wpidx] = turn_node

        else:
            self.wpname.insert(wpidx, wpname)
            self.wpedgeid.insert(wpidx, wpedgeid)
            self.turn_node.insert(wpidx, turn_node)

    def addwpt(self, iac, name, wpedgeid ="", turn_node=""):
        """Adds waypoint an returns index of waypoint, lat/lon [deg], alt[m]"""

        # For safety
        self.nwp = len(self.wpedgeid)

        name = name.upper().strip()

        newname = Route_edge.get_available_name(
            self.wpname, name, 3)

        wpidx = self.nwp

        self.addwpt_data(False, wpidx, newname, wpedgeid, turn_node)

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

            # set edge id and intersection/turn lon lat for actedge
            edge_traffic.actedge.wpedgeid[idx] = self.wpedgeid[wpidx]
            
            edge_traffic.actedge.nextturnnode[idx] = self.turn_node[wpidx]

            edge_traffic.actedge.intersection_lat[idx], edge_traffic.actedge.intersection_lon[idx] \
                = osmid_to_latlon(self.wpedgeid[wpidx], 1)

            edge_traffic.actedge.turn_intersection_lat[idx], edge_traffic.actedge.turn_intersection_lon[idx] \
                = osmid_to_latlon(self.turn_node[wpidx])    

            return True
        else:
            return False, "Waypoint " + wpnam + " not found"

    def getnextwp(self):
        
        if self.iactwp < len(self.wpedgeid) - 1:
           self.iactwp += 1

        wpedgeid = self.wpedgeid[self.iactwp]
        turn_node = self.turn_node[self.iactwp]

        intersection_lat ,intersection_lon = osmid_to_latlon(wpedgeid, 1)

        # only update turn data if aircraft will turn
        if len(turn_node):
            turn_intersection_lat ,turn_intersection_lon = osmid_to_latlon(turn_node)
        else:
            turn_intersection_lat = 0
            turn_intersection_lon = 0

        return wpedgeid, turn_node, intersection_lat, intersection_lon, turn_intersection_lat, turn_intersection_lon
    
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

######################## OTHER EDGE PLUGIN CODE  ##########################

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

    node_latlon = node_dict[node_id]

    node_lat = float(node_latlon.split("-",1)[0])
    node_lon = float(node_latlon.split("-",1)[1])

    return node_lat, node_lon

# load street_graphs
file_path = 'plugins/streets/'

# Opening edges.JSON as a dictionary
with open(f'{file_path}edges.json', 'r') as filename:
    edge_dict = json.load(filename)

# Opening edges.JSON as a dictionary
with open(f'{file_path}nodes.json', 'r') as filename:
    node_dict = json.load(filename)

# reverse traffic
node_dict = {v: k for k, v in node_dict.items()}

# Initialize EdgeTraffic class
edge_traffic = EdgeTraffic()

######################### FLOW CONTROL ##################

import dill
import networkx as nx

from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import PathPlanning
class PathPlans(Entity):
    def __init__(self):
        super().__init__()
        self.getGraph()
        self.graph = street_graph(self.G, self.edges)
        with self.settrafarrays():
            self.pathplanning = []
            
    # def create(self, n = 1):
    #     print(n)
    #     super().create(n)
    #     traf = bs.traf
    #     lat1 = traf.ap.route[-n].wplat[0]
    #     lon1 = traf.ap.route[-n].wplon[0]
    #     lat2 = traf.ap.route[-n].wplat[-1]
    #     lon2 = traf.ap.route[-n].wplon[-1]
    #     self.pathplanning[-n:] = PathPlanning(self.graph,lon1,lat1,lon2,lat2) 
        
    def getGraph(self):
        self.G = dill.load(open("plugins/streets/G-multigraph.dill", "rb"))
        # self.edge = dill.load(open("plugins/streets/edge_gdf.dill", "rb"))
        self.nodes, self.edges = graph_to_dfs(self.G)


def graph_to_dfs(G):
    """
    Adapted from osmnx code: https://github.com/gboeing/osmnx

    Convert a MultiDiGraph to node and edge DataFrames.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph

    Returns
    -------
    pandas.DataFrame or tuple
        gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges). gdf_nodes
        is indexed by osmid and gdf_edges is multi-indexed by u, v, key
        following normal MultiDiGraph structure.
    """
    import pandas as pd
    from shapely.geometry import LineString, Point
    import networkx as nx

    # create node dataframe
    nodes, data = zip(*G.nodes(data=True))

    # convert node x/y attributes to Points for geometry column
    geom = (Point(d["x"], d["y"]) for d in data)
    df_nodes = pd.DataFrame(data, index=nodes)
    df_nodes['geometry'] = list(geom)

    df_nodes.index.rename("osmid", inplace=True)

    # create edge dataframe 
    u, v, k, data = zip(*G.edges(keys=True, data=True))

    # subroutine to get geometry for every edge: if edge already has
    # geometry return it, otherwise create it using the incident nodes
    x_lookup = nx.get_node_attributes(G, "x")
    y_lookup = nx.get_node_attributes(G, "y")

    def make_geom(u, v, data, x=x_lookup, y=y_lookup):
        if "geometry" in data:
            return data["geometry"]
        else:
            return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))

    geom = map(make_geom, u, v, data)
    df_edges = pd.DataFrame(data)
    df_edges['geometry'] = list(geom)

    # add u, v, key attributes as index
    df_edges["u"] = u
    df_edges["v"] = v
    df_edges["key"] = k
    df_edges.set_index(["u", "v", "key"], inplace=True)

    return df_nodes, df_edges

# Initialize Path Plans
path_plans = PathPlans()

