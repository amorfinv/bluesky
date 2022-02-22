import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
import networkx as nx
import pandas as pd

import bluesky as bs
from bluesky.tools import geo

def get_lat_lon_from_osm_route(G, route):
    """
    Get lat and lon from an osmnx route (list of nodes) and nx.MultGraph.
    The function returns two numpy arrays with the lat and lon of route.
    Also return a GeoDataFrame with the lat and lon of the route as a
    linestring.
    Parameters
    ----------
    G : nx.MultiGraph
        Graph to get lat and lon from. Graph should be built
        with osmnx.get_undirected.
    route : list
        List of nodes to build edge and to get lat lon from
    Returns
    -------
    lat : numpy.ndarray
        Array with latitudes of route
    lon : numpy.ndarray
        Array with longitudes of route
    route_gdf : geopandas.GeoDataFrame
        GeoDataFrame with lat and lon of route as a linestring.
    """
    # add first node to route
    lons = np.array(G.nodes[route[0]]["x"])
    lats = np.array(G.nodes[route[0]]["y"])

    # loop through the rest for loop only adds from second point of edge
    for u, v in zip(route[:-1], route[1:]):
        # if there are parallel edges, select the shortest in length
        data = list(G.get_edge_data(u, v).values())[0]

        # extract coords from linestring
        xs, ys = data["geometry"].xy

        # Check if geometry of edge is in correct order
        if G.nodes[u]["x"] != data["geometry"].coords[0][0]:

            # flip if in wrong order
            xs = xs[::-1]
            ys = ys[::-1]

        # only add from the second point of linestring
        lons = np.append(lons, xs[1:])
        lats = np.append(lats, ys[1:])

    # make a linestring from the coords
    linestring = LineString(zip(lons, lats))

    # make into a gdf
    line_gdf = gpd.GeoDataFrame(geometry=[linestring], crs="epsg:4326")

    return lats, lons, line_gdf


def get_turn_arrays(lats, lons, cutoff_angle=25):
    """
    Get turn arrays from latitude and longitude arrays.
    The function returns three arrays with the turn boolean, turn speed and turn coordinates.
    Turn speed depends on the turn angle.
        - Speed set to 0 for no turns.
        - Speed is 10 knots for angles between 25 and 100 degrees.
        - Speed is 5 knots for angles between 100 and 150 degrees.
        - Speed is 2 knots for angles larger than 150 degrees.
    Parameters
    ----------
    lat : numpy.ndarray
        Array with latitudes of route
    lon : numpy.ndarray
        Array with longitudes of route
    cutoff_angle : int
        Cutoff angle for turning. Default is 25.
    Returns
    -------
    turn_bool : numpy.ndarray
        Array with boolean values for turns.
    turn_speed : numpy.ndarray
        Array with turn speed. If no turn, speed is 0.
    turn_coords : numpy.ndarray
        Array with turn coordinates. If no turn then it has (-9999.9, -9999.9)
    """

    # Define empty arrays that are same size as lat and lon
    turn_speed = np.zeros(len(lats))
    turn_bool = np.array([False] * len(lats), dtype=np.bool8)
    turn_coords = np.array([(-9999.9, -9999.9)] * len(lats), dtype="f,f")

    # Initialize variables for the loop
    lat_prev = lats[0]
    lon_prev = lons[0]

    # loop thru the points to calculate turn angles
    for i in range(1, len(lats) - 1):
        # reset some values for the loop
        lat_cur = lats[i]
        lon_cur = lons[i]
        lat_next = lats[i + 1]
        lon_next = lons[i + 1]

        # calculate angle between points
        d1 = geo.qdrdist(lat_prev, lon_prev, lat_cur, lon_cur)
        d2 = geo.qdrdist(lat_cur, lon_cur, lat_next, lon_next)

        # fix angles that are larger than 180 degrees
        angle = abs(d2 - d1)
        angle = 360 - angle if angle > 180 else angle

        # give the turn speeds based on the angle
        if angle > cutoff_angle and i != 0:

            # set turn bool to true and get the turn coordinates
            turn_bool[i] = True
            turn_coords[i] = (lat_cur, lon_cur)

            # calculate the turn speed based on the angle.
            if angle < 100:
                turn_speed[i] = 10
            elif angle < 150:
                turn_speed[i] = 5
            else:
                turn_speed[i] = 2
        else:
            turn_coords[i] = (-9999.9, -9999.9)

        # update the previous values at the end of the loop
        lat_prev = lat_cur
        lon_prev = lon_cur

    # make first entry to turn bool true (entering constrained airspace)
    turn_bool[0] = True

    return turn_bool, turn_speed, turn_coords

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
    pandas.GeoDataFrame tuple
        gdf_nodes and gdf_edges
    """

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
    df_edges.set_index(["u", "v"], inplace=True)

    # convert to geodataframe
    gdf_nodes = gpd.GeoDataFrame(df_nodes, geometry="geometry", crs="epsg:4326")
    gdf_edges = gpd.GeoDataFrame(df_edges, geometry="geometry", crs="epsg:4326")

    return gdf_nodes, gdf_edges