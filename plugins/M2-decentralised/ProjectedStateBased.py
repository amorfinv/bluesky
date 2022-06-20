''' State-based conflict detection. '''
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, split
import geopandas as gpd
# from rich import inspect
from shapely.affinity  import affine_transform, scale
from bluesky import stack
import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.aero import nm
from bluesky.traffic.asas import ConflictDetection
from time import time
from copy import deepcopy

def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'PROJECTEDSTATEBASED',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config


class M2StateBased(ConflictDetection):
    def __init__(self):
        super().__init__()
        self.dist_mat = np.array([])
        self.qdr_mat = np.array([])
        self.rpz_actual = 32 #m
        self.rpz_buffered = 40 #m
        return
        
    def clearconfdb(self):
        ''' Clear conflict database. '''
        self.confpairs_unique.clear()
        self.lospairs_unique.clear()
        self.confpairs.clear()
        self.lospairs.clear()
        self.qdr = np.array([])
        self.dist = np.array([])
        self.dcpa = np.array([])
        self.tcpa = np.array([])
        self.tLOS = np.array([])
        self.inconf = np.zeros(bs.traf.ntraf)
        self.tcpamax = np.zeros(bs.traf.ntraf)
        self.dist_mat = np.array([])
        self.qdr_mat = np.array([])
        return
        
    def update(self, ownship, intruder):
        ''' Perform an update step of the Conflict Detection implementation. '''
        self.confpairs, self.lospairs, self.inconf, self.tcpamax, self.qdr, \
            self.dist, self.dcpa, self.tcpa, self.tLOS, self.qdr_mat, self.dist_mat = \
                self.detect(ownship, intruder, self.rpz, self.hpz, self.dtlookahead)

        # confpairs has conflicts observed from both sides (a, b) and (b, a)
        # confpairs_unique keeps only one of these
        confpairs_unique = {frozenset(pair) for pair in self.confpairs}
        lospairs_unique = {frozenset(pair) for pair in self.lospairs}

        self.confpairs_all.extend(confpairs_unique - self.confpairs_unique)
        self.lospairs_all.extend(lospairs_unique - self.lospairs_unique)

        # Update confpairs_unique and lospairs_unique
        self.confpairs_unique = confpairs_unique
        self.lospairs_unique = lospairs_unique    
        
    def detect(self, ownship, intruder, rpz, hpz, dtlookahead):
        ''' Conflict detection between ownship (traf) and intruder (traf/adsb).'''

        ############### START PROJECTION ########################

        # make a deep copy of intruder coordinates and trk
        intruderlat = deepcopy(intruder.lat)
        intruderlon = deepcopy(intruder.lon)
        intrudertrk = deepcopy(intruder.trk)

        # t1 = time()
        # here find the position along route of ownship
        routes = ownship.ap.route
        
        # intialize the geo_dict
        geo_dict = {'geometry': [], 'acid': []}

        if ownship.ntraf >= 1:

            # for loop through aircraft id TODO: vectorize
            for idx, route in enumerate(routes):
                
                if not route.wplat:
                    continue

                # get the current location
                current_loc = gpd.GeoSeries(Point([ownship.lon[idx], ownship.lat[idx]]), crs='epsg:4326')

                # get the lookahead distance
                look_ahead_dist = ownship.selspd[idx] * dtlookahead[idx]
                
                # convert to utm
                current_loc = current_loc.to_crs(epsg=32633)

                # add lon lat to shapely linestring
                route_line = gpd.GeoSeries(LineString(zip(route.wplon, route.wplat)), crs='epsg:4326')
                route_line = route_line.to_crs(epsg=32633)

                # find closest point to linestring
                p1, _ = nearest_points(route_line.geometry.values[0], current_loc.geometry.values[0])

                # now split the line to remove eveything before current position
                _, end_line = split_line_with_point(route_line.geometry.values[0], p1)

                # now interpolate along this line
                look_ahead_dist = look_ahead_dist + rpz[0]
                look_ahead_dist = 100 if look_ahead_dist < 100 else look_ahead_dist
                end_point = end_line.interpolate(look_ahead_dist)

                # now split line again to get line with a lookahead tine
                look_ahead_line, _ = split_line_with_point(end_line, end_point)

                # fill the geo_dict
                geo_dict['geometry'].append(look_ahead_line)
                geo_dict['acid'].append(ownship.id[idx])

            # t2 = time()
            # print('Time to extrapolate: ', t2-t1)

            # t1 = time()
            # create geopandas geoseries
            geo_series = gpd.GeoSeries(geo_dict['geometry'], crs='epsg:32633', index=geo_dict['acid'])

            # check if look_ahead_lines_intersect
            own_inter, int_inter = geo_series.sindex.query_bulk(geo_series, predicate="intersects")

            # note that all intersect with themselves so you must check if there are any unique intersections
            # This happens when there are more intersections than aircraft
            actual_intersections = []
            if len(own_inter) > ownship.ntraf:

                # Get all unique intersections since there are more intersections than aircraft
                # Also because query_bulk returns also self-intersections
                uniq_arr, counts = np.unique(own_inter, axis=0, return_counts=True)

                # select indices from own_inter that correspond to unique values with a count greater than 1
                potential_intersections = np.arange(len(own_inter))[~np.in1d(own_inter, uniq_arr[counts == 1])]

                # stack the ownship and intruder intersection vertically (nx2) array
                own_int_array = np.column_stack((own_inter[potential_intersections], int_inter[potential_intersections]))

                # check rows and if the columns are equal delete that row
                actual_intersections = own_int_array[own_int_array[:,0] != own_int_array[:,1]]

            # t2 = time()
            # print("Time to check intersection: ", t2-t1)

            # t3 = time()

            # if they intersect rebuild intruder.lat and intuder.lon, intruder.trk so that state based works normally
            # TODO: fix all of the angle calculations and vectorize the for loop
            for intersection in actual_intersections:

                print('THERE IS AN INTERSECTION')

                curr_ownship = intersection[0]
                ownship_id = ownship.id[curr_ownship]

                curr_intruder = intersection[1]
                intruder_id = intruder.id[curr_intruder]

                # find intersection point between ownship and intruder using geo_series
                own_line = geo_series[ownship_id]
                int_line = geo_series[intruder_id]

                # get the intersection point
                intersection_point = own_line.intersection(int_line)

                # now split ownship and intruder lines with interseciton point
                own_cut_line, _ = split_line_with_point(own_line, intersection_point)
                int_cut_line, _ = split_line_with_point(int_line, intersection_point)
                
                # now make a straight line in the direction of ownship.trk that is same length
                # as the length of own_cut_line.. don't use ownship trk..interpolate line from second point of ownship 
                # intersecting line
                length_own_line = own_cut_line.length
                p1 = Point([own_cut_line.xy[0][0], own_cut_line.xy[1][0]])
                p2 = Point([own_cut_line.xy[0][1], own_cut_line.xy[1][1]])
                line_to_scale = LineString([p1, p2])
                line_to_scale_len = line_to_scale.length
                scale_factor = length_own_line/line_to_scale_len
                scaled_ownship_line = scale(line_to_scale, xfact=scale_factor, yfact=scale_factor, origin=p1)
                
                # get projected intersection point
                inter_x = scaled_ownship_line.xy[0][1]
                inter_y = scaled_ownship_line.xy[1][1]

                # now project the intruder
                # find the length of intruder line
                length_int_line = int_cut_line.length
                p1 = Point([int_cut_line.xy[0][-1], int_cut_line.xy[1][-1]])
                p2 = Point([int_cut_line.xy[0][-2], int_cut_line.xy[1][-2]])

                line_to_scale = LineString([p1, p2])
                line_to_scale_len = line_to_scale.length
                scale_factor = length_int_line/line_to_scale_len
                scaled_intruder_line = scale(line_to_scale, xfact=scale_factor, yfact=scale_factor, origin=p1)

                # now move this line to end at projected point of intersection (merge with above?)
                scaled_intruder_line = scale(scaled_intruder_line, origin=(inter_x, inter_y))

                # projected location of intruder
                int_x = scaled_intruder_line.xy[0][-1]
                int_y = scaled_intruder_line.xy[1][-1]

                intersectin_fun4 = gpd.GeoSeries([
                                                Point(own_cut_line.xy[0][0], own_cut_line.xy[1][0]), # current point of ownship
                                                Point(int_cut_line.xy[0][0], int_cut_line.xy[1][0]), # current point of intruder
                                                own_cut_line,  # ownship line from real own pos to real intersection
                                                int_cut_line,   # int line from real int pos to real intersection
                                                intersection_point, # real intersection point
                                                Point(inter_x, inter_y),  # projected point of intersection
                                                scaled_ownship_line, # projected line of ownship
                                                #scaled_intruder_line, # projected line of intruder
                                                #Point(int_x, int_y),  # projected point of intruder
                                                ], crs='epsg:32633')

                # funny stuff happening from second part of for loop check
                # convert to lat lon from utm
                new_point = gpd.GeoSeries(Point([int_x, int_y]), crs='epsg:32633')
                new_point = new_point.to_crs(epsg=4326)

                # assign intruder.lat and intruder.lon with int_x and int_y
                intruderlat[curr_intruder] = new_point.y
                intruderlon[curr_intruder] = new_point.x

                # TODO: override trk of ownship with scaled_ownship_line_track
                # TODO: override trk of intruder with scaled_intruder_line_track


            # t4 = time()
            # print("Time to project positions: ", t4-t3)

            # t3 = time()

        ############### END PROJECTION ########################
        # Calculate everything using the buffered RPZ
        rpz = np.zeros(len(rpz)) + self.rpz_buffered
        # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
        I = np.eye(ownship.ntraf)

        # Horizontal conflict ------------------------------------------------------

        # qdrlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
        qdr, dist = geo.kwikqdrdist_matrix(np.asmatrix(ownship.lat), np.asmatrix(ownship.lon),
                                    np.asmatrix(intruderlat), np.asmatrix(intruderlon))

        # Convert back to array to allow element-wise array multiplications later on
        # Convert to meters and add large value to own/own pairs
        qdr = np.asarray(qdr)
        dist = np.asarray(dist) * nm + 1e9 * I

        # Calculate horizontal closest point of approach (CPA)
        qdrrad = np.radians(qdr)
        dx = dist * np.sin(qdrrad)  # is pos j rel to i
        dy = dist * np.cos(qdrrad)  # is pos j rel to i

        # Ownship track angle and speed
        owntrkrad = np.radians(ownship.trk)
        ownu = ownship.gs * np.sin(owntrkrad).reshape((1, ownship.ntraf))  # m/s
        ownv = ownship.gs * np.cos(owntrkrad).reshape((1, ownship.ntraf))  # m/s

        # Intruder track angle and speed
        inttrkrad = np.radians(intrudertrk)
        intu = intruder.gs * np.sin(inttrkrad).reshape((1, ownship.ntraf))  # m/s
        intv = intruder.gs * np.cos(inttrkrad).reshape((1, ownship.ntraf))  # m/s

        du = ownu - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
        dv = ownv - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

        dv2 = du * du + dv * dv
        dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
        vrel = np.sqrt(dv2)

        tcpa = -(du * dx + dv * dy) / dv2 + 1e9 * I

        # Calculate distance^2 at CPA (minimum distance^2)
        dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

        # Check for horizontal conflict
        R2 = rpz * rpz
        swhorconf = dcpa2 < R2  # conflict or not

        # Calculate times of entering and leaving horizontal conflict
        dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
        dtinhor = dxinhor / vrel

        tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf
        touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)  # set very large if no conf

        # Vertical conflict --------------------------------------------------------

        # Vertical crossing of disk (-dh,+dh)
        dalt = ownship.alt.reshape((1, ownship.ntraf)) - \
            intruder.alt.reshape((1, ownship.ntraf)).T  + 1e9 * I

        dvs = ownship.vs.reshape(1, ownship.ntraf) - \
            intruder.vs.reshape(1, ownship.ntraf).T
        dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero

        # Check for passing through each others zone
        tcrosshi = (dalt + hpz) / -dvs
        tcrosslo = (dalt - hpz) / -dvs
        tinver = np.minimum(tcrosshi, tcrosslo)
        toutver = np.maximum(tcrosshi, tcrosslo)

        # Combine vertical and horizontal conflict----------------------------------
        tinconf = np.maximum(tinver, tinhor)
        toutconf = np.minimum(toutver, touthor)

        swconfl = np.array(swhorconf * (tinconf <= toutconf) * (toutconf > 0.0) * \
            (tinconf < dtlookahead) * (1.0 - I), dtype=np.bool)

        # --------------------------------------------------------------------------
        # Update conflict lists
        # --------------------------------------------------------------------------
        # Ownship conflict flag and max tCPA
        inconf = np.any(swconfl, 1)
        tcpamax = np.max(tcpa * swconfl, 1)

        # Select conflicting pairs: each a/c gets their own record
        confpairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swconfl))]
        # It's a LOS if the actual RPZ of 32m is violated.
        swlos = (dist < (np.zeros(len(rpz)) + self.rpz_actual)) * (np.abs(dalt) < hpz)
        lospairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swlos))]

        t4 = time()
        # print("Time to calculate: ", t4-t3)
        print('---------------------------------------------------------------------------------------------------------------------')

        return confpairs, lospairs, inconf, tcpamax, \
            qdr[swconfl], dist[swconfl], np.sqrt(dcpa2[swconfl]), \
                tcpa[swconfl], tinconf[swconfl], qdr, dist



def split_line_with_point(line, splitter):
    """Split a LineString with a Point
    Code borrowed from shapely
    """

    # point is on line, get the distance from the first point on line
    distance_on_line = line.project(splitter)

    if distance_on_line == 0:
        return line, line
    

    coords = list(line.coords)
    # split the line at the point and create two new lines
    current_position = 0.0
    for i in range(len(coords)-1):
        point1 = coords[i]
        point2 = coords[i+1]
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        segment_length = (dx ** 2 + dy ** 2) ** 0.5
        current_position += segment_length
        if distance_on_line == current_position:
            # splitter is exactly on a vertex
            return LineString(coords[:i+2]), LineString(coords[i+1:])
        elif distance_on_line < current_position:
            # splitter is between two vertices
            return LineString(coords[:i+1] + [splitter.coords[0]]), LineString([splitter.coords[0]] + coords[i+1:])
