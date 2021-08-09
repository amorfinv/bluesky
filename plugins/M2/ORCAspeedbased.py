from threading import ThreadError
from bluesky.traffic.asas import ConflictResolution
import bluesky as bs
import numpy as np
from bluesky import core
from bluesky import stack
from bluesky.traffic.asas import ConflictResolution
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union, nearest_points
from shapely.affinity import translate
from bluesky.tools.geo import kwikdist, kwikqdrdist, latlondist, qdrdist
from bluesky.tools.aero import nm, ft, kts
import bluesky as bs
import numpy as np
import itertools

def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'ORCASPEEDBASED',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config

class ORCASpeedBased(ConflictResolution): 
    # Define some variables
    def __init__(self):
        super().__init__()
        self.turn_speed = 10*kts
        
    def resolve(self, conf, ownship, intruder):
        '''We want to only solve in the velocity direction while still following the heading
        given by the autopilot. For each ownship, it calculates the minimum or maximum velocity
        needed to avoid all intruders. It then applies the solution closest to the current velocity.
        If there is no solution, it should then apply speed 0 by default, and the aircraft stops.'''
             
        # Make a copy of traffic data, track and ground speed
        newgscapped = np.copy(ownship.gs)
        newvs       = np.copy(ownship.vs)
        
        # Iterate over aircraft in conflict
        for idx in list(itertools.compress(range(len(bs.traf.cr.active)), bs.traf.cr.active)):
            # Find the pairs in which IDX is involved in a conflict
            idx_pairs = self.pairs(conf, ownship, intruder, idx)
            
            # Find solution for aircraft 'idx'
            gs_new, vs_new = self.ORCASpeedBased(conf, ownship, intruder, idx, idx_pairs)
            
            # Write the new velocity of aircraft 'idx' to traffic data
            newgscapped[idx] = gs_new    
            newvs[idx] = vs_new    
        
        # Speed based, and 2D, for now.
        alt            = ownship.ap.alt 
        newtrack       = ownship.ap.trk
        
        return newtrack, newgscapped, newvs, alt


    def ORCASpeedBased(self, conf, ownship, intruder, idx, idx_pairs):
        #print(f'------------ {ownship.id[idx]} ------------')
        # Extract ownship data
        v_ownship = np.array([ownship.gseast[idx], ownship.gsnorth[idx]])# [m/s]
        
        # Check if we can simply apply the waypoint constraint
        # next_spd_ok = True
        
        # Initialise some variables
        t = bs.settings.asas_dtlookahead
        solutions = []
        
        # Go through all conflict pairs for aircraft "idx", basically take
        # intruders one by one, and create their polygons
        for i, idx_pair in enumerate(idx_pairs):
            idx_intruder = intruder.id.index(conf.confpairs[idx_pair][1])
            #print(f'### {intruder.id[idx_intruder]} ###')
            v_intruder = np.array([intruder.gseast[idx_intruder], intruder.gsnorth[idx_intruder]])
            
            # Do the check for turn waypoints
            if not (ownship.actwp.nextspd[idx] == self.turn_speed) and \
            (conf.tLOS[idx_pair] * ownship.gs[idx] > nm \
                    * kwikdist(ownship.actwp.lat[idx],
                                ownship.actwp.lon[idx],
                                ownship.lat[idx],
                                ownship.lon[idx])):
                next_spd_ok = False
            
            # Extract conflict bearing and distance information
            qdr = conf.qdr[idx_pair]
            dist= conf.dist[idx_pair]
            
            # Get the separation distance
            r = (conf.rpz[idx]) * 1.1
            
            # Relative position vector between ownship and intruder
            x = np.array([np.sin(qdr)*dist, np.cos(qdr)*dist])
            v_rel = v_ownship - v_intruder
            
            circle = Point(x/t).buffer(r/t)
            
            # Get cutoff legs
            left_leg_circle_point, right_leg_circle_point = self.cutoff_legs(x, r, t)
            
            right_leg_extended = right_leg_circle_point * t
            left_leg_extended = left_leg_circle_point * t
            
            triangle_poly = Polygon([right_leg_extended, right_leg_circle_point,
                                    left_leg_circle_point, left_leg_extended])
            
            final_poly = cascaded_union([triangle_poly, circle])
            
            # plt.plot(*final_poly.exterior.xy)
            # plt.scatter(v_rel[0], v_rel[1])
            
            # Create relative velocity point
            v_rel_point = Point(v_rel[0], v_rel[1])
            
            # Find nearest point on polygon
            p1, p2 = nearest_points(final_poly.exterior, v_rel_point)
            
            # Let's see it
            v_change = np.array(list(p1.coords))[0] - v_rel
            
            # So now we need to compute the velocity change for each aircraft
            # such that the new relative velocity changes by v_change
            # Compute unit direction vector of each aircraft
            norm_own = self.norm(v_ownship)
            norm_intruder = self.norm(v_intruder)
            norm_change_sq = self.norm_sq(v_change)
            
            # Own change that is guaranteed to be in the same direction as where
            # we are currently heading. 
            own_change = (np.dot(v_change, v_ownship)/norm_own**2)*v_ownship
        
            if np.degrees(self.angle(own_change, v_ownship)) < 1:
                solutions.append(self.norm(own_change))
            else:
                solutions.append(-self.norm(own_change))
        
        # Get minimum and maximum speed of ownship
        vmin = ownship.perf.vmin[idx]
        vmax = ownship.perf.vmax[idx]
        
        # Get the minimum solution
        min_limit = min(solutions)
        # Get the maximum solution
        max_limit = max(solutions)
        # Check if the max_limit is truly bigger than our current speed
        if max_limit > ownship.gs[idx]:
            # Get the difference
            max_difference = max_limit - ownship.gs[idx]
        else:
            max_difference = 0
            max_limit = ownship.gs[idx]
            
        # Do the same for the minimum
        if min_limit < ownship.gs[idx]:
            # Get the difference
            min_difference = ownship.gs[idx] - min_limit
        else:
            min_difference = 0
            min_limit = ownship.gs[idx]
            
        # Apply the speed that is closest to our current one
        if min_difference < max_difference:
            gs_new = min_limit
        else:
            gs_new = max_limit
            
        # So we have all the velocity limits. According to ORCA, all speeds that are
        # lower than ours are basically lower limits. All speeds that are greater are
        # upper limits. So the minimum and maximum values for the limits tell us the
        # interval in which we cannot be. Thus, take the min, take the max, and our
        # new velocity is the one that 

        
        return gs_new, ownship.ap.vs[idx]
    
    def pairs(self, conf, ownship, intruder, idx):
        '''Returns the indices of conflict pairs that involve aircraft idx
        '''
        idx_pairs = np.array([], dtype = int)
        for idx_pair, pair in enumerate(conf.confpairs):
            if (ownship.id[idx] == pair[0]):
                idx_pairs = np.append(idx_pairs, idx_pair)
        return idx_pairs
    
    def reso_pairs(self, conf, ownship, intruder, idx):
        '''Returns the indices of aircraft that are resolving conflicts with aircraft idx.
        '''
        idx_confs = np.array([], dtype = int)
        for pair in self.resopairs:
            if pair[0] == ownship.id[idx]:
                idx_confs = np.append(idx_confs, ownship.id.index(pair[1]))
        return idx_confs
    
    def perp_left(self, a):
        ''' Gives perpendicular unit vector pointing to the "left" (+90 deg)
        for vector "a" '''
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b/np.linalg.norm(b)

    def perp_right(self, a):
        ''' Gives perpendicular unit vector pointing to the "right" (-90 deg)
        for vector "a" '''
        b = np.empty_like(a)
        b[0] = a[1]
        b[1] = -a[0]
        return b/np.linalg.norm(b)
        
    def cutoff_legs(self, x, r, t):
        '''Gives the cutoff point of the right leg.'''
        x = np.array(x)
        # First get the length of x
        x_len = self.norm(x)
        # Find the sine of the angle
        anglesin = r / x_len
        # Find the angle itself
        angle = np.arcsin(anglesin) # Radians
        
        # Find the rotation matrices
        rotmat_left = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        
        rotmat_right = np.array([[np.cos(-angle), -np.sin(-angle)],
                           [np.sin(-angle), np.cos(-angle)]])
        
        # Compute rotated legs
        left_leg = rotmat_left.dot(x)
        right_leg = rotmat_right.dot(x)  
        
        circ = x/t
        xc = circ[0]
        yc = circ[1]
        xp_r = right_leg[0]
        yp_r = right_leg[1]
        xp_l = left_leg[0]
        yp_l = left_leg[1]
        
        b_r = (-2 * xc - 2 * yp_r / xp_r * yc)
        a_r = 1 + (yp_r / xp_r) ** 2    
         
        b_l = (-2 * xc - 2 * yp_l / xp_l * yc)
        a_l = 1 + (yp_l / xp_l) ** 2    
        
        x_r = -b_r / (2 * a_r)
        x_l = -b_l / (2 * a_l)
        
        y_r = yp_r / xp_r * x_r
        y_l = yp_l / xp_l * x_l 

        # Compute normalised directions
        right_cutoff_leg_dir = self.normalized(right_leg)
        self.right_cutoff_leg_dir = right_cutoff_leg_dir
        
        left_cutoff_leg_dir = self.normalized(left_leg)
        self.left_cutoff_leg_dir = left_cutoff_leg_dir
        
        return np.array([x_l, y_l]), np.array([x_r, y_r])
    
    def in_confpairs(self, idx):
        in_bool = False
        for pair in bs.traf.cd.confpairs:
            if idx in pair:
                in_bool = True
                break
        return in_bool
                
    def perp(self, a):
        return np.array((a[1], -a[0]))
    
    def norm_sq(self, x):
        return np.dot(x, x)
    
    def norm(self,x):
        return np.sqrt(self.norm_sq(x))
    
    def normalized(self, x):
        l = self.norm_sq(x)
        assert l > 0, (x, l)
        return x / np.sqrt(l)
    
    def angle(self, a, b):
        ''' Find non-directional angle between vector a and b'''
        return np.arccos(np.dot(a,b)/(self.norm(a) * self.norm(b)))
    
    def dist_sq(self, a, b):
        return self.norm_sq(b - a)
    
    ### Modified hdgactive function
    @property
    def hdgactive(self):
        ''' Return a boolean array sized according to the number of aircraft
            with True for all elements where heading is currently controlled by
            the conflict resolution algorithm.
        '''
        #TODO: Here is a good place to implement Open Airpace vs Restricted Airspace logic
        return np.array([False] * len(self.active))

    @property
    def altactive(self):
        ''' Return a boolean array sized according to the number of aircraft
            with True for all elements where altitude is currently controlled by
            the conflict resolution algorithm.
        '''
        return np.array([False] * len(self.active))
    
    ### Modified resumenav function
    def resumenav(self, conf, ownship, intruder):
        '''
            Decide for each aircraft in the conflict list whether the ASAS
            should be followed or not, based on if the aircraft pairs passed
            their CPA AND if ownship is a certain distance away from the intruding
            aircraft.
        '''
        # Add new conflicts to resopairs and confpairs_all and new losses to lospairs_all
        self.resopairs.update(conf.confpairs)

        # Conflict pairs to be deleted
        delpairs = set()
        changeactive = dict()

        # Look at all conflicts, also the ones that are solved but CPA is yet to come
        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)
            # If the ownship aircraft is deleted remove its conflict from the list
            if idx1 < 0:
                delpairs.add(conflict)
                continue

            if idx2 >= 0:
                # Distance vector using flat earth approximation
                re = 6371000.
                dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                      np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                              ownship.lat[idx1])),
                                      np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

                # Relative velocity vector
                vrel = np.array([intruder.gseast[idx2] - ownship.gseast[idx1],
                                 intruder.gsnorth[idx2] - ownship.gsnorth[idx1]])

                # Check if conflict is past CPA
                past_cpa = np.dot(dist, vrel) > 0.0

                # Also check the distance and altitude between the two aircraft.
                distance = self.norm(dist)
                dist_ok = (distance > 50)
                alt_ok = abs((ownship.alt[idx1]-intruder.alt[idx2])/ft) >= (conf.hpz[idx1])
                vs_ok = abs(ownship.vs[idx1]) < 0.1


                # hor_los:
                # Aircraft should continue to resolve until there is no horizontal
                # LOS. This is particularly relevant when vertical resolutions
                # are used.
                hdist = np.linalg.norm(dist)
                hor_los = hdist < conf.rpz[idx1]

                # Bouncing conflicts:
                # If two aircraft are getting in and out of conflict continously,
                # then they it is a bouncing conflict. ASAS should stay active until
                # the bouncing stops.
                is_bouncing = \
                    abs(ownship.trk[idx1] - intruder.trk[idx2]) < 30.0 and \
                    hdist < conf.rpz[idx1] * self.resofach

            # Start recovery for ownship if intruder is deleted, or if past CPA
            # and not in horizontal LOS or a bouncing conflict
            if idx2 >= 0 and (((not past_cpa or not (dist_ok or (alt_ok and vs_ok)) or hor_los or is_bouncing))):
                # Enable ASAS for this aircraft
                changeactive[idx1] = True
            else:
                # Switch ASAS off for ownship if there are no other conflicts
                # that this aircraft is involved in.
                changeactive[idx1] = changeactive.get(idx1, False)
                # If conflict is solved, remove it from the resopairs list
                delpairs.add(conflict)
                # Re-enable vnav
                stack.stack(f'VNAV {ownship.id[idx1]} ON')

        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            self.active[idx] = active
            if not active:
                # Waypoint recovery after conflict: Find the next active waypoint
                # and send the aircraft to that waypoint.
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:  # To avoid problems if there are no waypoints
                    bs.traf.ap.route[idx].direct(
                        idx, bs.traf.ap.route[idx].wpname[iwpid])

        # Remove pairs from the list that are past CPA or have deleted aircraft
        self.resopairs -= delpairs