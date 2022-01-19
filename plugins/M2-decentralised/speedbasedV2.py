from bluesky.traffic.turbulence import Turbulence
from bluesky.core.simtime import timed_function
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
        'plugin_name':     'SPEEDBASEDV2',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config

class SpeedBasedV2(ConflictResolution):
    def __init__(self):
        super().__init__()
        self.layer_height = 30 * ft
        self.cruiselayerdiff = self.layer_height * 3
        self.frnt_tol = 20 # Degrees
        self.rpz = 40
        with self.settrafarrays():
            self.in_headon = []
            self.stuck = np.array([], dtype = bool)
            self.dest_lat = np.array([])
            self.dest_lon = np.array([])
        
    def resolve(self, conf, ownship, intruder):
        '''We want to only solve in the velocity direction while still following the heading
        given by the autopilot. For each ownship, it calculates the minimum or maximum velocity
        needed to avoid all intruders. It then applies the solution closest to the current velocity.
        If there is no solution, it should then apply speed 0 by default, and the aircraft stops.'''
             
        # Make a copy of traffic data, track and ground speed
        newgscapped = np.copy(ownship.gs)
        newvs       = np.copy(ownship.vs)
        newalt      = np.copy(ownship.alt)
        
        # idx1 is of the ownship, idx2 is of the intruder
        # Iterate over aircraft in conflict
        for idx1 in np.argwhere(conf.inconf).flatten():            
            # Find the pairs in which IDX is involved in a conflict
            idx_pairs = self.pairs(conf, ownship, intruder, idx1)
            # We're doing this because we want to solve for ALL intruders, not only pairwise
            # Find solution for aircraft 'idx'
            gs_new, alt_new = self.SpeedBasedV2(conf, ownship, intruder, idx1, idx_pairs)
            
            # Write the new velocity of aircraft 'idx' to traffic data
            newgscapped[idx1] = gs_new
            newalt[idx1]      = alt_new  
        
        # Speed based, and 2D, for now.
        newtrack       = ownship.ap.trk
        newvs          = ownship.ap.vs

        return newtrack, newgscapped, newvs, newalt
    
    def SpeedBasedV2(self, conf, ownship, intruder, idx1, idx_pairs):
        # The resolving function is structured as follows:
        # We first do some preliminary checks: 
        #       -whether we can ascend or descend 
        #       -if we're close to a turn waypoint
        # Then we do a series of checks that result in this
        # intruder being ignored or not: 
        #       -check if there is a loss of separation 
        #       -check if the intent information rules out the conflict 
        #       -check if priority checks out 
        #       -check if aircraft is coming from the back 
        # Lastly, we do conflict resolution
        print(f'-------- {ownship.id[idx1]} -------')
        # ------------------- Pre-processing --------------------
        # Extract ownship data
        v1 = np.array([ownship.gseast[idx1], ownship.gsnorth[idx1]])# [m/s]
        # Skip this aircraft if its speed is 0
        if self.norm(v1) == 0:
            # We're standing still, and we probably want to continue to stand still
            # Maintain current altitude
            print('Standing still')
            return 0, ownship.alt[idx1]
        # Also take distance to other aircraft
        dist2others = conf.dist_mat[idx1]
        # Vertical maneuvers bools
        can_ascend = True
        should_ascend = True
        # Get the lookahead time
        t = bs.settings.asas_dtlookahead
        # Get the separation distances
        self.hpz = (conf.hpz[idx1]) * bs.settings.asas_marv
        # Initialize velocity obstacles
        VelocityObstacles = []
        
        # ------------ Aircraft above or below check --------------
        # We check if we can ascend or descend by checking aircraft above
        # or below
        can_ascend, can_descend = self.ac_above_below_check(conf, ownship, intruder, idx1, dist2others)
        #print(can_ascend, can_descend)
        
        # ------------ Start of iteration --------------
        for idx_pair in idx_pairs:
            # Get the index of the intruder
            idx2 = intruder.id.index(conf.confpairs[idx_pair][1])
            # Get the velocity of the intruder
            v2 = np.array([intruder.gseast[idx2], intruder.gsnorth[idx2]])
            # Extract conflict bearing and distance information
            qdr = conf.qdr[idx_pair]
            dist= conf.dist[idx_pair]
            # Find the bearing of the intruder with respect to where we are heading
            qdr_intruder = ((qdr - ownship.trk[idx1]) + 180) % 360 - 180  
                
            # -------------- State related checks -----------
            # Determine if intruder is in front
            in_front = (-self.frnt_tol < qdr_intruder < self.frnt_tol)
            # Determine if intruder is in the back
            in_back = (qdr_intruder < -180 + self.frnt_tol or 180 - self.frnt_tol < qdr_intruder)
            # Determine if intruder is close in altitude:
            alt_ok = ((abs(ownship.alt[idx1] - intruder.alt[idx2])) > self.hpz)
            # Determine if we have a LOS
            los = (dist <= self.rpz)
            # Determine if intruder is right above or below
            above = ((ownship.alt[idx1] - intruder.alt[idx2]) < 0)
            below = ((ownship.alt[idx1] - intruder.alt[idx2]) > 0)
            # Determine if intruder is coming towards us
            head_on = (abs((np.degrees(self.angle(v1, v2)))) > (180-self.frnt_tol))
            # Does the intent check out? If true, then the paths won't intersect
            intent_ok = self.check_intent(conf, ownship, intruder, idx1, idx2)
            # Does the priority check out? If true, then ownship has greater priority
            priority_ok = self.check_prio(ownship, intruder, idx1, idx2)
            
            # -------------- What to do if LOS ----------------
            if los:
                # If the circles are overlapping, then VO will not work, so
                # there's no continuing after this if statement, we need return
                #  or continue statements in here
                if alt_ok:
                    # Our distance is small, but we're still ok altitude wise. That means
                    # that one of the aircraft was ascending/descending towards the
                    # other. 
                    if above:
                        # The intruder is above us, so if we have priority, we just continue our way
                        if priority_ok:
                            print('LOS, alt ok, intruder above, priority ok.')
                            continue
                        else:
                            # We do not have priority, so we stop what we are doing and let the intruder
                            # pass. We also maintain current altitude
                            return 0, ownship.alt[idx1]
                    elif below:
                        # The intruder is below
                        if priority_ok:
                            # We have priority, continue what we were doing
                            print('LOS, alt ok, intruder below, priority ok.')
                            continue
                        else:
                            #If we can ascend, then we'll do that, unless we're already ascending
                            if can_ascend and ownship.vs[idx1] < 0.1:
                                alt = self.get_above_cruise_layer(ownship, idx1)
                                stack.stack(f'ALT {ownship.id[idx1]} {alt}')
                                stack.stack(f'LNAV {ownship.id[idx1]} ON') 
                                stack.stack(f'VNAV {ownship.id[idx1]} ON')
                                print('LOS, alt ok, intruder below, ascending.')
                                continue
                                
                            elif not can_ascend:
                                # We cannot ascend, stop and let the aircraft pass
                                # We also maintain current altitude
                                print('LOS, alt ok, intruder below, cannot ascend.')
                                return 0, ownship.alt[idx1]
                            else:
                                # We're probably already ascending then, so just continue
                                print('LOS, alt ok, already ascending.')
                                continue
                
                else:
                    # There is a complete LOS, so if we have higher priority, then we continue
                    # otherwise we stop
                    if priority_ok:
                        print('Complete LOS, priority ok.')
                        continue
                    else:
                        # We have lower priority, so we go back to the layer we came from
                        # We also maintain current altitude.
                        print('Complete LOS, no priority.')
                        return 0, ownship.alt[idx1]
                    
            # -------------- What to do if intruder in front ----------
            if in_front:
                # If the aircraft in front is ascending, don't ascend
                if (intruder.vs[idx2] > 0.1) and not alt_ok:
                    should_ascend = False
                
                # Same for descent
                if (intruder.vs[idx2] < -0.1) and not alt_ok:
                    should_descend = False
                    
                # Check if the intruder is actually coming towards us
                if head_on:
                    # TODO: Don't assume this is rogue drones only, use priority to determine
                    # who goes where in layers
                    # Head-on conflict, bad situation. Immediately ascend 
                    if self.in_headon[idx1] != True:
                        # This means this aircraft hasn't solved for head-on already
                        # Basically, the aircraft with the higher flight number ascends, and the one with the lower
                        # flight number descends. This ensures that any encounter of the sort is
                        # solved, regardless of whether one of the drones is a rogue or not.
                        if self.check_flight_numbers(ownship, intruder, idx1, idx2):
                            # Means our flight number is smaller, descend
                            alt = self.get_below_cruise_layer(ownship, idx1)
                        else:
                            # We ascend
                            alt = self.get_above_cruise_layer(ownship, idx1)
                        stack.stack(f'ALT {ownship.id[idx1]} {alt}')
                        stack.stack(f'LNAV {ownship.id[idx1]} ON') 
                        stack.stack(f'VNAV {ownship.id[idx1]} ON') 
                        print('In front, head-on, attempt alt change.')
                    self.in_headon[idx1] = True
                    
            # ------------- What to do if intruder in back --------------       
            elif in_back:
                # We're the ones in the front, don't ascend
                should_ascend = False
                continue
            
            # If we pass this check ^ , then intruder is not in the back
            else:
                should_ascend = False
                should_descend = False
            
            # -------------- Intent check --------------
            if intent_ok:
                print('Intent ok')
                # This means that the aircraft paths don't even get close, 
                # so just continue.
                continue
            
            # -------------- Priority check ------------
            
            if priority_ok and not in_front:
                # We have the greater priority, so if we didn't continue
                # till now, we do now
                # Only do this if we're not in the back though
                print('Prio ok and not in front')
                continue
            
            # ------------ CONFLICT RESOLUTION -------------
            # We already checked once if we're in a loss of separation, 
            # but only if we are in the back, and then we would stop. 
        
                
            # Get Velocity Obstacle
            VelocityObstacles.append(self.get_VO(conf, ownship, intruder, idx1, idx2))
        
        #------------ FOR LOOP OVER ----------------
        #------------ Overall processing -----------
        #First of all, if we can ascend, let's do it
        if can_ascend and should_ascend and self.in_headon[idx1] != True:
            alt = self.get_above_cruise_layer(ownship, idx1)
            stack.stack(f'ALT {ownship.id[idx1]} {alt}')
            stack.stack(f'LNAV {ownship.id[idx1]} ON') 
            stack.stack(f'VNAV {ownship.id[idx1]} ON')
            
        #If we cannot ascend, let's mark ourselves as stuck
        elif not can_ascend and should_ascend:
            # Means we are most likely stuck behind an aircraft. Mark this aircraft
            # as stuck, and in queue for ascending
            self.stuck[idx1] = True
            print('STUCK')
            
        # Combine all velocity obstacles into one big polygon
        CombinedObstacles = cascaded_union(VelocityObstacles)
        
        # Get minimum and maximum speed of ownship
        vmin = ownship.perf.vmin[idx1]
        if bs.traf.ap.inturn[idx1] or bs.traf.ap.dist2turn[idx1] < 100:
            vmax = bs.traf.actwp.nextturnspd[idx1] 
        else:
            vmax = ownship.perf.vmax[idx1]
        # Create velocity line
        v_dir = self.normalized(v1)
        v_line_min = v_dir * vmin
        v_line_max = v_dir * vmax
        
        # Create velocity line
        line = LineString([v_line_min, v_line_max])
        # Get the intersection with the velocity obstacles
        intersection = CombinedObstacles.intersection(line)
        
        #---------------- RESOLUTION SPEEDS ---------------
        # Apply the VO resolution speed
        if intersection:
            solutions = []
            for velocity in list(intersection.coords):
                print(velocity)
                # Check whether to put velocity "negative" or "positive". 
                # Drones can fly backwards.
                if np.degrees(self.angle(velocity, v1)) < 1:
                    solutions.append(self.norm(velocity))
                else:
                    solutions.append(-self.norm(velocity))
            gs_new = min(solutions)
        else:
            # Nothing worked, do nothing
            gs_new = ownship.ap.tas[idx1]
        return gs_new, ownship.ap.alt[idx1]
    
    ##### Helper functions #####
    def check_speed(self, conf, ownship, intruder, idx1, speed):
        # Check if the given speed for aircraft idx1 would cause any conflicts
        # with other aircraft in the vicinity. however, ignore aircraft that
        # are in the back or have a lower priority, but always count ones
        # in the front
        # First of all, extract distance to all other aircraft
        dist2others = conf.dist_mat[idx1]
        # Extract the closest aircraft, within lookahead distance
        dist_look = ownship.gs[idx1] * conf.dtlookahead[idx1]
        idx_others = np.where(dist2others < dist_look)[0]
        # Get ownship speed
        v1 = np.array([ownship.gseast[idx1], ownship.gsnorth[idx1]])
        # Initialize Velocity Obstacles
        VelocityObstacles = []
        # Initialize los bool
        los = False
        for idx2 in idx_others:
            # Eliminate aircraft that are in the back, have a lower priority
            # but always consider aircraft in the front
            qdr = conf.qdr_mat[idx1, idx2]
            dist = conf.dist_mat[idx1, idx2]
            if dist < self.rpz:
                los = True
                continue
            qdr_intruder = ((qdr - ownship.trk[idx1]) + 180) % 360 - 180
            in_front = (-self.frnt_tol < qdr_intruder < self.frnt_tol)
            in_back = (qdr_intruder < -180 + self.frnt_tol or 180 - self.frnt_tol < qdr_intruder)
            priority_ok = self.check_prio(ownship, intruder, idx1, idx2)
            if (priority_ok and not in_front) or (in_back):
                continue
            VelocityObstacles.append(self.get_VO(conf, ownship, intruder, idx1, idx2))
        if los:
            return False
        # If there aren't any velocity obstacles, return true
        if not VelocityObstacles:
            return True
        # Combine all the VOs
        CombinedObstacles = cascaded_union(VelocityObstacles)
        # Create the speed vector
        v_dir = self.normalized(v1)
        v_to_check = v_dir * speed
        wpxv = v_to_check[0]
        wpyv = v_to_check[1]
        wpoint = Point(wpxv, wpyv)
        return not CombinedObstacles.contains(wpoint)
    
    def get_VO(self, conf, ownship, intruder, idx1, idx2):
        rpz = self.rpz
        t = conf.dtlookahead[idx1]
        # Get QDR and DIST of conflict
        qdr = conf.qdr_mat[idx1, idx2]
        dist = conf.dist_mat[idx1,idx2]
        # Get radians qdr
        qdr_rad = np.radians(qdr)
        # Get relative position
        x_rel = np.array([np.sin(qdr_rad)*dist, np.cos(qdr_rad)*dist])
        # Get the speed of the intruder
        v2 = np.array([intruder.gseast[idx2], intruder.gsnorth[idx2]])
        # Get cutoff legs
        left_leg_circle_point, right_leg_circle_point = self.cutoff_legs(x_rel, self.rpz, t)
        # Extend cutoff legs
        right_leg_extended = right_leg_circle_point * t
        left_leg_extended = left_leg_circle_point * t
        # Get the final VO
        final_poly = Polygon([right_leg_extended, (0,0), left_leg_extended])
        # Translate it by the velocity of the intruder
        final_poly_translated = translate(final_poly, v2[0], v2[1])
        # Return
        return final_poly_translated
    
    def check_prio(self, ownship, intruder, idx1, idx2):
        if not hasattr(ownship, 'priority'):
            # Determine which ACID number is bigger
            if self.check_flight_numbers(ownship, intruder, idx1, idx2):
                return True
            else:
                return False
            
        ownship_prio = ownship.priority[idx1]
        intruder_prio = intruder.priority[idx2]
        
        if (ownship_prio > intruder_prio) and self.in_headon[idx1] != True:
            # Priority of intruder is greater, continue.
            return True
        
        if (ownship_prio == intruder_prio) and self.in_headon[idx1] != True:
            # Determine which ACID number is bigger
            if self.check_flight_numbers(ownship, intruder, idx1, idx2):
                return True
            
        return False
    
    def check_flight_numbers(self, ownship, intruder, idx1, idx2):
        """If ACID of idx1 < idx2, returns True, else False.
        """
        id1= ownship.id[idx1]
        id2 = intruder.id[idx2]
        prio_bigger = int(''.join(filter(str.isdigit, id1))) < int(''.join(filter(str.isdigit, id2)))
        if prio_bigger:
                return True
        else:
            return False
            
    def check_intent(self, conf, ownship, intruder, idx1, idx2):
        if (intruder.intent[idx2] is not None) and (ownship.intent[idx1] is not None):
            intent1, target_alt1 = ownship.intent[idx1]
            intent2, target_alt2 = intruder.intent[idx2]
            # Find closest points between the two intent paths
            pown, pint = nearest_points(intent1, intent2)
            # Find the distance between the points
            point_distance = kwikdist(pown.y, pown.x, pint.y, pint.x) * nm #[m]
            # Also do vertical intent
            # Difference between own altitude and intruder target
            diff = ownship.alt[idx1] - target_alt2
            # Basically, there are three conditions to be met in order to skip
            # a conflict due to intent:
            # 1. The minimum distance between the horizontal intent lines is greater than r;
            # 2. The difference between the current altitude and the target altitude of the 
            # intruder is greater than the vertical separation margin;
            # 3. The altitude difference and vertical velocity of the intruder have the same sign.
            # This means that if the aircraft is coming from above (negative), and the altitude difference
            # is positive (thus target altitude is below ownship), then their paths will intersect. 
            if ((point_distance > self.rpz) or (abs(diff) >= self.hpz)) and \
                (abs(intruder.vs[idx2]) < 0.1):
                    # Intent is ok
                return True
            else:
                return False
            
    def get_above_cruise_layer(self, ownship, idx1):
        # Get the cruise layer above the current altitude of the ownship
        if hasattr(bs.traf, 'closest_cruise_layer_top'):
            return bs.traf.closest_cruise_layer_top[idx1]
        else:
            return (ownship.alt[idx1])/ft + 75
        
    def get_below_cruise_layer(self, ownship, idx1):
        # Get the cruise layer below the current altitude of the ownship
        if hasattr(bs.traf, 'closest_cruise_layer_bottom'):
            return bs.traf.closest_cruise_layer_bottom[idx1]
        else:
            return (ownship.alt[idx1])/ft - 75
        
    def in_cruise_layer(self, ownship, idx1):
        if hasattr(bs.traf, 'flight_layer_type'):
            return bs.traf.flight_layer_type[idx1] == 'C'
        else:
            return True
                
    def ac_above_below_check(self, conf, ownship, intruder, idx1, dist2others):
        can_ascend = True
        can_descend = True
        # Get aircraft that are close
        is_close = np.where(dist2others < self.rpz * 2)[0]
        # Get the vertical distance for these aircraft
        vertical_dist = ownship.alt[idx1] - intruder.alt[is_close]
        # Check if any is smaller than cruise layer difference
        cruise_diff_ascend = np.logical_and(0 > vertical_dist, vertical_dist > (-self.cruiselayerdiff * 1.1))
        cruise_diff_descend = np.logical_and(0 < vertical_dist, vertical_dist < (self.cruiselayerdiff * 1.1))
        # Check also if any is smaller than conf.hpz
        conf_diff = np.array([np.abs(x) > conf.hpz[idx1] for x in vertical_dist])
        # Do the and operation on these two
        dealbreaker_ascend = np.logical_and(cruise_diff_ascend, conf_diff) 
        dealbreaker_descend = np.logical_and(cruise_diff_descend, conf_diff)
        
        if np.any(dealbreaker_ascend):
            can_ascend = False
        if np.any(dealbreaker_descend):
            can_descend = False
            
        return can_ascend, can_descend
                        
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
                # We want enough distance between aircraft
                dist_ok = (distance > 100) 
                # We also want enough altitude
                alt_ok = abs((ownship.alt[idx1]-intruder.alt[idx2])/ft) >= (2 * self.layer_height)
                # We also want to make sure that the aircraft has finished its vertical manoeuvre 
                vs_ok = abs(ownship.vs[idx1]) < 0.1
                # We also want to make sure that the autopilot doesn't start making it ascend or
                # descend again in the direction of the conflict
                ap_vs_ok = True #abs(ownship.ap.vs[idx1]) < 0.1
                # Lastly, we want to make sure that the autopilot speed command doesn't create
                # other conflicts
                ap_spd_ok = self.check_speed(conf, ownship, intruder, idx1, ownship.ap.tas[idx1])

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
                    
                # Group some checks together
                # The autopilot is truly ok if both the vertical separation and the speed
                # it wants to apply are ok
                ap_ok = ap_spd_ok and alt_ok 
                # Navigation is ok if 
                # - Altitude is ok and vertical speed is 0 OR 
                # - Distance between the aircraft is ok OR 
                # - The autopilot is ok AND
                # - The autopilot vertical speed is ok
                nav_ok = ((alt_ok or dist_ok) or ap_ok) and ap_vs_ok
                conf_ok = (past_cpa and not hor_los and not is_bouncing) or alt_ok

            # Start recovery for ownship if intruder is deleted, or if past CPA
            # and not in horizontal LOS or a bouncing conflict
            if idx2 >= 0 and (not (nav_ok and conf_ok)):
                # Enable ASAS for this aircraft
                changeactive[idx1] = True
                # We also need to check if this aircraft needs to be doing a turn, aka, if the
                # autopilot speed is lower than the CR speed. If it is, then we need to update
                # the speed to that value. Thus, either the conflict is still ok and CD won't be
                # triggered again, or a new conflict will be triggered and CR will take over again.
                if self.tas[idx1] > bs.traf.ap.tas[idx1]:
                    self.tas[idx1] = bs.traf.ap.tas[idx1]
            else:
                # Switch ASAS off for ownship if there are no other conflicts
                # that this aircraft is involved in.
                changeactive[idx1] = changeactive.get(idx1, False)
                # If conflict is solved, remove it from the resopairs list
                delpairs.add(conflict)
                # In case it was a head-on conflict
                self.in_headon[idx1] = False
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