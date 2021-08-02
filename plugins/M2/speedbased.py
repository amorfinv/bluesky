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
from bluesky.tools.aero import nm, ft
import bluesky as bs
import numpy as np
import itertools

def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SPEEDBASED',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config

class SpeedBased(ConflictResolution): 
    # Define some variables
    def __init__(self):
        super().__init__()
        self.cruiselayerdiff = 75 * ft
        self.min_alt = 25 * ft
        with self.settrafarrays():
            self.in_headon_conflict = []
        
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
            gs_new, vs_new = self.SpeedBased(conf, ownship, intruder, idx, idx_pairs)
            
            # Write the new velocity of aircraft 'idx' to traffic data
            newgscapped[idx] = gs_new    
            newvs[idx] = vs_new    
        
        # Speed based, and 2D, for now.
        alt            = ownship.ap.alt 
        newtrack       = ownship.ap.trk
        
        return newtrack, newgscapped, newvs, alt


    def SpeedBased(self, conf, ownship, intruder, idx, idx_pairs):
        print(f'------------ {ownship.id[idx]} ------------')
        # Extract ownship data
        v_ownship = np.array([ownship.gseast[idx], ownship.gsnorth[idx]])# [m/s]
        
        # Also take distance to other aircraft
        dist2others = conf.dist_mat[idx]
        
        # Descend and ascend checks
        can_ascend = True
        should_ascend = True
        
        # Check if aircraft can ascend or descend to another cruise layer
        # Basically, we check if there are other aircraft above or below
        for idx_other, dist in enumerate(dist2others):
            # First, check if distance is smaller than rpz * 1.5
            if dist < (conf.rpz[idx] + conf.rpz[idx_other]) * 2:
                # Check if the vertical distance is smaller than one layer hop, but also
                # that we're not already in a conflict with this aircraft
                vertical_dist = ownship.alt[idx] - intruder.alt[idx_other]
                if abs(vertical_dist) < self.cruiselayerdiff * 1.1 and abs(vertical_dist) > conf.hpz[idx]:
                    # Ok so this basically means we cannot ascend or descend
                    if vertical_dist < 0:
                        # An aircraft is above
                        can_ascend = False
        
        #print(f'#1 - can ascend - {can_ascend}')
        
        # Initialise some variables
        t = bs.settings.asas_dtlookahead
        target_alt = ownship.alt[idx]
        
        VelocityObstacles = []
        # Go through all conflict pairs for aircraft "idx", basically take
        # intruders one by one, and create their polygons
        for i, idx_pair in enumerate(idx_pairs):
            idx_intruder = intruder.id.index(conf.confpairs[idx_pair][1])
            print(f'### {intruder.id[idx_intruder]} ###')
            v_intruder = np.array([intruder.gseast[idx_intruder], intruder.gsnorth[idx_intruder]])
            # Extract conflict bearing and distance information
            qdr = conf.qdr[idx_pair]
            dist= conf.dist[idx_pair]
            
            dist2 = qdrdist(ownship.lat[idx], ownship.lon[idx], 
                            intruder.lat[idx_intruder], intruder.lon[idx_intruder])

            dist3 = latlondist(ownship.lat[idx], ownship.lon[idx], 
                            intruder.lat[idx_intruder], intruder.lon[idx_intruder])
            
            # Get the separation distance
            r = (conf.rpz[idx]) * 1.1
            
            # Find the bearing of the intruder with respect to where we are heading
            qdr_intruder = ((qdr - ownship.trk[idx]) + 180) % 360 - 180  
            
            # If we have a loss of separation, or the conflict is vertical,
            # just break, and stop the vertical speed
            print(f'#2 - dist < r - {dist < r}')
            print(f'#2 - dist - {dist}')
            print(f'#2 - r - {r}')
            print(f'#3 - qdr_intruder - {qdr_intruder}') 
            
            # First, let's clear some vertical matters. If an intruder is in front and
            # is performing a vertical maneuver, then prevent aircraft in back from
            # performing the same maneuver.
            if (-10 < qdr_intruder < 10):
                if (dist < r):
                    #print(dist, r)
                    # We have a loss of separation
                    can_ascend = False
                    can_descend = False
                    return 1, 0
                if intruder.vs[idx_intruder] > 0.1:
                    # Aircraft in front is performing an ascent maneuver
                    should_ascend = False
                    
                # Here is a good place to also check if this is a head-on conflict
                if ((np.degrees(self.angle(v_ownship, v_intruder))) > 170):
                    # This is a head-on conflict, immediately ascend into an unused layer (50 ft above)
                    # Also, current aircraft doesn't have a vertical velocity so an altitude command
                    # is possible. 
                    if self.in_headon_conflict[idx] != True:
                        stack.stack(f'ALT {ownship.id[idx]} {(ownship.alt[idx])/ft + 50}')
                        
                    self.in_headon_conflict[idx] = True
                    
            if not(-10 < qdr_intruder < 10):
                should_ascend = False
            
            print(f'#4 - should_ascend - {should_ascend}')   
            print(f'#5 - in_headon_conflict - {self.in_headon_conflict[idx]}')   
            # Check if intruder is coming from the back. If yes, then ignore it.
            if (-180 <= qdr_intruder  < -160) or (160 <= qdr_intruder  < 180):
                # From the back, but if we're in a loss of separation, then just
                # advance normally
                if dist < r:
                    return ownship.ap.tas[idx], ownship.ap.vs[idx]
                # go to next pair
                continue    
                    
            # Let's also do some intent check if the intent plugin is loaded
            own_intent, own_target_alt = ownship.intent[idx]
            intruder_intent, intruder_target_alt = intruder.intent[idx_intruder]
            # Find closest points between the two intent paths
            pown, pint = nearest_points(own_intent, intruder_intent)
            # Find the distance between the points
            point_distance = kwikdist(pown.y, pown.x, pint.y, pint.x) * nm #[m]
            # Also do vertical intent
            # Difference between own altitude and intruder target
            diff = ownship.alt[idx] - intruder_target_alt
            # Basically, there are three conditions to be met in order to skip
            # a conflict due to intent:
            # 1. The minimum distance between the horizontal intent lines is greater than r;
            # 2. The difference between the current altitude and the target altitude of the 
            # intruder is greater than the vertical separation margin;
            # 3. The altitude difference and vertical velocity of the intruder have the same sign.
            # This means that if the aircraft is coming from above (negative), and the altitude difference
            # is positive (thus target altitude is below ownship), then their paths will intersect. 
            #print(f'#6 - point_distance > r - {point_distance > r}')  
            #print(f'#7 - abs(diff) >= conf.hpz[idx] - {abs(diff) >= conf.hpz[idx]}')  
            if ((point_distance > r) or (abs(diff) >= conf.hpz[idx])) and \
                (abs(intruder.vs[idx_intruder]) < 0.1):
                continue
            
            # Finally, let's do a priority check. We ignore aircraft with a priority
            # (or callsign in case of draw) less then the ownships'.
            ownship_prio = ownship.priority[idx]
            intruder_prio = intruder.priority[idx_intruder]
            
            print(f'#8 - ownship_prio - {ownship_prio}')
            print(f'#9 - intruder_prio - {ownship_prio}')
            
            if (ownship_prio > intruder_prio) and self.in_headon_conflict[idx] != True:
                # Priority of intruder is greater, continue.
                continue
            
            print(f'#10 - prio=prio - {ownship_prio == intruder_prio}')
            if (ownship_prio == intruder_prio) and self.in_headon_conflict[idx] != True:
                # Determine which ACID number is bigger
                id_ownship = ownship.id[idx]
                id_intruder = intruder.id[idx_intruder]
                prio_bigger = int(''.join(filter(str.isdigit, id_ownship))) > int(''.join(filter(str.isdigit, id_intruder)))
                print(f'#10.1 - prio_bigger - {prio_bigger}')
                if prio_bigger:
                    continue
                
            # --------------- Actual conflict resolution calculation------------
            # Until now we had exceptions, now we do actual maneuvers.
            # If we didn't skip this aircraft until now, do a final loss of separation
            # check for any other situation in which it could happen
            if dist < r:
                return 1, 0
            # Set the target altitude in case we can ascend
            target_alt = intruder.alt[idx] + self.cruiselayerdiff
            
            # Convert qdr from degrees to radians
            qdr = np.radians(qdr)

            # Relative position vector between ownship and intruder
            x = np.array([np.sin(qdr)*dist, np.cos(qdr)*dist])
            
            print('x', x)

            # Get cutoff legs
            left_leg_circle_point, right_leg_circle_point = self.cutoff_legs(x, r, t)

            right_leg_extended = right_leg_circle_point * t
            left_leg_extended = left_leg_circle_point * t
            
            final_poly = Polygon([right_leg_extended, (0,0), left_leg_extended])
            
            final_poly_translated = translate(final_poly, v_intruder[0], v_intruder[1])
            
            VelocityObstacles.append(final_poly_translated)
            
        # Went through all intruders, now let's try to hop a layer
        if can_ascend and should_ascend and self.in_headon_conflict[idx] != True:
            stack.stack(f'ALT {ownship.id[idx]} {target_alt/ft}')
        
        # Combine all velocity obstacles into one figure
        CombinedObstacles = cascaded_union(VelocityObstacles)
        
        # Get minimum and maximum speed of ownship
        vmin = ownship.perf.vmin[idx]
        vmax = ownship.perf.vmax[idx]
        # Create velocity line
        v_dir = self.normalized(v_ownship)
        v_line_min = v_dir * vmin
        v_line_max = v_dir * vmax
        
        # Create velocity line
        line = LineString([v_line_min, v_line_max])
        intersection = CombinedObstacles.intersection(line)
        
        # Check if autopilot given speed is also good for velocity obstacle
        apyv = ownship.ap.tas[idx] * np.sin(np.radians(ownship.ap.trk[idx]))
        apxv = -ownship.ap.tas[idx] * np.cos(np.radians(ownship.ap.trk[idx]))
        appoint = Point(apxv, apyv)
        print(f'#11 - apxv, apyv - {apxv, apyv}')
        ap_ok = not CombinedObstacles.contains(appoint)
        print(f'#12 - ap_ok = {ap_ok}')
        
        if ap_ok:
            print('')
            return ownship.ap.tas[idx], ownship.ap.vs[idx]

        # First check if the autopilot speed creates any conflict
        if intersection:
            solutions = []
            for velocity in list(intersection.coords):
                # Check whether to put velocity "negative" or "positive". 
                # Drones can fly backwards.
                if np.degrees(self.angle(velocity, v_ownship)) < 1:
                    solutions.append(self.norm(velocity))
                else:
                    solutions.append(-self.norm(velocity))
            gs_new = min(solutions)
        else:
            # Maintain current speed
            gs_new = ownship.gs[idx]
        
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
        
        print('r', r)
        print('x', x)
        print('x_len', x_len)
        
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
    
    #@core.timed_function(name = 'stuck_checker', dt=5)
    def check_traffic(self):
        """This function does a periodic sweep of all aircraft and
        checks whether they are stuck behind a slow moving aircraft
        or whether they can descend.
        """        
        # Get the required variables from Bluesky
        traf = bs.traf
        conf = traf.cd
        ownship = traf
        intruder = traf
        for idx, acid in enumerate(traf.id):
            # First, check if we can actually do anything with this aircraft
            # We ignore an aircraft if it is currently performing a vertical maneuver
            if abs(ownship.vs[idx]) > 0.01:
                continue
            
            # We also ignore it if it's currently in a head-on conflict
            if self.in_headon_conflict[idx] == True:
                continue
            
            # We definitely ignore it if it is currently solving a conflict
            if self.active[idx] == True:
                continue
            
            # Descend and ascend checks
            can_ascend, can_descend, should_ascend, should_descend = \
                    self.check_ascent_descent(conf, ownship, intruder, idx)
                    
            if (not can_ascend and not can_descend) or (not should_ascend and not should_descend):
                # Stop here, there's nothing to do anyway with this aircraft
                continue
            
            # We check if aircraft is active but is no longer in conflict pairs. 
            # It means it's a candidate for getting stuck. We can then check the heading
            # between the two aircraft and see if one is behind the other. 
            if self.active[idx] and not self.in_confpairs(idx):
                idx_others = self.reso_pairs(conf, ownship, intruder, idx)
                for idx_intruder in idx_others:
                    qdr_intruder = ((conf.qdr_mat[idx, idx_intruder]- ownship.trk[idx]) + 180) % 360 - 180
                    if not(-10 < qdr_intruder < 10):
                        should_ascend = False
                target_alt = ownship.alt[idx] + self.cruiselayerdiff
                if can_ascend and should_ascend:
                    # Aircraft can ascend to next layer
                    stack.stack(f'ALT {ownship.id[idx]} {target_alt/ft}') 
                    # Continue to next aircraft
                    continue
            
            # Now we descend if we can
            if can_descend and should_descend:
                target_alt = ownship.alt[idx] - self.cruiselayerdiff
                # Check if we're above the minimum altitude
                if target_alt >= self.min_alt:
                    stack.stack(f'ALT {ownship.id[idx]} {target_alt/ft}') 
                    continue
        return
                
    
    def check_ascent_descent(self, conf, ownship, intruder, idx):
        # Check distance to other aircraft
        dist2others = conf.dist_mat[idx]
        qdr2others  = conf.qdr_mat[idx]
        
        dlookahead = bs.settings.asas_dtlookahead * ownship.gs[idx]
        # Descend and ascend checks
        # These two tell me if there is any aircraft above and below
        can_ascend = True 
        can_descend = True 
        # These two tell me if the aircraft in front is ascending or descending
        should_ascend = True
        should_descend = True
        
        # Also check the bearing of all the neighbors to see if there is anyone in
        # front, otherwise we don't need to ascend.
        qdr_list = np.array([])
        
        # Check if aircraft can ascend or descend to another cruise layer
        # Basically, we check if there are other aircraft above or below
        for idx_other, dist in enumerate(dist2others):
            # Check if there is any aircraft in front within the lookahead time that
            # is doing a vertical maneuver
            if dist < dlookahead:
                qdr = qdr2others[idx_other]
                qdr_intruder = ((qdr - ownship.trk[idx]) + 180) % 360 - 180
                qdr_list = np.append(qdr_list, qdr_intruder)
                
                # Check if there is any aircraft in front that is doing a maneuver
                if (-10 < qdr_intruder < 10 and intruder.vs[idx_other] > 0.01):
                    should_ascend = False
                elif (-10 < qdr_intruder < 10 and intruder.vs[idx_other] < -0.01):
                    should_descend = False
                    
                # Checking if any aircraft are above
                # Check if distance is smaller than rpz * 1.5
                if dist < (conf.rpz[idx]) * 2:
                    # Check if the vertical distance is smaller than one layer hop, but also
                    # that we're not already in a conflict with this aircraft
                    vertical_dist = ownship.alt[idx] - intruder.alt[idx_other]
                    if abs(vertical_dist) < self.cruiselayerdiff * 1.1 and abs(vertical_dist) > conf.hpz[idx]:
                        # Ok so this basically means we cannot ascend or descend
                        if vertical_dist < 0:
                            # An aircraft is above
                            can_ascend = False
                        elif vertical_dist > 0:
                            # An aircraft is below
                            can_descend = False  
        # Finally, if nobody is in front of us, then don't ascend
        if any([not(-10 < x < 10) for x in qdr_list]):
            should_ascend = False 
        return can_ascend, can_descend, should_ascend, should_descend
        
    
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
                alt_ok = abs((ownship.alt[idx1]-intruder.alt[idx2])/ft) >= (self.cruiselayerdiff - 1)
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
                # In case it was a head-on conflict
                self.in_headon_conflict[idx1] = False
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