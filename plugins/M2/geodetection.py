""" Detection of geofence conflicts."""
import numpy as np
import bluesky as bs
from bluesky.core import Entity, timed_function
from shapely.ops import nearest_points
from shapely.geometry.polygon import Polygon, Point, LineString
from bluesky.tools import geo
from bluesky.tools.aero import nm

def init_plugin():
    # Create new geofences dictionary
    bs.traf.geod = GeofenceDetection()
    # Create new point search MATRIX
    
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name'      : 'GEODETECTION',
        'plugin_type'      : 'sim',
        'update_interval'  :  1.0,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        # 'update':          update,

        # The preupdate function is called before traffic is updated. Use this
        # function to provide settings that need to be used by traffic in the current
        # timestep. Examples are ASAS, which can give autopilot commands to resolve
        # a conflict.
        # 'preupdate':       preupdate,

        # Reset all geofences
        'reset':         bs.traf.geod.reset
        }

    # init_plugin() should always return these two dicts.
    return config
class Tile:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class GeofenceDetection(Entity):
    def __init__(self):
        super().__init__()
        # Dictionaries to keep track of whether aircraft are inside or
        # outside geofences. The only reason these exist is that, if the geofence is large
        # enough (or tile zoom is large enough) and an aircraft is inside it, because 
        # of the tile-based optimisation, it will not be detected. For this situation 
        # specifically, we need to keep track of aircraft that are inside a geofence. 
        
        # Conflict detection
        self.geoconfs = dict() # Stores current conflicts
        self.geobreaches = dict() # Stores current breaches
        self.method = 'RTREE'
        
        # Historic lists
        self.allgeobreaches = [] # Stores all pastgeofence breaches 
        self.allgeoconfs = [] # Stores all past geofence conflicts
        return
    
    # This function is called from within traffic
    @timed_function(name = 'geofencedetection', dt = 0.1)
    def update(self):
        ownship = bs.traf
        # Select detection method
        if self.method == 'TILES':
            self.GeodetectTiles(ownship)   
        elif self.method == 'RTREE':
            self.GeodetectRtree(ownship)
        return
    
    @timed_function(name = 'geocleanup', dt = 0.1)
    def cleanup(self):
        # Cleans up geofence breaches
        # Go through all the entries in geofence breaches
        newdict = dict()
        for acid in self.geobreaches:
            if acid not in bs.traf.id:
                continue
            
            i = bs.traf.id.index(acid)
            pos_point = Point(bs.traf.lat[i], bs.traf.lon[i])
            geofences_to_stay = set()
            for geofence in self.geobreaches[acid]:
                if geofence.getPoly().contains(pos_point):
                    geofences_to_stay.add(geofence)
            
            if geofences_to_stay:
                newdict[acid] = geofences_to_stay
                
        self.geobreaches = newdict
                
        
    def reset(self):
        self.geoconfs = dict()
        self.geobreaches = dict()
        self.allgeobreaches = []
        self.allgeoconfs = []
        return
        
    def delgeofence(self, geofencename):
        '''Delete geofence from the dictionaries in this class.'''
        # First do the easy bit, simply remove geofence entry
        if geofencename in self.geoconfs:
            self.geofencehasac.pop(geofencename)
            
        # Then the harder part. For loop over dictionary
        for key in self.geoconfs:
            if geofencename in self.geoconfs[key]:
                self.geoconfs[key].remove(geofencename)
                
        for key in self.geoconfs:
            if geofencename in self.geoconfs[key]:
                self.geoconfs[key].remove(geofencename)
        return
        
    def delaircraft(self, acid):
        '''Delete acid from the dictionaries in this class.'''
        # Easy bit, remove ac entry
        if acid in self.acingeofence:
            self.acingeofence.pop(acid)
            
        # Loop over geofence dictionary
        for key in self.geofencehasac:
            if acid in self.geofencehasac[key]:
                self.geofencehasac[key].remove(acid)
        return
                
    def GeodetectTiles(self, ownship):
        '''The tiles based method is more accurate than the Rtree method, as it does not assume the Earth to be
        flat. However, it is about 10x slower. It also performes badly if geofences are really big while the zoom
        level is really small.'''
        # Check if geofence plugin is enabled
        if 'geofence' not in bs.core.varexplorer.varlist:
            return 'Geofence plugin not loaded.'
        # Retrieve geofence plugin instance
        geofenceplugin = bs.core.varexplorer.varlist['geofence'][0]
        
        # Get geofence data from plugin
        geofenceTileData = geofenceplugin.TileData
        geofenceData = geofenceplugin.geofences
        
        # If there are no geofences then there is no point in continuing
        if not geofenceData:
            return
        
        # Get zoom level
        z = geofenceTileData.z
        
        # Lookahead distance
        dlookahead = bs.settings.geofence_dlookahead
        
        # For now, detection is a bit unoptimised as in we need to loop over all aircraft and
        # see if there are geofences in their tiles. But as long as the number of aircraft is
        # not exaggerated (like, 10k at once), then this is fine. 
        ntraf = ownship.ntraf
        for i in np.arange(ntraf):
            acid = ownship.id[i]
            geoinvicinity = []
            # Get aircraft position and tile
            aclat, aclon = ownship.lat[i], ownship.lon[i]
            actileX, actileY = self.tileXY(aclat, aclon, z)
            
            # Get aircraft position and tile
            pos_ac = np.array([ownship.lat[i], ownship.lon[i], ownship.alt[i]])
            hdg_ac = ownship.trk[i]
            
            # Project the current position of the aircraft in the direction of the heading
            pos_next = geo.qdrpos(pos_ac[0], pos_ac[1], hdg_ac, dlookahead / nm)
            
            # Create a line for convenience, as line.bounds gives the correct format for the rtree
            trajectory = LineString(np.array([pos_ac[:2], pos_next]))
            
            # This is a set to automatically avoid duplicate geofence names
            geofence_names = set()
            
            # Get adjacent tiles depending on lookahead distance or time
            tiles = self.getAdjacentTiles(actileX, actileY, z)
            
            # Retrieve names of geofences in considered tiles
            for tile in tiles:
                tile = (tile.x, tile.y)
                if tile in geofenceTileData.tiledictionary:
                    geofence_names = geofence_names.union(geofenceTileData.tiledictionary[tile])
            
            if geofence_names:
                for geofence_name in geofence_names:
                    # Get the geofence itself
                    geofence = geofenceData[geofence_name]
                    # Create shapely polygon
                    geofencepoly = Polygon(geofence.getPointArray())
                    # Find closest point on this polygon w.r.t aircraft, p2 unused
                    # When geofences are very big, this becomes inaccurate, as this does
                    # not take into account the curvature of the Earth. 
                    # TODO: find a solution for this to follow curvature of Earth. 
                    p1, p2 = nearest_points(geofencepoly, Point(aclat, aclon))
                    # Get closest point Coordinates
                    plat, plon = p1.x, p1.y
                    # Get distance between aircraft and geofence
                    distance = geo.latlondist(aclat, aclon, plat, plon)
                    if distance < bs.settings.geofence_dlookahead:
                        # Geofence in vicinity, add the object itself to dictionary
                        geoinvicinity.append(geofence) 
                        
            # Detect conflicts with geofences
            self.GeoconfDetect(pos_ac, acid, i, trajectory, geoinvicinity)
        return
    
    def GeodetectRtree(self, ownship):
        ''' This detection method uses spacial indexing and bounding boxes. It is only accurate
        on a small scale as it basically assumes the world is flat.'''
        # Check if geofence plugin is enabled
        if 'geofenceold' not in bs.core.varexplorer.varlist:
            return 'Geofence plugin not loaded.'
        
        # Load the plugin
        geofenceplugin = bs.core.varexplorer.varlist['geofenceold'][0]
        
        # Get the necessary data from the plugin
        geofenceData = geofenceplugin.geofences
        geoidx = geofenceplugin.geoidx
        geoidx_idtogeo = geofenceplugin.geoidx_idtogeo
        
        # Skip everything if no geofences are defined.
        if not geofenceData:
            return
        
        # Create the dict
        ntraf = ownship.ntraf
        dlookahead = bs.settings.geofence_dlookahead
        
        for i in range(ntraf):
            geoinvicinity = []
            acid = ownship.id[i]
            # Get aircraft position and tile
            pos_ac = np.array([ownship.lat[i], ownship.lon[i], ownship.alt[i]])
            hdg_ac = ownship.trk[i]
            
            # Project the current position of the aircraft in the direction of the heading
            pos_next = geo.qdrpos(pos_ac[0], pos_ac[1], hdg_ac, dlookahead / nm)
            
            # Create a line for convenience, as line.bounds gives the correct format for the rtree
            trajectory = LineString(np.array([pos_ac[:2], pos_next]))
            
            # Find the list of potentially intersecting geofences
            intersectionlist = list(geoidx.intersection(trajectory.bounds))
            
            for id in intersectionlist:
                geofencename = geoidx_idtogeo[id]
                geoinvicinity.append(geofenceData[geofencename])
                
            # Detect conflicts with geofences
            self.GeoconfDetect(pos_ac, acid, i, trajectory, geoinvicinity)
        return
            
    def GeoconfDetect(self, pos_ac, acid, idx_ac, trajectory, geoinvicinity):
        ''' Detects conflicts between an aircraft an the geofences in its vicinity.
        Assumes flat earth.'''
        # Iterate over geofence objects and check if there is an intersection
        for geofence in geoinvicinity:
            # First do horizontal check. We check using shapely if projected line intersects polygon.
            geopoly = geofence.getPoly()
            
            #Check if we have breached the geofence
            pos_point = Point(pos_ac[0], pos_ac[1])
            if geopoly.contains(pos_point):   
                if acid not in self.geobreaches:
                    self.geobreaches[acid] = set()  
                    self.allgeobreaches.append((acid, geofence.name))
                    self.geobreaches[acid].add(geofence)
                    bs.traf.numgeobreaches[idx_ac] += 1
                    bs.traf.numgeobreaches_all += 1
                    return   
                          
                # Check if this breach was already there
                if geofence not in self.geobreaches[acid]:
                    self.allgeobreaches.append((acid, geofence.name))
                    self.geobreaches[acid].add(geofence)
                    bs.traf.numgeobreaches[idx_ac] += 1
                    bs.traf.numgeobreaches_all += 1   
                    return  
        return
 
    # Helper functions   
    def getAdjacentTiles(self, tileX, tileY, z):
        # We want to look inside a 3x3 grid of tiles around the aircraft
        tilelist = []
        for x in [tileX-1, tileX, tileX + 1]:
            # If x is negative, or larger than the greatest possible x, skip
            if x < 0 or x > (2**z)-1:
                continue
            
            for y in [tileY-1, tileY, tileY +1]:
                # Do the same check as before
                if y < 0 or y > (2**z)-1:
                    continue
                
                # Append tile
                tilelist.append(Tile(x, y))
        
        return tilelist
    
    def bbox_corner_tiles(self, lat1, lon1, lat2, lon2, z):
        tile_nw = self.tileXY(lat1, lon1, z)
        tile_ne = self.tileXY(lat1, lon2, z)
        tile_sw = self.tileXY(lat2, lon1, z)
        tile_se = self.tileXY(lat2, lon2, z)
        return tile_nw, tile_ne, tile_sw, tile_se

    def tile_corners(self, x, y, z):
        lat2, lon1, lat1, lon2 = self.tileEdges(x, y, z)
        corners = (lat2, lon2), (lat2, lon1), (lat1, lon1), (lat1, lon2)
        # order is SE, SW, NW, NE
        return corners

    def tileEdges(self, x, y, z):
        lat1, lat2 = self.latEdges(y, z)
        lon1, lon2 = self.lonEdges(x, z)
        return lat2, lon1, lat1, lon2  # S,W,N,E

    def tileXY(self, lat, lon, z):
        x, y = self.latlon2xy(lat, lon, z)
        return int(x), int(y)

    def latlon2xy(self, lat, lon, z):
        n = self.numTiles(z)
        x, y = self.latlon2relativeXY(lat, lon)
        return n * x, n * y

    def latEdges(self, y, z):
        n = self.numTiles(z)
        unit = 1 / n
        relY1 = y * unit
        relY2 = relY1 + unit
        lat1 = self.mercatorToLat(np.pi * (1 - 2 * relY1))
        lat2 = self.mercatorToLat(np.pi * (1 - 2 * relY2))
        return lat1, lat2

    def lonEdges(self, x, z):
        n = self.numTiles(z)
        unit = 360 / n
        lon1 = -180 + x * unit
        lon2 = lon1 + unit
        return lon1, lon2

    def xy2latlon(self, x, y, z):
        n = self.numTiles(z)
        relY = y / n
        lat = self.mercatorToLat(np.pi * (1 - 2 * relY))
        lon = -180.0 + 360.0 * x / n
        return lat, lon

    def latlon2relativeXY(self, lat, lon):
        x = (lon + 180) / 360
        y = (1 - np.log(np.tan(np.radians(lat)) + self.sec(np.radians(lat))) / np.pi) / 2
        return x, y

    @staticmethod
    def numTiles(z):
        return pow(2, z)

    @staticmethod
    def mercatorToLat(mercatorY):
        return np.degrees(np.arctan(np.sinh(mercatorY)))

    @staticmethod
    def sec(x):
        return 1 / np.cos(x)