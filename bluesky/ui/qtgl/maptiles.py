from os import path, makedirs, remove
import numpy as np
import OpenGL.GL as gl
from urllib.request import urlopen
from urllib.error import URLError
import concurrent.futures
from PIL import Image

import bluesky as bs
from bluesky.core import Entity

from .glhelpers import RenderObject

# Register settings defaults
bs.settings.set_variable_defaults(
    mpt_path='data/graphics', mpt_server='opentopomap', enable_tiles=True,
    mpt_url='https://a.tile.opentopomap.org/{z}/{x}/{y}.png')

class MapTiles(Entity):
    """
    Default map server is from OpenTopoMap. As of March 12, 2021 data is open to use. https://opentopomap.org/about

    TODO List:
        - Texture upload in multiple threads. Wait until new qt implementation
        - Disappearing screen bug when panning maptiles
        - Vertex shader different projection
        - Relative zoom also based on image size

    Future ideas:
        - Create license text and add to image
        - Make stack command 
        - Figure out how to use sources with multiple servers.
        - Accept other types of map tile formats. Like TMS or WMTS
        https://alastaira.wordpress.com/2011/07/06/converting-tms-tile-coordinates-to-googlebingosm-tile-coordinates/

    'osm' tile standard: url contains tile path "{z}/{x}/{y}.png". Example of sources that use osm standard:
        - OpenTopoMap: https://opentopomap.org/
            - This is the default map tiler for BlueSky. As of March 15, 2021 the tiles are open for download and use
              as long as credit is given to OpenTopoMap. Please refer to https://opentopomap.org/about.
              OpenTopoMap tries to update tiles every 4 weeks.
        - OpenStreetMap: https://www.openstreetmap.org/
            - At the moment OpenStreetMap can only be accessed with a Valid User-Agent. Please contact OSM
              to get a valid HTTP User-Agent and alter the download_tile() method so that a valid User-Agent
              can be passed. Refer to tile usage policy: https://operations.osmfoundation.org/policies/tiles/
        - OpenStreetMap Germany: https://www.openstreetmap.de/
            - Unlike the regular OpenStreetMap, OSM germany does not require a valid HTTP User-Agent. Please refer
              to the tile usage policy: https://www.openstreetmap.de/germanstyle.html.
        - maptiler: https://www.maptiler.com/
            - Maptiler is a commercial product. Please refer to https://www.maptiler.com/cloud/terms/ for terms of
              service.
        - local_host:
            - Create your own tiles and serve them locally. Learn how to generate tiles at https://openmaptiles.org/.
               Use tileserver-GL https://tileserver.readthedocs.io/en/latest/index.html to render and serve the tiles.
               Url example is: http://localhost:8080/styles/basic-preview/{z}/{x}/{y}.png

    Other resources
        - information on tile storage info based on bbox.
          https://tools.geofabrik.de/calc/#type=geofabrik_standard&bbox=-80.448849,25.625192,-80.104825,25.90675

    """

    def __init__(self, radar_widget):
        """
        :param radar_widget: bring in radar widget to get shaders and screen information.
        """
        # Inherit from BlueSky Entity
        super().__init__()

        # radar widget used to bring in shaders and screen size
        self.radar_widget = radar_widget

        # Process settings.cfg
        self.enable_tiles = bs.settings.enable_tiles

        # create tile directory path
        self.tile_dir = path.join(bs.settings.mpt_path, bs.settings.mpt_server)

        # check for errors in config file url and set url_prefix and url_suffix for downloading of tiles
        img_std = '{z}/{x}/{y}'

        try:
            start_index = bs.settings.mpt_url.index(img_std)
            self.url_prefix = bs.settings.mpt_url[:start_index]
            self.url_suffix = bs.settings.mpt_url[start_index + len(img_std):]
            if 'png' in self.url_suffix:
                self.tile_format = '.png'
            else:
                print('Only accepting PNG formats at the moment')
                print('Failed to load map tiles!!')
                self.enable_tiles = False
        except ValueError:
            # this just checks if the tile image standard for downloading is set to {z}/{x}/{y}
            print(f'Incorrect tile format in cfg file. Please make sure url contains {img_std}')
            print('Failed to load map tiles!!')
            self.enable_tiles = False
        

        # Bounding box coordinates and zoom level initialization
        # lat1 (north), lon1(west), lat2(south), lon2(east), zoom_level (map tile zoom level)
        self.lat1 = None
        self.lon1 = None
        self.lat2 = None
        self.lon2 = None
        self.zoom_level = None

        # Initialize other variables
        self.map_textures = []
        self.tiles = []
        self.tile_array = []
        self.tile_offset = []
        self.local_paths = []
        self.local_paths_offset = []
        self.tex_columns = 0
        self.tex_rows = 0
        self.tex_width = 0
        self.tex_height = 0
        self.tile_size = 0
        self.bbox_corners = None

    ''' Drawing functions start here '''

    def tile_reload(self):
        # Tile reloading from screen bbox. Note that tiles are really only good for a zoom level greater than 8.
        # Anything smaller appears very deformed due to difference in projections of bluesky and maptiles.

        # get screen zoom
        screen_zoom = self.radar_widget.zoom

        # clear everything to start fresh
        self.clear_tiles()

        # Get screen bbox coordinates for tile downloading
        self.lat1, self.lon1 = self.radar_widget.pixelCoordsToLatLon(0, 0)
        self.lat2, self.lon2 = self.radar_widget.pixelCoordsToLatLon(self.radar_widget.width, self.radar_widget.height)

        # simple screen width factor. 478 is number of pixels in which the zoom factors were developed. It was also developed for a tile that is 512x512 px
        screen_factor = (self.radar_widget.width) / 909
        zoom_array = np.array([1.4, 2.3, 4.6, 9.1, 18.5, 37.0, 73.0, 150.0, 410.0, 820.0, 1640.0, 2320.0]) / screen_factor

        # Get zoom level based on screen level.
        if screen_zoom < zoom_array[0]: 
            self.zoom_level = 8
        elif screen_zoom < zoom_array[1]:
            self.zoom_level = 9
        elif screen_zoom < zoom_array[2]:
            self.zoom_level = 10
        elif screen_zoom < zoom_array[3]:
            self.zoom_level = 11
        elif screen_zoom < zoom_array[4]:
            self.zoom_level = 12
        elif screen_zoom < zoom_array[5]:
            self.zoom_level = 13
        elif screen_zoom < zoom_array[6]:
            self.zoom_level = 14
        elif screen_zoom < zoom_array[7]:
            self.zoom_level = 15
        elif screen_zoom < zoom_array[8]:
            self.zoom_level = 16
        elif screen_zoom < zoom_array[9]:
            self.zoom_level = 17
        elif screen_zoom < zoom_array[10]:
            self.zoom_level = 18
        elif screen_zoom < zoom_array[11]:
            self.zoom_level = 19
        else:
            self.zoom_level = 20

        # Load tiles
        self.tile_load()

    def tile_load(self):
        # create a tile array and texture offset values
        self.create_tile_array()

        # process tiles, download tiles if necessary
        self.process_tiles()

        # Bind texture aray and give image data. 
        
        # Calculate pixel size in a single tile by opening one image and checking. (square tiles)
        self.tile_size = Image.open(self.local_paths[0]).height

        # Calculate pixel size needed for the texture array
        self.tex_width = self.tile_size * self.tex_columns
        self.tex_height = self.tile_size * self.tex_rows

        # Create texture and bind.
        # For some reason code only works when adding texture name to a list
        texture = gl.glGenTextures(1)
        self.map_textures.append(texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.map_textures[0])

        # Specify storage needed for the texture array
        gl.glTexStorage3D(gl.GL_TEXTURE_2D_ARRAY, 1, gl.GL_RGB8, self.tex_width, self.tex_height, 1)

        # Loop to send image data to texture array.
        # TODO: figure out faster way to send img_data byte information (perhaps in multiple threads) or maybe QT does this?
        for item in self.local_paths_offset:
            image_path = item[0]
            offset_y = item[1][0] * self.tile_size
            offset_x = item[1][1] * self.tile_size

            # open image, get byte information and specify subimage of texture array
            image = Image.open(image_path)
            img_data = image.tobytes()
            gl.glTexSubImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, offset_x, offset_y, 0, self.tile_size, self.tile_size, 1,
                                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_data)

        # Set texture parameters
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D_ARRAY)
        gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

    def tile_render(self):

        # Create vertex array object for the 2D texture array
        texvertices = np.array(self.bbox_corners, dtype=np.float32)
        textexcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)
        self.tiles.append(RenderObject(gl.GL_TRIANGLE_FAN, vertex=texvertices, texcoords=textexcoords))

    def paint_map(self):

        # Use maptile shader for texture array and have one draw call
        self.radar_widget.maptile_shader.use()

        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.map_textures[0])
        self.tiles[0].draw()

    def clear_tiles(self):
        # unbind and delete textures from memory
        for i in range(len(self.tiles)):
            self.tiles[i].unbind_all()

        gl.glDeleteTextures(len(self.map_textures), self.map_textures)

        # Clear variables for new tiles
        self.local_paths = []
        self.tiles = []
        self.map_textures = []
        self.local_paths_offset = []
        self.tile_offset = []

    ''' Non-drawing functions start here '''

    def process_tiles(self):
        # Download tiles in multiple threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.download_tile, self.tile_array)

        # Create local path names so texture array can access them
        for item in self.tile_array:
            # create image paths
            x = item[0]  # tile x
            y = item[1]  # tile y
            img_path = path.join(str(self.zoom_level), str(x), str(y))
            local_path = path.join(self.tile_dir, img_path + self.tile_format)

            # create array of local paths for texture array
            self.local_paths.append(local_path)

        # Create a list with tuples of local paths and their corresponding texture array offset
        self.local_paths_offset = list(zip(self.local_paths, self.tile_offset))

    def create_tile_array(self):
        # get tile number of corners of bounding box
        tile_nw, tile_ne, tile_sw, tile_se = self.bbox_corner_tiles(self.lat1, self.lon1,
                                                                    self.lat2, self.lon2, self.zoom_level)

        # Get bbox corner lat/lon for rendering (texvertices). This is usually a larger bounding box than screen
        self.bbox_corners = self.bbox_latlon(tile_nw, tile_ne, tile_sw, tile_se, self.zoom_level)

        # Find size of a 2D array for tiles and create the array.
        # tex_columns an tex_rows are the number of columns and rows, respectively, in 2D array.
        # This 2D array has the same size as the texture array that will be rendered by OpenGL.
        # tile_array contains a (x,y) tuple with tile numbers (slippy-map tile number)
        self.tex_columns = tile_ne[0] - tile_nw[0] + 1
        self.tex_rows = tile_sw[1] - tile_nw[1] + 1
        self.tile_array = np.zeros((self.tex_rows, self.tex_columns), [('x_loc', int), ('y_loc', int)])

        # fill NW corner of tile array to give a starting point for 2D array
        self.tile_array[0, 0] = tile_nw

        # Loop through each element of the array and fill with corresponding tile number
        # loop through rows. x is constant in the same row
        for row in range(self.tex_rows):
            # for each row loop through a column. y is constant in the same column.
            for col in range(self.tex_columns):
                if col == 0:
                    # change first column of the row per the first column of row above
                    if row != 0:
                        # don't change first entry (0,0) because that is already in 2d array
                        self.tile_array[row, col] = (self.tile_array[row - 1, col][0],
                                                     self.tile_array[row - 1, col][1] + 1)

                else:
                    # change column of the row based on previous column of that same row
                    self.tile_array[row, col] = (self.tile_array[row, col - 1][0] + 1,
                                                 self.tile_array[row, col - 1][1])

                # Create list of tile offsets of texture array. 
                # These values tell you how much to offset (row, col) the tile from the top left corner of texture array
                self.tile_offset.append((row, col))

        # flatten tile array
        self.tile_array = self.tile_array.flatten()

    def download_tile(self, tile):
        # Create paths from tile zoom, x, y
        # osm tile_format downloads have the same image path as local path. 
        # Note that url_img_path is a string due to difference in path strings per operating system. "/" vs "\"
        x = tile[0]  # tile x
        y = tile[1]  # tile y
        img_path = path.join(str(self.zoom_level), str(x), str(y))
        url_img_path = f'{str(self.zoom_level)}/{str(x)}/{str(y)}'
        local_path = path.join(self.tile_dir, img_path + self.tile_format)

        # Only go into loop if the full_local_path does not exist. If previously downloaded, it will ignore
        if not path.exists(local_path):

            # create new path, first create directories for zoom level and x
            img_dirs = path.join(self.tile_dir, str(self.zoom_level), str(x))

            # Create directory only if it doesn't exist
            try:
                makedirs(img_dirs)
            except FileExistsError:
                pass
            
            # Create image url for downloading
            image_url = self.url_prefix + url_img_path + self.url_suffix

            # request tile from server and save to full_local_path
            with open(local_path, "wb") as infile:
                try:
                    url_request = urlopen(image_url)
                    infile.write(url_request.read())
                except URLError:
                    print(f'Failed to download tiles. Ensure that url is valid: {image_url}')
                    remove(local_path)

            # Convert to RGB for OpenGL and save
            img = Image.open(local_path).convert('RGB')
            img.save(local_path)

    '''
    ----------------------------------------------------------------------
    Translates between lat/long and the slippy-map tile numbering scheme
    
    Code was adapted from:
    http://wiki.openstreetmap.org/index.php/Slippy_map_tilenames

    Written by Oliver White, 2007
    This file is public-domain
    ----------------------------------------------------------------------
    '''

    def bbox_latlon(self, tile_nw, tile_ne, tile_sw, tile_se, z):
        # NW Tuple
        _, _, nw, _ = self.tile_corners(tile_nw[0], tile_nw[1], z)
        # SE Tuple
        se, _, _, _ = self.tile_corners(tile_se[0], tile_se[1], z)
        # NE Tuple
        _, _, _, ne = self.tile_corners(tile_ne[0], tile_ne[1], z)
        # SW Tuple
        _, sw, _, _ = self.tile_corners(tile_sw[0], tile_sw[1], z)
        return se, sw, nw, ne

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
