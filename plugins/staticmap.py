""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, ui  #, settings, navdb, sim, scr, tools
from bluesky.ui import gl

from os import path, makedirs, remove
from urllib.request import urlopen
from urllib.error import URLError
import numpy as np

import concurrent.futures
from PIL import Image

import bluesky as bs


# Register settings defaults
bs.settings.set_variable_defaults(
    mpt_path='data/graphics', mpt_server='opentopomap', tile_standard='osm',
    enable_tiles=True,
    mpt_url=['https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
             'https://b.tile.opentopomap.org/{z}/{x}/{y}.png',
             'https://c.tile.opentopomap.org/{z}/{x}/{y}.png'],
    lat1=52.47, lon1=4.69, lat2=52.24, lon2=5.0, zoom_level=8)


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'STATICMAP',

        # The type of this plugin.
        'plugin_type':     'gui',
        }

    # init_plugin() should always return a configuration dict.
    return config

class MyVisual(ui.RenderObject, layer=100):
    ''' Example new render object for BlueSky. '''
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.map_tiles = MapTiles()

        self.shape = ui.VertexArrayObject(gl.GL_TEXTURE_2D_ARRAY)

        self.map_tiles.tile_load()


    def create(self):
        self.map_tiles.tile_render()

    def draw(self):
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)
        self.map_tiles.paint_map()


class MapTiles():
    """
    BBOX examples:

        -Miami BBOX:
            lat1 = 25.91
            lon1 = -80.45
            lat2 = 25.62
            lon2 = -80.1

        -Manhattan BBOX:
            lat1 = 40.894799
            lon1 = -74.024019
            lat2 = 40.697206
            lon2 = -73.898962

        -Amsterdam BBOX:
            lat1 = 52.47
            lon1 = 4.69
            lat2 = 52.24
            lon2 = 5.0

    tile_standards:
        -'osm' standard: url contains tile path "{z}/{x}/{y}.png"
         Example of sources that use osm standard:
            -OpenTopoMap: https://opentopomap.org/
                -This is the default map tiler for BlueSky. As of March 15, 2021 the tiles are open for download and use
                 as long as credit is given to OpenTopoMap. Please refer to https://opentopomap.org/about.
                 OpenTopoMap tries to update tiles every 4 weeks.
            -OpenStreetMap: https://www.openstreetmap.org/
                -At the moment OpenStreetMap can only be accessed with a Valid User-Agent. Please contact OSM
                 to get a valid HTTP User-Agent and alter the download_tile() method so that a valid User-Agent
                can be passed. Refer to tile usage policy: https://operations.osmfoundation.org/policies/tiles/
            -OpenStreetMap Germany: https://www.openstreetmap.de/
                -Unlike the regular OpenStreetMap, OSM germany does not require a valid HTTP User-Agent. Please refer
                 to the tile usage policy: https://www.openstreetmap.de/germanstyle.html.
            -maptiler: https://www.maptiler.com/
                -Maptiler is a commercial product. Please refer to https://www.maptiler.com/cloud/terms/ for terms of
                 service.
            -local_host:
                -Create your own tiles and serve them locally. Learn how to generate tiles at https://openmaptiles.org/.
                 Use tileserver-GL https://tileserver.readthedocs.io/en/latest/index.html to render and serve the tiles.
                 Url example is: http://localhost:8080/styles/basic-preview/{z}/{x}/{y}.png
    """

    def __init__(self):
        """
        :param lat1: FLOAT, Latitude 1 of bounding box (north)
        :param lon1: FLOAT, Longitude 1 of bounding box (west)
        :param lat2: FLOAT, Latitude 2 of bounding box (south)
        :param lon2: FLOAT, Longitude 2 of bounding box (east)
        :param zoom_level: INTEGER, Zoom level for map tiles. 14 or 15 is recommended to limit number of requests
                           see the link below for size estimation of bbox:
            https://tools.geofabrik.de/calc/#type=geofabrik_standard&bbox=-80.448849,25.625192,-80.104825,25.90675
        """

        # Bounding box coordinates and zoom level used if dynamic tiles is off.
        self.lat1 = bs.settings.lat1
        self.lon1 = bs.settings.lon1
        self.lat2 = bs.settings.lat2
        self.lon2 = bs.settings.lon2
        self.zoom_level = bs.settings.zoom_level

        # Initialize some variables
        self.map_textures = []
        self.tiles = []
        self.tile_array = []
        self.tile_offset = []
        self.local_paths = []
        self.local_paths_offset = []
        self.enable_tiles = bs.settings.enable_tiles
        self.download_fail = False
        self.tex_columns = 0
        self.tex_rows = 0
        self.tex_width = 0
        self.tex_height = 0
        self.tile_size = 0
        self.bbox_corners = None

        # create tile directory path
        self.tile_dir = path.join(bs.settings.mpt_path,'tiles', bs.settings.mpt_server)

        # check for errors in config file url and set url_prefix and url_suffix for downloading of tiles
        img_std = '{z}/{x}/{y}'

        try:
            start_index = bs.settings.mpt_url[0].index(img_std)
            self.url_prefix = bs.settings.mpt_url[0][:start_index]
            self.url_suffix = bs.settings.mpt_url[0][start_index + len(img_std):]
            self.tile_format = '.png'
        except ValueError:
            # this just checks if the tile image standard for downloading is set to {z}/{x}/{y}
            print(f'Incorrect tile format in cfg file. Please make sure url contains {img_std}')
            print('Failed to load map tiles!!')
            self.enable_tiles = False
        except UnboundLocalError:
            # this just checks if google standard was set as the default standard. Later this will have to be edited
            print("Incorrect tile standard in cfg file. Tile standard must be 'osm'")
            print('Failed to load map tiles!!')
            self.enable_tiles = False

    # Drawing functions start here
    def tile_load(self):
        # create a tile array
        self.create_tile_array()


        # process tiles, download tiles if necessary
        self.process_tiles()

        print(self.bbox_corners)
        # Bind tiles as textures.
        # Texture arrays binding code
        # Number of pixels in tile. Open one image and check
        self.tile_size = Image.open(self.local_paths[0]).height

        # number of tiles in texture
        self.tex_width = self.tile_size * self.tex_columns
        self.tex_height = self.tile_size * self.tex_rows
        texture = gl.glGenTextures(1)

        self.map_textures.append(texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.map_textures[0])
        gl.glTexStorage3D(gl.GL_TEXTURE_2D_ARRAY, 1, gl.GL_RGB8, self.tex_width, self.tex_height, 1)

        # send image data to texture array.
        for item in self.local_paths_offset:
            image_path = item[0]
            offset_y = item[1][0] * self.tile_size
            offset_x = item[1][1] * self.tile_size

            image = Image.open(image_path)
            img_data = image.tobytes()
            gl.glTexSubImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, offset_x, offset_y, 0, self.tile_size, self.tile_size, 1,
                                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_data)

        # Set texture parameters
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D_ARRAY)
        gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)


    def tile_render(self):

        # Vertex array object for the 2D texture array
        texvertices = np.array(self.bbox_corners, dtype=np.float32)
        textexcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)
        self.tiles.append(RenderObject(gl.GL_TRIANGLE_FAN, vertex=texvertices, texcoords=textexcoords))

    def paint_map(self):

        # Use maptile shader for texture array and have one draw call
        self.radar_widget.maptile_shader.use()

        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.map_textures[0])
        self.tiles[0].draw()

    # Non-drawing functions from here on
    def process_tiles(self):
        # Download tiles in multiple threads. If download fails self.enable_tiles = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.download_tile, self.tile_array)

        # Create local path names and get corner info
        for item in self.tile_array:
            x = item[0]  # tile x
            y = item[1]  # tile y

            img_path = path.join(str(self.zoom_level), str(x), str(y))
            local_path = path.join(self.tile_dir, img_path + self.tile_format) 

            # create arrays of local paths for later use
            self.local_paths.append(local_path)

        # Create local path dictionary with image offsets
        self.local_paths_offset = list(zip(self.local_paths, self.tile_offset))

    def create_tile_array(self):
        # get tile number of corners of bounding box
        tile_nw, tile_ne, tile_sw, tile_se = bbox_corner_tiles(self.lat1, self.lon1,
                                                               self.lat2, self.lon2, self.zoom_level)

        # Get bbox corner lat/lon for rendering
        self.bbox_corners = bbox_latlon(tile_nw, tile_ne, tile_sw, tile_se, self.zoom_level)

        # Find size of 2D array for tiles
        self.tex_columns = tile_ne[0] - tile_nw[0] + 1
        self.tex_rows = tile_sw[1] - tile_nw[1] + 1
        self.tile_array = np.zeros((self.tex_rows, self.tex_columns), [('x_loc', int), ('y_loc', int)])

        # fill NW corner of tile array
        self.tile_array[0, 0] = tile_nw

        # loop through rows. x is constant in one row
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

                # create tile offsets of texture array to create list of tuples with the local_path
                self.tile_offset.append((row, col))

        # flatten tile array
        self.tile_array = self.tile_array.flatten()

    def download_tile(self, tile):
        # Only run loop while downloading did not fail. if download tiles fails self.enable_tiles is set to false
        if self.enable_tiles:

            # Create image paths, raw is unaltered image, local_path is one that is shown on screen
            x = tile[0]  # tile x
            y = tile[1]  # tile y
            img_path = path.join(str(self.zoom_level), str(x), str(y))
            raw_local_path = path.join(self.tile_dir, img_path + self.tile_format)

            # Download tile if it has not been downloaded
            if not path.exists(raw_local_path):
                # create new paths, first create directories for zoom level and x
                img_dirs = path.join(self.tile_dir, str(self.zoom_level), str(x))

                # Create directory only if it doesn't exist
                try:
                    makedirs(img_dirs)
                except FileExistsError:
                    pass

                # download image from web
                url_img_path = f'{str(self.zoom_level)}/{str(x)}/{str(y)}'
                image_url = self.url_prefix + url_img_path + self.url_suffix

                # request
                with open(raw_local_path, "wb") as infile:
                    try:
                        url_request = urlopen(image_url)
                        infile.write(url_request.read())
                    except URLError:
                        print(f'Failed to download tiles. Ensure that url is valid: {image_url}')
                        remove(raw_local_path)

                        # set some variables to cancel map tile loop
                        self.enable_tiles = False

                # Convert to RGBA
                img = Image.open(raw_local_path).convert('RGB')
                img.save(raw_local_path)

# ----------------------------------------------------------------------
# Translates between lat/long and the slippy-map tile numbering scheme
#
# Code was adapted from:
# http://wiki.openstreetmap.org/index.php/Slippy_map_tilenames
#
# Written by Oliver White, 2007
# This file is public-domain
# ----------------------------------------------------------------------

def bbox_latlon(tile_nw, tile_ne, tile_sw, tile_se, z):
    # NW Tuple
    _, _, nw, _ = tile_corners(tile_nw[0], tile_nw[1], z)
    # SE Tuple
    se, _, _, _ = tile_corners(tile_se[0], tile_se[1], z)
    # NE Tuple
    _, _, _, ne = tile_corners(tile_ne[0], tile_ne[1], z)
    # SW Tuple
    _, sw, _, _ = tile_corners(tile_sw[0], tile_sw[1], z)
    return se, sw, nw, ne

def bbox_corner_tiles(lat1, lon1, lat2, lon2, z):
    tile_nw = tileXY(lat1, lon1, z)
    tile_ne = tileXY(lat1, lon2, z)
    tile_sw = tileXY(lat2, lon1, z)
    tile_se = tileXY(lat2, lon2, z)
    return tile_nw, tile_ne, tile_sw, tile_se

def tile_corners(x, y, z):
    lat2, lon1, lat1, lon2 = tileEdges(x, y, z)
    corners = (lat2, lon2), (lat2, lon1), (lat1, lon1), (lat1, lon2)
    # order is SE, SW, NW, NE
    return corners

def tileEdges(x, y, z):
    lat1, lat2 = latEdges(y, z)
    lon1, lon2 = lonEdges(x, z)
    return lat2, lon1, lat1, lon2  # S,W,N,E

def tileXY(lat, lon, z):
    x, y = latlon2xy(lat, lon, z)
    return int(x), int(y)

def latlon2xy(lat, lon, z):
    n = numTiles(z)
    x, y = latlon2relativeXY(lat, lon)
    return n * x, n * y

def latEdges(y, z):
    n = numTiles(z)
    unit = 1 / n
    relY1 = y * unit
    relY2 = relY1 + unit
    lat1 = mercatorToLat(np.pi * (1 - 2 * relY1))
    lat2 = mercatorToLat(np.pi * (1 - 2 * relY2))
    return lat1, lat2

def lonEdges(x, z):
    n = numTiles(z)
    unit = 360 / n
    lon1 = -180 + x * unit
    lon2 = lon1 + unit
    return lon1, lon2

def xy2latlon(x, y, z):
    n = numTiles(z)
    relY = y / n
    lat = mercatorToLat(np.pi * (1 - 2 * relY))
    lon = -180.0 + 360.0 * x / n
    return lat, lon

def latlon2relativeXY(lat, lon):
    x = (lon + 180) / 360
    y = (1 - np.log(np.tan(np.radians(lat)) + sec(np.radians(lat))) / np.pi) / 2
    return x, y

def numTiles(z):
    return pow(2, z)

def mercatorToLat(mercatorY):
    return np.degrees(np.arctan(np.sinh(mercatorY)))

def sec(x):
    return 1 / np.cos(x)
