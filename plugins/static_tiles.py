""" Plugin that loads maptiles for a given bounding box and zoom level.
The tiles are loaded into a texture array and drawn on screen."""
import numpy as np
import traceback
import math
from os import makedirs, path, remove
import weakref
from collections import OrderedDict
from urllib.request import urlopen
from urllib.error import URLError

from PyQt5.Qt import Qt
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage
import concurrent.futures
from PIL import Image

from bluesky import stack, ui, settings, navdb, sim, scr, tools
from bluesky.ui import gl
from bluesky.ui.qtgl import glhelpers as glh

# get variable defaults
settings.set_variable_defaults(
    tilesource='opentopomap',
    tile_sources={
        'opentopomap': {
            'source': ['https://a.tile.opentopomap.org/{zoom}/{x}/{y}.png',
                       'https://b.tile.opentopomap.org/{zoom}/{x}/{y}.png',
                       'https://c.tile.opentopomap.org/{zoom}/{x}/{y}.png'],
            'max_download_workers': 2,
            'max_tile_zoom': 17,
            'license': 'map data: © OpenStreetMap contributors, SRTM | map style: © OpenTopoMap.org (CC-BY-SA)'},
        'cartodb': {
            'source': ['https://cartodb-basemaps-b.global.ssl.fastly.net/light_nolabels/{zoom}/{x}/{y}.png'],
            'max_tile_zoom': 20,
            'license': 'CartoDB Grey and white, no labels'
        },
        'nasa': {
            'source': ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}.jpg'],
            'max_tile_zoom': 13,
            'license': 'Satellite images from NASA via ESRI'
        }
    })

# initialize the plugin
def init_plugin():
    ''' Plugin initialisation function. '''
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'STATIC_TILES',

        # The type of this plugin.
        'plugin_type':     'gui',
        }

    return config

# create stack command that accepts tile source, bounding box and zoom level
@stack.command
def statictiles(lat1: float=48.3255, lon1: float=16.1515, lat2: float=48.0931, 
                lon2: float=16.5865, zoom: int=14, source: str='opentopomap'):
    '''STATICTILES 48.3255 16.1515 48.0931 16.5865 14 opentopomap'''
    # set arguments as global variables
    global lat1_stack, lon1_stack, lat2_stack, lon2_stack, zoom_stack, source_stack

    lat1_stack = lat1
    lon1_stack = lon1
    lat2_stack = lat2
    lon2_stack = lon2
    zoom_stack = zoom
    source_stack = source


class MapTiles(ui.RenderObject, layer=100):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.maptiles = glh.VertexArrayObject(glh.gl.GL_TRIANGLE_FAN)

        img_std = '{zoom}/{x}/{y}'

        # get url source
        tile_source = source_stack
        url_source = settings.tile_sources[tile_source]['source'][0]

        start_index = url_source.index(img_std)
        self.url_prefix = url_source[:start_index]
        self.url_suffix = url_source[start_index + len(img_std):]
        self.tile_format = '.png'

        self.enable_tiles = False

        self.lat1 = lat1_stack
        self.lon1 = lon1_stack
        self.lat2 = lat2_stack
        self.lon2 = lon2_stack

        self.zoom_level = zoom_stack

        self.map_textures = []
        self.tiles = []
        self.tile_array = []
        self.render_corners = []
        self.tile_offset = []
        self.local_paths = []
        self.local_paths_offset = []
        self.tex_columns = 0
        self.tex_rows = 0
        self.tex_width = 0
        self.tex_height = 0
        self.tile_size = 0
        self.bbox_corners = None

        # set some directories
        self.cache_path = settings.cache_path
        self.tile_dir = path.join(self.cache_path, tile_source)


    def create(self):

        # Create vertex array object for the 2D texture array
        mapvertices = np.array(self.bbox_corners, dtype=np.float32)
        texcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)

        self.maptiles.create(vertex=mapvertices, texcoords=texcoords, texture=fname)

        if self.array_load:
            # Vertex array object for the 2D texture array
            mapvertices = np.array(self.bbox_corners, dtype=np.float32)
            texcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)
            self.maptiles.create(vertex=texvertices, texcoords=textexcoords))
        else:
            # Vertex array object for each individual texture
            for corner in self.render_corners:
                texvertices = np.array(corner, dtype=np.float32)
                textexcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)
                self.maptiles.append(RenderObject(gl.GL_TRIANGLE_FAN, vertex=texvertices, texcoords=textexcoords))

    def draw(self):
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)
        if self.array_load:
            # Use maptile shader for texture array and have one draw call
            self.radar_widget.maptile_shader.use()

            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.map_textures[0])
            self.maptiles[0].draw()

        else:
            # Use radar widget shader for each individual texture and have one draw call per texture
            self.radar_widget.texture_shader.use()
            for i in range(len(self.tile_array)):
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.map_textures[i])
                self.maptiles[i].draw()
            
    def clear_tiles(self):
        # unbind and delete textures from memory
        for i in range(len(self.maptiles)):
            self.tiles[i].unbind_all()

        gl.glDeleteTextures(len(self.map_textures), self.map_textures)

        # Clear variables for new tiles
        self.local_paths = []
        self.tiles = []
        self.map_textures = []
        self.local_paths_offset = []
        self.tile_offset = []

        self.array_load = False


    def tile_load(self):
        # create a tile array and texture offset values
        self.create_tile_array()

        # process tiles, download tiles if necessary
        self.process_tiles()

        if self.array_load:
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
        
        else:

            for image_path in self.local_paths:
                texture = gl.glGenTextures(1)
                self.map_textures.append(texture)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                image = Image.open(image_path)
                img_data = image.tobytes()
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.width, image.height, 0, gl.GL_RGB,
                                gl.GL_UNSIGNED_BYTE, img_data)

                # Texture parameters
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_LINEAR)


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

# write opengl vertex and fragment shader
vertex_shader = """
#version 330
#define VERTEX_IS_LATLON 0

// Uniform block of global data
layout (std140) uniform global_data {
int wrap_dir;           // Wrap-around direction
float wrap_lon;         // Wrap-around longitude
float panlat;           // Map panning coordinates [deg]
float panlon;           // Map panning coordinates [deg]
float zoom;             // Screen zoom factor [-]
int screen_width;       // Screen width in pixels
int screen_height;      // Screen height in pixels
};

layout (location = 0) in vec2 vertex_in;
layout (location = 1) in vec2 texcoords_in;
layout (location = 2) in vec3 a_color;

out vec2 texcoords_fs;
out vec3 v_color;

void main()
{

	vec2 vAR = vec2(1.0, float(screen_width) / float(screen_height));
	vec2 flat_earth = vec2(cos(radians(panlat)), 1.0);

	// Vertex position and rotation calculations
	vec2 position = vec2(0, 0);
	position -= vec2(panlon, panlat);

	// Lat/lon vertex coordinates are flipped: lat is index 0, but screen y-axis, and lon is index 1, but screen x-axis
	gl_Position = vec4(vAR * flat_earth * zoom * (position + vertex_in.yx), 0.0, 1.0);
	texcoords_fs = texcoords_in.ts;

	// color to fragment shader
	v_color = a_color;
}
"""

fragment_shader = """
#version 330
 
// Interpolated values from the vertex shaders
in vec2 texcoords_fs;
in vec3 v_color;

// Ouput data
out vec4 color;
 
// Values that stay constant for the whole mesh.
uniform sampler2DArray tex_sampler;
 
void main()
{ 
    // Output color = color of the texture at the specified UV
    color = texture(tex_sampler, vec3(texcoords_fs, 0.0));
}
"""

