from os import path, makedirs, remove
from PyQt5.QtOpenGL import QGLWidget
import numpy as np
import OpenGL.GL as gl
from urllib.request import urlopen

import bluesky as bs
from bluesky import settings
from .glhelpers import BlueSkyProgram, RenderObject, Font, UniformBuffer, \
    update_buffer, create_empty_buffer

# Register settings defaults
settings.set_variable_defaults(
    mpt_path='data/graphics',
    mpt_server='opentopomap',
    tile_standard='google',
    mpt_url=['https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
             'https://b.tile.opentopomap.org/{z}/{x}/{y}.png',
             'https://c.tile.opentopomap.org/{z}/{x}/{y}.png'])


class MapTiles(QGLWidget):
    """
    Default map server is from OpenTopoMap. As of March 12, 2021 data is open to use. https://opentopomap.org/about

    TO-DO List:
        -Handle Exceptions when downloading fails and others. disable map tiles if there is an error
        -Add method to clear tile directories.
        -Alter drawing functions so that png files may be loaded as textures
        -create stack command and incorporate into bluesky in smart way
        -add license text on map tiles.
        -get code to accept OSM server. OSM (https://operations.osmfoundation.org/policies/tiles/)
         requires valid HTTP-User-Agent. However, german version of OSM can still be used with
         https://a.tile.openstreetmap.de/{z}/{x}/{y}.png, https://b.tile.openstreetmap.de/{z}/{x}/{y}.png or
         https://c.tile.openstreetmap.de/{z}/{x}/{y}.png
         see tile usage policy (https://www.openstreetmap.de/germanstyle.html
         -get code to accept google tiles. https://developers.google.com/maps/documentation/tile/#tile_requests
          Google tiles require a session token before downloading data.
        -get code to accept bing map tiles:
         https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key

    Future ideas:
        -update zoom level with screen zoom.
        -parallel tile download when possible. Also figure out how to use sources with multiple servers.
        -accept other types of map tile formats. Like TMS or WMTS
        https://alastaira.wordpress.com/2011/07/06/converting-tms-tile-coordinates-to-googlebingosm-tile-coordinates/

    BBOX examples

        -Miami BBOX
            lat1 = 25.91
            lon1 = -80.45
            lat2 = 25.62
            lon2 = -80.1

        -Manhattan BBOX
            lat1 = 40.894799
            lon1 = -74.024019
            lat2 = 40.697206
            lon2 = -73.898962

    """

    def __init__(self, lat1=25.68, lon1=-80.31, lat2=25.63, lon2=-80.28, zoom_level=11, shareWidget=None,
                 LOAD_ALTERED=False, ALTER_TILE=False, INVERT=True, CONTRAST=True, con_factor=1.5, DELETE_RAW=False):
        """
        :param lat1: FLOAT, Latitude 1 of bounding box (north)
        :param lon1: FLOAT, Latitude 1 of bounding box (west)
        :param lat2: FLOAT, Latitude 2 of bounding box (south)
        :param lon2: FLOAT, Latitude 2 of bounding box (east)
        :param zoom_level: INTEGER, Zoom level for map tiles. 14 or 15 is recommended to limit number of requests
                           see the link below for size estimation of bbox:
            https://tools.geofabrik.de/calc/#type=geofabrik_standard&bbox=-80.448849,25.625192,-80.104825,25.90675
        :param shareWidget:
        :param LOAD_ALTERED: BOOLEAN, will load the altered tiles to gui
        :param ALTER_TILE: BOOLEAN, alter the tile based on INVERT and CONTRAST
        :param INVERT: BOOLEAN, invert the image so that it has ATM type look
        :param CONTRAST: BOOLEAN, increase contrast if desired
        :param con_factor: FLOAT, >1 increases contrast, <1 decreases contrast
        :param DELETE_RAW: BOOLEAN, If image is altered, the unaltered can be deleted to limit directory size.
                           Setting LOAD_ALTERED=True after will ensure that tiles are not downloaded if they are saved.
                           As long as the raw data is in directory, the images will not download.
        """

        # Map settings from CLI? maybe can be also in settings file (I think putting them in config file
        # is better for this as these settings probably won't change much)
        self.LOAD_ALTERED = LOAD_ALTERED
        self.ALTER_TILE = ALTER_TILE
        self.INVERT = INVERT
        self.CONTRAST = CONTRAST
        self.con_factor = con_factor
        self.DELETE_RAW = DELETE_RAW

        # Bounding box coordinates
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2

        # zoom level
        self.zoom_level = zoom_level

        # Initialize some variables.
        self.map_textures = []
        self.tiles = []
        self.tile_array = []
        self.local_paths = []

        # Get inheritance
        super().__init__(shareWidget=shareWidget)

        # Process setting default variables
        self.tile_dir = path.join(settings.mpt_path, settings.mpt_server)

        if settings.tile_standard == 'google':
            img_std = '{z}/{x}/{y}'

        try:
            start_index = settings.mpt_url[0].index(img_std)
            self.url_prefix = settings.mpt_url[0][:start_index]
            self.url_suffix = settings.mpt_url[0][start_index + len(img_std):]
        except ValueError:
            print('Incorrect tile format in cfg file. Please make sure tile paths is written as {z}/{x}/{y}')
        except UnboundLocalError:
            print("Incorrect tile standard in cfg file. Only accepting 'google' at the moment ")

        if 'png' in self.url_suffix:
            self.tile_format = '.png'
        else:
            self.tile_format = '.png'

        # Convert to texture. delete once you figure out how to load png into gui
        self.tex_filetype = '.dds'

    # Drawing functions below
    def tile_load(self):

        # create tile array
        self.create_tile_array()

        # download tiles
        self.process_tiles()

        for image_path in self.local_paths:
            self.map_textures.append(self.bindTexture(path.join(image_path)))

    def tile_render(self):

        for tile in self.tile_array:
            texvertices = np.array(self.tile_corners(tile[0], tile[1], self.zoom_level), dtype=np.float32)
            textexcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)
            self.tiles.append(RenderObject(gl.GL_TRIANGLE_FAN, vertex=texvertices, texcoords=textexcoords))

    def paint_map(self, main_widget):
        main_widget.texture_shader.use()

        for i in range(len(self.tile_array)):
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.map_textures[i])
            self.tiles[i].draw()

    # Non-drawing functions below
    def process_tiles(self):

        # loop through tile array
        for item in self.tile_array:

            # Create image paths, raw is unaltered image, local_path is one that is shown on screen
            img_path = path.join(str(self.zoom_level), str(item[0]), f'{item[1]}')
            raw_local_path = path.join(self.tile_dir, img_path + self.tile_format)
            alt_local_path = path.join(self.tile_dir, f'{img_path}a{self.tile_format}')
            local_path = raw_local_path

            # Download tile if it has not been downloaded
            if not path.exists(raw_local_path):

                # Check if raw data was deliberately deleted. If yes then also check that alternate file path exists
                if not self.DELETE_RAW and not path.exists(alt_local_path):
                    # create new paths, first create directories for zoom level and x
                    img_dirs = path.join(self.tile_dir, str(self.zoom_level), str(item[0]))

                    # Create directory only if it doesn't exist
                    try:
                        makedirs(img_dirs)
                    except FileExistsError:
                        pass

                # download image from web
                self.download_tile(raw_local_path, img_path)

            # Check if Altered Images should be loaded.
            if self.LOAD_ALTERED:
                # check if you want to make a new change or if path exists. If none is true
                if not path.exists(alt_local_path) or self.ALTER_TILE:
                    local_path = self.alter_tile(alt_local_path, raw_local_path)
                else:
                    local_path = alt_local_path

            # Delete raw images. This is done to limit storage.
            if self.DELETE_RAW:
                remove(raw_local_path)

            # Convert to texture, remove once you figure out how to put .png files in gui
            local_path = self.convert_to_texture(local_path)

            self.local_paths.append(local_path)

    def create_tile_array(self):
        # get tile number of corners
        tile_nw, tile_ne, tile_sw, tile_se = self.bbox_corner_tiles(self.lat1, self.lon1,
                                                                    self.lat2, self.lon2, self.zoom_level)

        # size of 2D array for tiles
        n_columns = tile_ne[0] - tile_nw[0] + 1
        n_rows = tile_sw[1] - tile_nw[1] + 1
        self.tile_array = np.zeros((n_rows, n_columns), [('x_loc', int), ('y_loc', int)])

        # fill NW corner of tile array
        self.tile_array[0, 0] = tile_nw

        for row in range(n_rows):
            # loop through rows
            for col in range(n_columns):
                # for each row loop through a column

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

        # flatten tile array
        self.tile_array = self.tile_array.flatten()

    def download_tile(self, raw_local_path, img_path):

        # download image from web
        image_url = self.url_prefix + img_path + self.url_suffix
        with open(raw_local_path, "wb") as infile:
            infile.write(urlopen(image_url).read())

    def alter_tile(self, alt_local_path, raw_local_path):

        # use pillow library for image operations
        from PIL import Image, ImageChops, ImageEnhance

        # convert image to rgb
        altered_image = Image.open(raw_local_path).convert('RGB')

        # invert image
        if self.INVERT:
            altered_image = ImageChops.invert(altered_image)

        # increase contrast, increasing contrast factor means more contrast
        if self.CONTRAST:
            enhancer = ImageEnhance.Contrast(altered_image)
            altered_image = enhancer.enhance(self.con_factor)

        altered_image.save(alt_local_path)
        return alt_local_path

    def convert_to_texture(self, local_path):
        # remove once you figure out how to put .png files in gui
        from wand import image as image_wand

        with image_wand.Image(filename=local_path) as img:
            local_path = local_path[:-4] + self.tex_filetype
            img.compression = "dxt5"
            img.save(filename=local_path)
        return local_path

    # ----------------------------------------------------------------------
    # Translates between lat/long and the slippy-map tile numbering scheme
    #
    # Code was adapted from:
    # http://wiki.openstreetmap.org/index.php/Slippy_map_tilenames
    #
    # Written by Oliver White, 2007
    # This file is public-domain
    # ----------------------------------------------------------------------

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
