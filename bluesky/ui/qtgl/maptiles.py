from os import path, makedirs, remove
from PyQt5.QtCore import Qt, QEvent, qCritical, QTimer, QT_VERSION
from PyQt5.QtOpenGL import QGLWidget
from ctypes import c_float, c_int, Strgomucture
import numpy as np
import OpenGL.GL as gl
from math import *
from urllib.request import urlopen
from PIL import Image, ImageChops, ImageEnhance
from wand import image as imga

import bluesky as bs
from bluesky import settings
from .glhelpers import BlueSkyProgram, RenderObject, Font, UniformBuffer, \
    update_buffer, create_empty_buffer

# Register settings defaults
settings.set_variable_defaults(
    gfx_path='data/graphics',
    text_size=13, apt_size=10,
    wpt_size=10, ac_size=16,
    asas_vmin=200.0, asas_vmax=500.0)


class MapTiles(QGLWidget):

    def __init__(self, shareWidget=None):
        self.map_textures = []
        self.tiles = []

        super().__init__(shareWidget=shareWidget)

        self.tile_array = []
        self.local_paths = []

        # --- info below should be defined outside class and imported into maptiles.py ---
        # Path information
        self.tex_filetype = '.dds'

        # opentopomap tilers
        self.tile_dir = 'opentopomap'
        self.url_prefix = 'https://a.tile.opentopomap.org/'
        self.url_suffix = '.png'
        self.tile_format = '.png'

        # maptiler api information
        # self.tile_dir = 'maptiler'
        # self.map_type = 'streets'
        # self.api_key = ''
        # self.url_prefix = f'https://api.maptiler.com/maps/{self.map_type}/'
        # self.url_suffix = f'.png?key={self.api_key}'
        # self.tile_format = '.png'

        # Bounding box coordinates, lat1 (north), lon1 (west), lat2 (south), lon2 (east)
        self.lat1 = 25.688843
        self.lon1 = -80.312536
        self.lat2 = 25.6366
        self.lon2 = -80.283713

        # all of miami
        # self.lat1 = 25.91
        # self.lon1 = -80.45
        # self.lat2 = 25.62
        # self.lon2 = -80.1

        # zoom level. 14/15 is recommended to limit number of requests see the link below for size estimation of city
        # see https://tools.geofabrik.de/calc/#type=geofabrik_standard&bbox=-80.448849,25.625192,-80.104825,25.90675
        # for prelim estimation. Note that it is not completely exact.
        self.zoom_level = 15

    # Drawing functions below
    def tile_load(self):

        # create tile array
        self.create_tile_array()

        # download tiles
        self.download_tiles()

        for img_path in self.local_paths:
            self.map_textures.append(self.bindTexture(path.join(img_path)))

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
    def download_tiles(self):

        # loop through tile array
        for item in self.tile_array:
            img_path = path.join(str(self.zoom_level), str(item[0]), f'{item[1]}')

            local_path = path.join(settings.gfx_path, self.tile_dir, img_path + self.tex_filetype)

            if path.exists(local_path):
                pass
            else:
                # create new paths, first create directories for zoom level and x
                img_dirs = path.join(settings.gfx_path, self.tile_dir, str(self.zoom_level), str(item[0]))

                # Create path only if it doesn't exist
                try:
                    makedirs(img_dirs)
                except FileExistsError:
                    pass

                # download image from web
                tmp_local_path = path.join(settings.gfx_path, self.tile_dir, img_path + self.tile_format)
                image_url = self.url_prefix + img_path + self.url_suffix

                with open(tmp_local_path, "wb") as infile:
                    infile.write(urlopen(image_url).read())

                # invert image
                inverted_image = ImageChops.invert(Image.open(tmp_local_path).convert('RGB'))

                # increase contrast, increasing contrast factor means more contrast
                con_factor = 1.5
                enhancer = ImageEnhance.Contrast(inverted_image)
                con_image = enhancer.enhance(con_factor)
                con_image.save(tmp_local_path)

                # Convert to texture
                with imga.Image(filename=tmp_local_path) as img:
                    img.compression = "dxt5"
                    img.save(filename=local_path)

                # delete png images
                remove(tmp_local_path)

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
        lat1 = self.mercatorToLat(pi * (1 - 2 * relY1))
        lat2 = self.mercatorToLat(pi * (1 - 2 * relY2))
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
        lat = self.mercatorToLat(pi * (1 - 2 * relY))
        lon = -180.0 + 360.0 * x / n
        return lat, lon

    def latlon2relativeXY(self, lat, lon):
        x = (lon + 180) / 360
        y = (1 - log(tan(radians(lat)) + self.sec(radians(lat))) / pi) / 2
        return x, y

    @staticmethod
    def numTiles(z):
        return pow(2, z)

    @staticmethod
    def mercatorToLat(mercatorY):
        return degrees(atan(sinh(mercatorY)))

    @staticmethod
    def sec(x):
        return 1 / cos(x)
