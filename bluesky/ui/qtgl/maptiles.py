from os import path, makedirs, remove
from PyQt5.QtOpenGL import QGLWidget
import numpy as np
import OpenGL.GL as gl
from urllib.request import urlopen
from urllib.error import URLError
import requests
import concurrent.futures

import bluesky as bs
from bluesky import settings
from .glhelpers import BlueSkyProgram, RenderObject, Font, UniformBuffer, \
    update_buffer, create_empty_buffer

# Register settings defaults
settings.set_variable_defaults(
    mpt_path='data/graphics', mpt_server='opentopomap', tile_standard='google',
    mpt_url=['https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
             'https://b.tile.opentopomap.org/{z}/{x}/{y}.png',
             'https://c.tile.opentopomap.org/{z}/{x}/{y}.png'],
    LOAD_ALTERED=False, ALTER_TILE=False, INVERT=True, CONTRAST=True,
    con_factor=1.5, enable_tiles=False, dynamic_tiles=False,
    lat1=25.68, lon1=-80.31, lat2=25.63, lon2=-80.28, zoom_level=8)


class MapTiles(QGLWidget):
    """
    Default map server is from OpenTopoMap. As of March 12, 2021 data is open to use. https://opentopomap.org/about

    TO-DO List:
        -Fix image operation settings.
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
        -"bing" standard: url contains mapArea and zoomlevel as "?mapArea={map_area}&zoomlevel={zoom_level}"
            -Bing Maps is a commercial product. Note that Bing Maps uses a similar tile setup as osm. The difference is
             just in how the request is made. There are several ways to make a request, see:
             https://docs.microsoft.com/en-us/bingmaps/rest-services/imagery/get-a-static-map#pushpin-limits. The url
             that BlueSky works with is as follows:
             "https://dev.virtualearth.net/REST/v1/Imagery/Map/Road?mapArea={mapArea}&zoomlevel={zoomlevel}&fmt=png&key={key}"
             This is done so that tiles are pulled in a similar way. Please refer to the Bing Maps API Terms of Use.
             https://www.microsoft.com/en-us/maps/product.
    """

    def __init__(self, lat1=settings.lat1, lon1=settings.lon1, lat2=settings.lat2, lon2=settings.lon2,
                 zoom_level=settings.zoom_level, shareWidget=None):
        """
        :param lat1: FLOAT, Latitude 1 of bounding box (north)
        :param lon1: FLOAT, Longitude 1 of bounding box (west)
        :param lat2: FLOAT, Latitude 2 of bounding box (south)
        :param lon2: FLOAT, Longitude 2 of bounding box (east)
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
        # Bounding box coordinates
        self.lat1 = lat1
        self.lon1 = lon1
        self.lat2 = lat2
        self.lon2 = lon2

        # zoom level
        self.zoom_level = zoom_level

        # Initialize some variables
        self.map_textures = []
        self.tiles = []
        self.tile_array = []
        self.render_corners = []
        self.local_paths = []
        self.enable_tiles = settings.enable_tiles
        self.download_fail = False
        # Check if map is dynamic. Tiles change with zoom level
        self.dynamic_tiles = settings.dynamic_tiles

        # Get inheritance
        super().__init__(shareWidget=shareWidget)

        # Process setting default variables

        # create tile directory
        self.tile_dir = path.join(settings.mpt_path, settings.mpt_server)

        # check for errors in config file url and set url_prefix and url_suffix for downloading of tiles
        if settings.tile_standard == 'osm':
            img_std = '{z}/{x}/{y}'
        elif settings.tile_standard == 'bing':
            img_std = '?mapArea={maparea}&zoomlevel={zoomlevel}'

        try:
            start_index = settings.mpt_url[0].index(img_std)
            self.url_prefix = settings.mpt_url[0][:start_index]
            self.url_suffix = settings.mpt_url[0][start_index + len(img_std):]
            if 'png' in self.url_suffix:
                self.tile_format = '.png'
            else:
                self.tile_format = '.png'  # useless code at the moment. will matter with different formats
        except ValueError:
            # this just checks if the tile image standard for downloading is set to {z}/{x}/{y}
            print(f'Incorrect tile format in cfg file. Please make sure url contains {img_std}')
            print('Failed to load map tiles!!')
            self.enable_tiles = False
        except UnboundLocalError:
            # this just checks if google standard was set as the default standard. Later this will have to be edited
            print("Incorrect tile standard in cfg file. Tile standard must be 'osm' or 'bing'.")
            print('Failed to load map tiles!!')
            self.enable_tiles = False

        # Image operations
        self.LOAD_ALTERED = settings.LOAD_ALTERED
        self.ALTER_TILE = settings.ALTER_TILE
        self.INVERT = settings.INVERT
        self.CONTRAST = settings.CONTRAST
        self.con_factor = settings.con_factor

        # ------Convert to texture. delete once you figure out how to load png into gui------
        self.tex_filetype = '.dds'
        # -------------- delete this once you figure out png---

    # Drawing functions below
    def tile_load(self):

        # create a tile array
        self.create_tile_array()

        # process tiles, download tiles if necessary
        self.process_tiles()

        for image_path in self.local_paths:
            image = path.join(image_path)
            self.map_textures.append(self.bindTexture(image))

    def tile_reload(self, screen_zoom=0.4, zoom_array=np.array([1.2, 2.5, 6.0, 10.0, 15.0, 30.0]),
                    bbox=None):
        # set zoom level
        if bbox is None:
            bbox = [52.47, 4.69, 52.24, 5.0]

        # clear everything
        self.clear_tiles()

        self.lat1 = bbox[0]
        self.lon1 = bbox[1]
        self.lat2 = bbox[2]
        self.lon2 = bbox[3]

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
        else:
            self.zoom_level = 14
        # print(f'zoom level {self.zoom_level}')
        # create a tile array
        self.create_tile_array()

        # process tiles, download tiles if necessary
        self.process_tiles()

        for image_path in self.local_paths:
            image = path.join(image_path)
            self.map_textures.append(self.bindTexture(image))

    def tile_render(self):
        for corner in self.render_corners:
            texvertices = np.array(corner, dtype=np.float32)
            textexcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)
            self.tiles.append(RenderObject(gl.GL_TRIANGLE_FAN, vertex=texvertices, texcoords=textexcoords))

    def paint_map(self, main_widget):
        main_widget.texture_shader.use()

        for i in range(len(self.tile_array)):
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.map_textures[i])
            self.tiles[i].draw()

    def clear_tiles(self):
        self.local_paths = []
        self.tiles = []
        self.map_textures = []
        self.render_corners = []

    # Non-drawing functions below
    def process_tiles(self):

        # Download tiles in multiple threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.download_tile, self.tile_array)

        # Image operations
        for item in self.tile_array:
            x = item[0]  # tile x
            y = item[1]  # tile y
            img_path = path.join(str(self.zoom_level), str(x), str(y))
            alt_local_path = path.join(self.tile_dir, f'{img_path}a{self.tile_format}')
            raw_local_path = path.join(self.tile_dir, img_path + self.tile_format)
            local_path = raw_local_path

            # Skip image operations if downloading failed. Maybe should place this in a separate function
            # since this is checked at start of for loop. Only reason this is set is because there will be an
            # exception if downloading fails. Maybe using a try statement instead of..but idk yet
            if self.enable_tiles:
                # Check if Altered Images should be loaded.
                if self.LOAD_ALTERED:
                    # check if you want to make a new change or if path exists.
                    if not path.exists(alt_local_path) or self.ALTER_TILE:
                        local_path = self.alter_tile(alt_local_path, raw_local_path)
                    else:
                        local_path = alt_local_path

                # Convert to texture, remove once you figure out how to put .png files in gui
                if not path.exists(local_path[:-4] + self.tex_filetype):
                    local_path = self.convert_to_texture(local_path)
                else:
                    local_path = local_path[:-4] + self.tex_filetype

            # create arrays of local paths for later use
            self.local_paths.append(local_path)

            # create array of tile corners for later use
            if settings.tile_standard == 'osm':
                img_corner = self.tile_corners(x, y, self.zoom_level)
            elif settings.tile_standard == 'bing':
                # use image metadata to get corner info for rendering. perhaps save this data so it goes faster on
                # reruns
                img_corner = self.bing_bbox(x, y)

            self.render_corners.append(img_corner)

    def create_tile_array(self):
        # get tile number of corners of bounding box
        tile_nw, tile_ne, tile_sw, tile_se = self.bbox_corner_tiles(self.lat1, self.lon1,
                                                                    self.lat2, self.lon2, self.zoom_level)

        # Find size of 2D array for tiles
        n_columns = tile_ne[0] - tile_nw[0] + 1
        n_rows = tile_sw[1] - tile_nw[1] + 1
        self.tile_array = np.zeros((n_rows, n_columns), [('x_loc', int), ('y_loc', int)])

        # fill NW corner of tile array
        self.tile_array[0, 0] = tile_nw

        # loop through rows. x is constant in one row
        for row in range(n_rows):
            # for each row loop through a column. y is constant in the same column.
            for col in range(n_columns):
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
        # print(f'Requesting {len(self.tile_array)} tiles at zoom level {self.zoom_level}')

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
                if settings.tile_standard == 'osm':
                    # osm tile_format downloads have the same image path as local folder
                    url_img_path = img_path
                elif settings.tile_standard == 'bing':
                    # alter image path for bing maps url download
                    lat2, lon1, lat1, lon2 = self.tileEdges(x, y, self.zoom_level)
                    map_area = f'{lat2},{lon1},{lat1},{lon2}'
                    url_img_path = f'?mapArea={map_area}&zoomlevel={str(self.zoom_level)}'

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

    def bing_bbox(self, x, y):

        # get area of tile and create the bing url for a metadata request
        lat2, lon1, lat1, lon2 = self.tileEdges(x, y, self.zoom_level)
        map_area = f'{lat2},{lon1},{lat1},{lon2}'

        url_img_path = f'?mapArea={map_area}&zoomlevel={str(self.zoom_level)}'

        image_url = self.url_prefix + url_img_path + self.url_suffix + '&mmd=1'

        # create request
        bbox_image = requests.get(image_url).json()['resourceSets'][0]['resources'][0]['bbox']
        img_corner = ((bbox_image[0], bbox_image[3]), (bbox_image[0], bbox_image[1]),
                      (bbox_image[2], bbox_image[1]), (bbox_image[2], bbox_image[3]))
        return img_corner

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
