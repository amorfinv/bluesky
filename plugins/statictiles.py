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

from PyQt5.QtGui import (QOpenGLTexture, QImage)


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
        'plugin_name':     'STATICTILE',

        # The type of this plugin.
        'plugin_type':     'gui',
        }

    # init_plugin() should always return a configuration dict.
    return config

class MyMap(ui.RenderObject, layer=100):
    def __init__(self, parent):
        super().__init__(parent=parent)
        
        self.map_tiles = ui.VertexArrayObject(ui.gl.GL_TRIANGLE_FAN)

    def create(self):
        bbox_corners = [(51.6180165487737, 5.625), (51.6180165487737, 4.21875), (52.48278022207821, 4.21875), (52.48278022207821, 5.625)]
        texvertices = np.array(bbox_corners, dtype=np.float32)
        textexcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)
        
        fname = 'data/graphics/tiles/opentopomap/8/131/84.png'
        self.map_tiles.create(vertex=texvertices, texcoords=textexcoords, texture=fname)

    def draw(self):
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)
        self.map_tiles.draw()

class MyTiles(ui.RenderObject, layer=100):
    def __init__(self, parent):
        super().__init__(parent=parent)
        
        self.map = ui.VertexArrayObject(ui.gl.GL_TRIANGLE_FAN, shader_type='tiled')
        self.texture = ui.Texture(target=QOpenGLTexture.Target2DArray)

    def create(self):
        bbox_corners = [(51.6180165487737, 5.625), (51.6180165487737, 4.21875), (52.48278022207821, 4.21875), (52.48278022207821, 5.625)]
        texvertices = np.array(bbox_corners, dtype=np.float32)
        textexcoords = np.array([(1, 1), (1, 0), (0, 0), (0, 1)], dtype=np.float32)
        
        fname = 'data/graphics/tiles/opentopomap/8/131/84.png'
        self.map_tiles.create(vertex=texvertices, texcoords=textexcoords, texture=fname)

        # mapvertices = np.array(
        #     [-90.0, 540.0, -90.0, -540.0, 90.0, -540.0, 90.0, 540.0], dtype=np.float32)
        
        # self.texture.create()
        # self.texture.add_bounding_box(-90, -180, 90, 180)
        # self.map.create(vertex=mapvertices, texture=self.texture)
        # self.offsetzoom_loc = glh.ShaderSet.get_shader(
        #     'tiled').uniformLocation('offset_scale')
        # # Make sure that we have textures on first draw
        # self.texture.on_panzoom_changed(True)

    def draw(self):
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)
        self.map_tiles.draw()

        # self.shaderset.enable_wrap(False)
        # shader = glh.ShaderSet.get_shader('tiled')
        # shader.bind()
        # shader.setUniformValue(self.offsetzoom_loc, *self.texture.offsetscale)
        # self.map.draw()



import math
from os import makedirs, path
import weakref
from collections import OrderedDict
from urllib.request import urlopen
from urllib.error import URLError
import numpy as np

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtGui import QImage

import bluesky as bs
from bluesky.core import Signal
import bluesky.ui.qtgl.glhelpers as glh
class TiledTexture(ui.Texture):
    ''' Tiled texture implementation for the BlueSky GL gui. '''
    def __init__(self, glsurface, tilesource='opentopomap'):
        super().__init__(target=ui.Texture.Target2DArray)
        self.threadpool = QThreadPool()
        tileinfo = bs.settings.tile_sources.get(tilesource)
        if not tileinfo:
            raise KeyError(f'Tile source {tilesource} not found!')
        max_dl = tileinfo.get('max_download_workers', bs.settings.max_download_workers)
        self.maxzoom = tileinfo.get('max_tile_zoom', bs.settings.max_tile_zoom)
        self.threadpool.setMaxThreadCount(min(bs.settings.max_download_workers, max_dl))
        self.tilesource = tilesource
        self.tilesize = (256,256)
        self.curtiles = OrderedDict()
        self.fullscreen = False
        self.offsetscale = np.array([0, 0, 1], dtype=np.float32)
        self.bbox = list()
        self.glsurface = glsurface
        self.indextexture = glh.Texture(target=glh.Texture.Target2D)
        self.indexsampler_loc = 0
        self.arraysampler_loc = 0
        # bs.net.actnodedata_changed.connect(self.actdata_changed)
        # Signal('panzoom').connect(self.on_panzoom_changed)


    # def create(self):
    #     ''' Create this texture in GPU memory. '''
    #     if self.isCreated():
    #         return
    #     super().create()
    #     # Fetch a temporary tile image to get dimensions
    #     tmptile = Tile(self.tilesource, 1, 1, 1, 0, 0)
    #     img = tmptile.image
    #     self.setFormat(glh.Texture.RGBA8_UNorm)
    #     self.tilesize = (img.width(), img.height())
    #     self.setSize(img.width(), img.height())
    #     self.setLayers(bs.settings.tile_array_size)
    #     super().bind()
    #     self.allocateStorage()
    #     self.setWrapMode(glh.Texture.DirectionS,
    #                      glh.Texture.ClampToBorder)
    #     self.setWrapMode(glh.Texture.DirectionT,
    #                      glh.Texture.ClampToBorder)
    #     self.setMinMagFilters(glh.Texture.Linear, glh.Texture.Linear)

    #     # Initialize index texture
    #     # RG = texcoord offset, B = zoom factor, A = array index
    #     itexw = int(np.sqrt(bs.settings.tile_array_size) * 4 / 3 + 10)
    #     itexh = int(np.sqrt(bs.settings.tile_array_size) * 3 / 4 + 10)
    #     self.indextexture.create()
    #     self.indextexture.setFormat(glh.Texture.RGBA32I)
    #     self.indextexture.setSize(itexw, itexh)
    #     self.indextexture.bind(1)
    #     # self.indextexture.allocateStorage(glh.Texture.RGBA_Integer, glh.Texture.Int32)

    #     idxdata = np.array(itexw * itexh *
    #                        [(0, 0, 0, -1)], dtype=np.int32)
    #     glh.gl.glTexImage2D_alt(glh.Texture.Target2D, 0, glh.Texture.RGBA32I,
    #                             itexw, itexh, 0, glh.Texture.RGBA_Integer,
    #                             glh.Texture.Int32, idxdata.tobytes())

    #     self.indextexture.setWrapMode(glh.Texture.DirectionS,
    #                                   glh.Texture.ClampToBorder)
    #     self.indextexture.setWrapMode(glh.Texture.DirectionT,
    #                                   glh.Texture.ClampToBorder)
    #     self.indextexture.setMinMagFilters(glh.Texture.Nearest, glh.Texture.Nearest)

    #     shader = glh.ShaderSet.get_shader('tiled')
    #     self.indexsampler_loc = shader.uniformLocation('tile_index')
    #     self.arraysampler_loc = shader.uniformLocation('tile_texture')

    # def bind(self, unit=0):
    #     ''' Bind this texture for drawing. '''
    #     # Set sampler locations
    #     glh.ShaderProgram.bound_shader.setUniformValue(self.indexsampler_loc, 2)
    #     glh.ShaderProgram.bound_shader.setUniformValue(self.arraysampler_loc, 4)
    #     # Bind index texture to texture unit 0
    #     self.indextexture.bind(2)
    #     # Bind tile array texture to texture unit 1
    #     super().bind(4)

    # def on_panzoom_changed(self, finished=False):
    #     ''' Update textures whenever pan/zoom changes. '''
    #     # Check if textures need to be updated
    #     viewport = self.glsurface.viewportlatlon()
    #     surfwidth_px = self.glsurface.width()
    #     # First determine floating-point, hypothetical values
    #     # to calculate the required tile zoom level
    #     # floating-point number of tiles that fit in the width of the view
    #     ntiles_hor = surfwidth_px / self.tilesize[0]
    #     # Estimated width in longitude of one tile
    #     est_tilewidth = abs(viewport[3] - viewport[1]) / ntiles_hor

    #     zoom = tilezoom(est_tilewidth, self.maxzoom)
    #     # With the tile zoom level get the required number of tiles
    #     # Top-left and bottom-right tiles:
    #     x0, y0 = latlon2tilenum(*viewport[:2], zoom)
    #     x1, y1 = latlon2tilenum(*viewport[2:], zoom)
    #     nx = abs(x1 - x0) + 1
    #     ny = abs(y1 - y0) + 1

    #     # Calculate the offset of the top-left tile w.r.t. the screen top-left corner
    #     tile0_topleft = np.array(tilenum2latlon(x0, y0, zoom))
    #     tile0_bottomright = np.array(tilenum2latlon(x0 + 1, y0 + 1, zoom))
    #     tilesize_latlon0 = np.abs(tile0_bottomright - tile0_topleft)
    #     offset_latlon0 = viewport[:2] - tile0_topleft
    #     tex_y0, tex_x0 = np.abs(offset_latlon0 / tilesize_latlon0)

    #     # Calculate the offset of the bottom-right tile w.r.t. the screen bottom right corner
    #     tile1_topleft = np.array(tilenum2latlon(x1, y1, zoom))
    #     tile1_bottomright = np.array(tilenum2latlon(x1 + 1, y1 + 1, zoom))
    #     tilesize_latlon1 = np.abs(tile1_bottomright - tile1_topleft)
    #     offset_latlon1 = viewport[2:] - tile1_topleft
    #     tex_y1, tex_x1 = np.abs(offset_latlon1 / tilesize_latlon1) + [ny - 1, nx - 1]
    #     # Store global offset and scale for shader uniform
    #     self.offsetscale = np.array(
    #         [tex_x0, tex_y0, tex_x1 - tex_x0, tex_y1 - tex_y0], dtype=np.float32)
    #     # Determine required tiles
    #     index_tex = []
    #     curtiles = OrderedDict()
    #     curtiles_difzoom = OrderedDict()
    #     for j, y in enumerate(range(y0, y1 + 1)):
    #         for i, x in enumerate(range(x0, x1 + 1)):
    #             # Find tile index if already loaded
    #             idx = self.curtiles.pop((x, y, zoom), None)
    #             if idx is not None:
    #                 # correct zoom, so dx,dy=0, zoomfac=1
    #                 index_tex.extend((0, 0, 1, idx))
    #                 curtiles[(x, y, zoom)] = idx
    #             else:
    #                 if finished:
    #                     # Tile not loaded yet, fetch in the background
    #                     task = TileLoader(self.tilesource, zoom, x, y, i, j)
    #                     task.signals.finished.connect(self.load_tile)
    #                     self.threadpool.start(task)

    #                 # In the mean time, check if more zoomed-out tiles are loaded that can be used
    #                 for z in range(zoom - 1, max(2, zoom - 5), -1):
    #                     zx, zy, dx, dy = zoomout_tilenum(x, y, z - zoom)
    #                     idx = self.curtiles.pop((zx, zy, z), None)
    #                     if idx is not None:
    #                         curtiles_difzoom[(zx, zy, z)] = idx
    #                         zoomfac = int(2 ** (zoom - z))
    #                         dxi = int(round(dx * zoomfac))
    #                         dyi = int(round(dy * zoomfac))
    #                         # offset zoom, so possible offset dxi, dyi
    #                         index_tex.extend((dxi, dyi, zoomfac, idx))
    #                         break
    #                 else:
    #                     # No useful tile found
    #                     index_tex.extend((0, 0, 0, -1))
    #     # Update curtiles ordered dict
    #     curtiles.update(curtiles_difzoom)
    #     curtiles.update(self.curtiles)
    #     self.curtiles = curtiles
    #     data = np.array(index_tex, dtype=np.int32)
    #     self.glsurface.makeCurrent()
    #     self.indextexture.bind(2)
    #     glh.gl.glTexSubImage2D_alt(glh.Texture.Target2D, 0, 0, 0, nx, ny,
    #                                glh.Texture.RGBA_Integer,
    #                                glh.Texture.Int32, data.tobytes())

    # def load_tile(self, tile):
    #     ''' Send loaded image data to GPU texture array.

    #         This function is called on callback from the
    #         asynchronous image load function.
    #     '''
    #     layer = len(self.curtiles)
    #     if layer >= bs.settings.tile_array_size:
    #         # we're exceeding the size of the GL texture array. Replace the least-recent tile
    #         _, layer = self.curtiles.popitem()

    #     self.curtiles[(tile.tilex, tile.tiley, tile.zoom)] = layer
    #     # Update the ordering of the tile dict: the new tile should be on top
    #     self.curtiles.move_to_end((tile.tilex, tile.tiley, tile.zoom), last=False)
    #     idxdata = np.array([0, 0, 1, layer], dtype=np.int32)
    #     self.glsurface.makeCurrent()
    #     self.indextexture.bind(2)
    #     glh.gl.glTexSubImage2D_alt(glh.Texture.Target2D, 0, tile.idxx, tile.idxy,
    #                                1, 1, glh.Texture.RGBA_Integer,
    #                                glh.Texture.Int32, idxdata.tobytes())
    #     super().bind(4)
    #     self.setLayerData(layer, tile.image)
    #     self.indextexture.release()
    #     self.release()

bs.settings.set_variable_defaults(
    tile_array_size=100,
    max_download_workers=2,
    max_tile_zoom=18,
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


class Tile:
    ''' Wrapper object for tile data and properties. '''

    def __init__(self, source, zoom, tilex, tiley, idxx, idxy):
        super().__init__()
        self.source = source
        self.zoom = zoom
        self.tilex = tilex
        self.tiley = tiley
        self.idxx = idxx
        self.idxy = idxy
        self.ext = source[source.rfind('.'):]

        self.image = None
        # For the image data, check cache path first
        fpath = path.join(bs.settings.cache_path, source, str(zoom), str(tilex))
        fname = path.join(fpath, f'{tiley}{self.ext}')
        if path.exists(fname):
            self.image = QImage(fname).convertToFormat(QImage.Format_ARGB32)
        else:
            # Make sure cache directory exists
            makedirs(fpath, exist_ok=True)
            for url in bs.settings.tile_sources[source]['source']:
                try:
                    url_request = urlopen(url.format(
                        zoom=zoom, x=tilex, y=tiley))
                    data = url_request.read()
                    self.image = QImage.fromData(
                        data).convertToFormat(QImage.Format_ARGB32)
                    with open(fname, 'wb') as fout:
                        fout.write(data)
                    break
                except URLError as e:
                    print(e, f'({url.format(zoom=zoom, x=tilex, y=tiley)})')


class TileLoader(QRunnable):
    ''' Thread worker to load tiles in the background. '''
    class Signals(QObject):
        ''' Container class for worker signals. '''
        finished = pyqtSignal(Tile)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.signals = TileLoader.Signals()

    def run(self):
        ''' Function to execute in the worker thread. '''
        tile = Tile(*self.args, **self.kwargs)
        if tile.image is not None:
            self.signals.finished.emit(tile)
