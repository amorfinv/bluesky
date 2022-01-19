# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:14:20 2021

@author: nipat
"""

import dill
import networkx as nx

from plugins.streets.flow_control import street_graph,bbox
from plugins.streets.agent_path_planning import *
from plugins.streets.open_airspace_grid import *

dilled_list=dill.load(open("path_plans/Path_Planning.dill", 'rb'),ignore=True)
paths_dict=dilled_list[0]
graph=dilled_list[1]