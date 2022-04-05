# %%
import dill
import plugins.streets.agent_path_planning
import plugins.streets.flow_control
import os
import geopandas as gpd
import numpy as np
import json

def kwikdist(origin, destination):
    """
    Quick and dirty dist [nm]
    In:
        lat/lon, lat/lon [deg]
    Out:
        dist [nm]
    """
    # We're getting these guys as strings
    lona = float(origin[0])
    lata = float(origin[1])

    lonb = float(destination[0])
    latb = float(destination[1])

    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata)
    dlon    = np.radians(((lonb - lona)+180)%360-180)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)

    dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist    = re * dangle
    return dist

# get the pairs list
origins = gpd.read_file('/home/niki/Desktop/M2Decentralised/M2_test_decentralised/current_code/whole_vienna/gis/Sending_nodes.gpkg').to_numpy()[:,0:2]
destinations = gpd.read_file('/home/niki/Desktop/M2Decentralised/M2_test_decentralised/current_code/whole_vienna/gis/Recieving_nodes.gpkg').to_numpy()[:,0:2]

pairs = []
round_int = 10
for origin in origins:
    for destination in destinations:
        if kwikdist(origin, destination) >=800:
            lon1 = origin[0]
            lat1 = origin[1]
            lon2 = destination[0]
            lat2 = destination[1]
            pairs.append((round(lon1,round_int),round(lat1,round_int),round(lon2,round_int),round(lat2,round_int)))



# list files in directory
list_dills = os.listdir('path_plan_dills/')

# failed_dills = {}
failed_dills = []
for file in list_dills:

    try: 
        dill.load(open(f'path_plan_dills/{file}', 'rb'),ignore=True)
    except:
        # print('Error loading file: ' + file)
        # split file at '_'
        # failed_dill = file.split('_')[0]

        # get the origin and destination from pairs
        # failed_dills[failed_dill] = pairs[int(failed_dill)]
        failed_dills.append(f'path_plan_dills/{file}')
        continue

print(len(failed_dills))
# # write dictionary to json
# with open('failed_dills.json', 'w') as outfile:
#     json.dump(failed_dills, outfile)

# write failed dill to text file
with open('failed_dills.txt', 'w') as outfile:
    for failed_dill in failed_dills:
        outfile.write(failed_dill + '\n')
# %%
