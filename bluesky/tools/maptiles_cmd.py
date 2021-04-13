import bluesky as bs

# Other settings of maptiles can be set from seetings.cfg file. 
# Refer to bluesky/ui/qtgl/maptiles.py for more information on the different settings.

def maptiles(*args):
	
	if not args == ():

		# arguments passed from stack are bounding box and zoom level

		# Eensure that bounding box is in correct order for map tile processing.
		# lat1 (north), lon1 (west), lat2 (south), lon2 (east)

		if args[0] > args[2]:
			lat1 = args[0]
			lat2 = args[2]
		else:
			lat1 = args[2]
			lat2 = args[0]

		if args[1] < args[2]:
			lon1 = args[1]
			lon2 = args[3]
		else:
			lon1 = args[3]
			lon2 = args[1]
		
		# Get zoom level
		zoom = args[4]

		bs.scr.maptile_cmd(lat1, lon1, lat2, lon2, zoom)
	else:
		# if nothing is passed then the function turns maptiles on and off. 
		# If you turn it off and then turn on again it turns on dynamically
		bs.scr.maptile_cmd()