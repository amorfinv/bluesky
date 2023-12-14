import numpy as np
from bluesky.tools import geo

def main():
    lats = [51.92538138, 51.92468805, 51.92301863, 51.92253655, 51.92259552,
        51.92050521, 51.91982034, 51.91951172, 51.91896227, 51.91698976,
        51.91648101, 51.91558946, 51.91540193, 51.91485868, 51.91426837,
        51.91380112, 51.91315579, 51.91282719, 51.91246114, 51.91215457,
        51.91169534, 51.90851617, 51.90767466, 51.90372489, 51.90297921,
        51.90140286, 51.90044015, 51.89947969, 51.89831673, 51.89535075,
        51.89751413, 51.8994996 , 51.90047455, 51.90151927, 51.9018552 ,
        51.90289151, 51.90304379, 51.90325892, 51.90357007, 51.90385237,
        51.90393509, 51.90389358, 51.90406931, 51.90347338, 51.90301345,
        51.90233364, 51.90194732, 51.9018468 , 51.90159256, 51.90051724,
        51.89898188, 51.89878228, 51.89784874, 51.89740124, 51.89737495,
        51.89718895, 51.89658587, 51.89604456, 51.89590416, 51.89554405,
        51.8952795 ]

    lons = [4.46200949, 4.46221277, 4.46271118, 4.46284714, 4.46385339,
        4.46458932, 4.46509094, 4.46335767, 4.46333887, 4.46290264,
        4.46285903, 4.46305818, 4.46313617, 4.46336886, 4.46362837,
        4.46372998, 4.46382986, 4.46377586, 4.46366943, 4.46360625,
        4.4638119 , 4.4658944 , 4.46632017, 4.46794897, 4.46810197,
        4.4681262 , 4.46813629, 4.46814633, 4.46815847, 4.47654645,
        4.48694547, 4.49610129, 4.49630676, 4.49653875, 4.49661245,
        4.49686061, 4.49752749, 4.49846937, 4.49983165, 4.50117286,
        4.50161721, 4.50230028, 4.50265808, 4.50410697, 4.50364388,
        4.50514071, 4.50569335, 4.50548709, 4.50493104, 4.50630885,
        4.50820675, 4.50834404, 4.50817994, 4.50828211, 4.50793929,
        4.50793654, 4.50685326, 4.50685223, 4.50704488, 4.5063573 ,
        4.50585223]

    turn_bool, turn_speed, turn_coords = get_turn_arrays(lats, lons)

    print(turn_bool)
    print(turn_speed)
    edges = ['819-826', '819-826', '826-917', '917-930', '930-950', '950-956', '956-1042', '956-1042', '1042-1057', '1057-1014', '1014-907', '1014-907', '907-779', '779-776']


def get_turn_arrays(lats, lons, cutoff_angle=25):
    """
    Get turn arrays from latitude and longitude arrays.
    The function returns three arrays with the turn boolean, turn speed and turn coordinates.
    Turn speed depends on the turn angle.
        - Speed set to 0 for no turns.
        - Speed is 10 knots for angles between 25 and 100 degrees.
        - Speed is 5 knots for angles between 100 and 150 degrees.
        - Speed is 2 knots for angles larger than 150 degrees.
    Parameters
    ----------
    lat : numpy.ndarray
        Array with latitudes of route
    lon : numpy.ndarray
        Array with longitudes of route
    cutoff_angle : int
        Cutoff angle for turning. Default is 25.
    Returns
    -------
    turn_bool : numpy.ndarray
        Array with boolean values for turns.
    turn_speed : numpy.ndarray
        Array with turn speed. If no turn, speed is 0.
    turn_coords : numpy.ndarray
        Array with turn coordinates. If no turn then it has (-9999.9, -9999.9)
    """

    # Define empty arrays that are same size as lat and lon
    turn_speed = np.zeros(len(lats))
    turn_bool = np.array([False] * len(lats), dtype=np.bool_)
    turn_coords = np.array([(-9999.9, -9999.9)] * len(lats), dtype="f,f")

    # Initialize variables for the loop
    lat_prev = lats[0]
    lon_prev = lons[0]

    # loop thru the points to calculate turn angles
    for i in range(1, len(lats) - 1):
        # reset some values for the loop
        lat_cur = lats[i]
        lon_cur = lons[i]
        lat_next = lats[i + 1]
        lon_next = lons[i + 1]

        # calculate angle between points
        a1, _ = geo.qdrdist(lat_prev, lon_prev, lat_cur, lon_cur)
        a2, _ = geo.qdrdist(lat_cur, lon_cur, lat_next, lon_next)

        # fix angles that are larger than 180 degrees
        angle = abs(a1 - a2)
        angle = 360 - angle if angle > 180 else angle

        # give the turn speeds based on the angle
        if angle > cutoff_angle and i != 0:

            # set turn bool to true and get the turn coordinates
            turn_bool[i] = True
            turn_coords[i] = (lat_cur, lon_cur)

            # calculate the turn speed based on the angle.
            turn_speed[i] = 5

        else:
            turn_coords[i] = (-9999.9, -9999.9)

        # update the previous values at the end of the loop
        lat_prev = lat_cur
        lon_prev = lon_cur

    return turn_bool, turn_speed, turn_coords


if __name__ == '__main__':
    main()