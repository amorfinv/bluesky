#include <cmath>
#include <pyhelpers.hpp>
#include <geo.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

inline bool dist2_ac(const double &tlookahead,
                       const double &lat1, const double &lon1, const double &gs1,
                       const double &lat2, const double &lon2)
{
    // The groundspeed of ownship and intruder as vectors
    double u1        = gs1 * sin(trk1),
           v1        = gs1 * cos(trk1),
           u2        = gs2 * sin(trk2),
           v2        = gs2 * cos(trk2);

    double detdist   = tlookahead * gs1
    // The relative velocity vector
    double du        = u1 - u2,
           dv        = v1 - v2;
    kwik_in kin = kwik_in(lat1, lon1, lat2, lon2);
    double dist      = kwikdist(kin);

    return (dist <= detdist, dist);
}

