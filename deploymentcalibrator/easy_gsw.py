import gsw
from . import polygon
from numpy import mean as npmean

BALTIC = ((9.813,52.881),
          (9.022, 56.486),
          (20.624, 67.424),
          (27.500, 67.250),
          (48.480, 53.700))


baltic = polygon.Polygon(*[polygon.Point(x,y) for x,y in BALTIC])


def density_from_C_t(C, t, p, lon, lat):
    SP = gsw.SP_from_C(C, t, p)

    lon_mean = npmean(lon)
    lat_mean = npmean(lat)

    if baltic.contains(polygon.Point(lon_mean, lat_mean)):
        SA = gsw.SA_from_SP_Baltic(SP, lon_mean, lat_mean)
    else:
        SA = gsw.SA_from_SP(SP, p, lon_mean, lat_mean)
    density = gsw.rho_t_exact(SA, t, p)
    pDensity = gsw.pot_rho_t_exact(SA, t, p, 0)
    return density, pDensity

def SA_from_C_t(C, t, p, lon, lat):
    SP = gsw.SP_from_C(C, t, p)

    lon_mean = npmean(lon)
    lat_mean = npmean(lat)

    if baltic.contains(polygon.Point(lon_mean, lat_mean)):
        SA = gsw.SA_from_SP_Baltic(SP, lon_mean, lat_mean)
    else:
        SA = gsw.SA_from_SP(SP, p, lon_mean, lat_mean)
    return SA

def CT_from_C_t(C, t, p, lon, lat):
    SP = gsw.SP_from_C(C, t, p)
    lon_mean = npmean(lon)
    lat_mean = npmean(lat)

    if baltic.contains(polygon.Point(lon_mean, lat_mean)):
        SA = gsw.SA_from_SP_Baltic(SP, lon_mean, lat_mean)
    else:
        SA = gsw.SA_from_SP(SP, p, lon_mean, lat_mean)
    CT = gsw.CT_from_t(SA, t, p)
    return CT
