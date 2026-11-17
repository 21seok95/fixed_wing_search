"""Transverse Mercator on the WGS84 ellipsoid (the projection of UTM) for the real-region maps.

User 2026-11-17 (docs/scenario_redesign.md): the 120 km maps use this instead of the local equirectangular projection of the 60 km maps,
whose east-west lengths are off by about 0.3 % at the edges of a 60 km map and 0.65 % at the edges of a 120 km map; the same transform
turns GPS positions into map coordinates when the policy flies a real aircraft.

Kruger's series to third order in the third flattening n, as given for UTM (Karney 2011, "Transverse Mercator with an accuracy of a few
nanometers"): the terms left out are of order a n^4, below 0.1 mm. ``tm_forward`` / ``tm_inverse`` take the central meridian and the
scale factor; ``LocalTM`` is a map frame (origin at a region centre, central meridian through it, scale 1, x east and y north in metres);
``utm`` gives standard UTM zone coordinates (for exchange with other tools).
"""
from __future__ import annotations

import math

import numpy as np

A_WGS84 = 6378137.0
F_WGS84 = 1/298.257223563
_n = F_WGS84/(2 - F_WGS84)
_A = A_WGS84/(1 + _n)*(1 + _n**2/4 + _n**4/64)
_E = 2*math.sqrt(_n)/(1 + _n)                                   # first eccentricity
_ALPHA = (_n/2 - 2*_n**2/3 + 5*_n**3/16, 13*_n**2/48 - 3*_n**3/5, 61*_n**3/240)
_BETA = (_n/2 - 2*_n**2/3 + 37*_n**3/96, _n**2/48 + _n**3/15, 17*_n**3/480)
_DELTA = (2*_n - 2*_n**2/3 - 2*_n**3, 7*_n**2/3 - 8*_n**3/5, 56*_n**3/15)


def tm_forward(lat, lon, lon0: float, k0: float = 1.0):
    """Latitude, longitude (deg) -> (x east of the central meridian ``lon0``, y north of the equator) in metres."""
    phi = np.radians(np.asarray(lat, float))
    dl = np.radians(np.asarray(lon, float) - lon0)
    s = np.sin(phi)
    t = np.sinh(np.arctanh(s) - _E*np.arctanh(_E*s))           # tangent of the conformal latitude
    xi_p = np.arctan2(t, np.cos(dl))
    eta_p = np.arctanh(np.sin(dl)/np.sqrt(1 + t*t))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_ALPHA, start=1):
        xi += a*np.sin(2*j*xi_p)*np.cosh(2*j*eta_p)
        eta += a*np.cos(2*j*xi_p)*np.sinh(2*j*eta_p)
    return k0*_A*eta, k0*_A*xi


def tm_inverse(x, y, lon0: float, k0: float = 1.0):
    """(x east of the central meridian ``lon0``, y north of the equator) in metres -> latitude, longitude (deg)."""
    xi = np.asarray(y, float)/(k0*_A)
    eta = np.asarray(x, float)/(k0*_A)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, start=1):
        xi_p -= b*np.sin(2*j*xi)*np.cosh(2*j*eta)
        eta_p -= b*np.cos(2*j*xi)*np.sinh(2*j*eta)
    chi = np.arcsin(np.sin(xi_p)/np.cosh(eta_p))                 # conformal latitude
    phi = chi.copy()
    for j, d in enumerate(_DELTA, start=1):
        phi += d*np.sin(2*j*chi)
    return np.degrees(phi), lon0 + np.degrees(np.arctan2(np.sinh(eta_p), np.cos(xi_p)))


class LocalTM:
    """Map frame of a region: transverse Mercator with the central meridian through the centre, scale 1, origin at the centre
    (x east, y north, metres). Grid north is true north along the centre meridian; elsewhere they differ by ``convergence``."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.y0 = float(tm_forward(self.lat0, self.lon0, self.lon0)[1])

    def to_xy(self, lat, lon):
        x, y = tm_forward(lat, lon, self.lon0)
        return x, y - self.y0

    def to_latlon(self, x, y):
        return tm_inverse(x, np.asarray(y, float) + self.y0, self.lon0)

    def convergence_deg(self, lat, lon, d: float = 1e-4):
        """Angle (deg, counter-clockwise) from grid north to true north at the points: grid heading = true heading - convergence
        (headings clockwise from north). Zero on the centre meridian, positive east of it in the northern hemisphere."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        x0, y0 = self.to_xy(lat - d, lon)
        x1, y1 = self.to_xy(lat + d, lon)
        return np.degrees(np.arctan2(-(x1 - x0), y1 - y0))


def utm(lat, lon):
    """Standard UTM coordinates (zone, easting, northing) of points in one zone (the zone of the first point), for exchange with other
    tools; the maps themselves use ``LocalTM``."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    zone = int(math.floor((float(np.ravel(lon)[0]) + 180)/6) + 1)
    x, y = tm_forward(lat, lon, (zone - 1)*6 - 180 + 3, k0=0.9996)
    return zone, 500000.0 + x, np.where(lat < 0, y + 10_000_000.0, y)


# ---------------------------------------------------------------- for checks
def meridian_radii(lat):
    """Meridional (M) and prime-vertical (N) radii of curvature (m) at the latitudes (deg)."""
    s2 = np.sin(np.radians(np.asarray(lat, float)))**2
    e2 = F_WGS84*(2 - F_WGS84)
    w = np.sqrt(1 - e2*s2)
    return A_WGS84*(1 - e2)/w**3, A_WGS84/w
