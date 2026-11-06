"""Real-region terrain from open data, in the same format as the synthetic terrain.

Sources (all open, read over HTTP; raw downloads are cached under ``rl_search/data/real/<key>/raw``):
- Elevation: Copernicus DEM GLO-30 (a surface model: tree and roof heights are included).
- Land cover: ESA WorldCover 2021 v200 (10 m, read at its 20 m overview).
- Roads: OpenStreetMap ``highway`` ways through the Overpass API (ODbL, attribution required).

Grid: the 60 km square around the region centre on the synthetic axis (-30 km .. 30 km, 50 m),
x east and y north from the centre (local equirectangular projection, error < 0.1 % here).

Layers (agreed with the user 2026-11-06):
- ``water``: WorldCover water plus a two-cell band along OSM river centre lines wherever the river is
  mapped, with gaps of up to ``RIVER_GAP_M`` filled (the 20 m land cover misses parts of 50-150 m
  wide rivers and leaves one-cell chains that targets slip through diagonally), minus bridge cells. ``fords``: bridge cells over water (crossings, the dam crest is land).
- ``stream``: OSM rivers and streams that are not water cells (drawn only, passable).
- ``mountain``: empty. Steep ground (>= ``max_slope_deg``) is the only terrain no-go.
- ``forest``: WorldCover tree cover. ``built``: built-up land outside the city zones (counts as cover).
- ``city``: large towns, hand-off zones for ground units: built-up share >= 40 % in a 500 m window,
  holes filled, connected areas >= 3 km^2.
- ``surface``: road tiers outside the city zones. 2 = fast (motorway, trunk, primary, secondary,
  tertiary and their links), 1 = slow (unclassified, track). Village streets, service roads,
  footways and paths are left out.
- A road graph of the two tiers (nodes split at shared OSM nodes) for a later vehicle model.
"""
from __future__ import annotations

import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, label, map_coordinates, uniform_filter

from rl_search.terrain import Terrain, TerrainConfig, build_terrain

DATA = Path("rl_search/data/real")


@dataclass(frozen=True)
class Region:
    key: str
    name_ko: str
    lat: float
    lon: float


REGIONS = {r.key: r for r in (
    Region("chungju", "충주·제천", 37.05, 128.05),
    Region("chuncheon", "춘천·홍천", 37.82, 127.82),
    Region("andong", "안동", 36.60, 128.70),
    Region("nonsan", "논산·부여", 36.25, 127.00),
)}

FAST = ("motorway", "trunk", "primary", "secondary", "tertiary")
SLOW = ("unclassified", "track")
HIGHWAY_CODES = FAST + SLOW
WC_TREE, WC_BUILT, WC_WATER = 10, 50, 80
CITY_WINDOW_M, CITY_DENSITY, CITY_MIN_KM2 = 500.0, 0.4, 3.0
RIVER_GAP_M = 300.0       # longest stretch of an OSM river line without land-cover water that is filled
HTTP_HEADERS = {"User-Agent": "fixed-wing-search-research/0.1"}
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter")


def road_tier(highway: str) -> int:
    h = highway.removesuffix("_link")
    return 2 if h in FAST else 1 if h in SLOW else 0


# ------------------------------------------------------------------ projection
def scale(lat0: float) -> tuple[float, float]:
    """Metres per degree of longitude and latitude at ``lat0``."""
    return 111320.0*math.cos(math.radians(lat0)), 111320.0


def bbox(region: Region, half_m: float) -> tuple[float, float, float, float]:
    kx, ky = scale(region.lat)
    pad = 1.02*half_m
    return region.lat - pad/ky, region.lat + pad/ky, region.lon - pad/kx, region.lon + pad/kx


def to_xy(region: Region, lat, lon):
    kx, ky = scale(region.lat)
    return (np.asarray(lon) - region.lon)*kx, (np.asarray(lat) - region.lat)*ky


# ------------------------------------------------------------------ remote Cloud-Optimized GeoTIFF
class _HTTPFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (small reads cached in 64 KiB blocks)."""
    BLOCK = 1 << 16

    def __init__(self, url: str):
        import requests
        self._s = requests.Session()
        self.url, self.pos, self.cache = url, 0, {}
        r = self._s.head(url, allow_redirects=True, headers=HTTP_HEADERS, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, a, b):
        import requests
        for k in range(6):
            try:
                r = self._s.get(self.url, headers={**HTTP_HEADERS, "Range": f"bytes={a}-{b - 1}"}, timeout=120)
                r.raise_for_status()
                return r.content
            except requests.RequestException:
                if k == 5:
                    raise
                time.sleep(2*(k + 1))

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        a, b = self.pos, min(self.pos + n, self.size)
        if b <= a:
            return b""
        if b - a > 4*self.BLOCK:
            data = self._get(a, b)
        else:
            blocks = []
            for blk in range(a//self.BLOCK, (b - 1)//self.BLOCK + 1):
                if blk not in self.cache:
                    s = blk*self.BLOCK
                    self.cache[blk] = self._get(s, min(s + self.BLOCK, self.size))
                blocks.append(self.cache[blk])
            o = a - (a//self.BLOCK)*self.BLOCK
            data = b"".join(blocks)[o:o + b - a]
        self.pos = b
        return data

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def read_cog_window(url, lat0, lat1, lon0, lon1, level=0):
    """Pixels of a north-up EPSG:4326 COG inside the window: (array, row latitudes, column longitudes)."""
    import tifffile
    with tifffile.TiffFile(_HTTPFile(url)) as tif:
        base = tif.pages[0]
        sx, sy, _ = base.tags["ModelPixelScaleTag"].value
        tp = base.tags["ModelTiepointTag"].value
        lon_o, lat_o = tp[3], tp[4]
        page = base if level == 0 else tif.series[0].levels[level].pages[0]
        psx, psy = sx*base.shape[1]/page.shape[1], sy*base.shape[0]/page.shape[0]
        H, W = page.shape[:2]
        r0, r1 = max(0, int((lat_o - lat1)/psy)), min(H, int(np.ceil((lat_o - lat0)/psy)))
        c0, c1 = max(0, int((lon0 - lon_o)/psx)), min(W, int(np.ceil((lon1 - lon_o)/psx)))
        th, tw = page.tilelength, page.tilewidth
        ntx = (W + tw - 1)//tw
        out = np.zeros((r1 - r0, c1 - c0), page.dtype)
        fh = tif.filehandle
        for ty in range(r0//th, (r1 - 1)//th + 1):
            for tx in range(c0//tw, (c1 - 1)//tw + 1):
                i = ty*ntx + tx
                if page.databytecounts[i] == 0:
                    continue
                fh.seek(page.dataoffsets[i])
                tile = np.asarray(page.decode(fh.read(page.databytecounts[i]), i)[0]).reshape(th, tw, -1)[..., 0]
                ys, xs = ty*th, tx*tw
                a0, a1, b0, b1 = max(r0, ys), min(r1, ys + th), max(c0, xs), min(c1, xs + tw)
                out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = tile[a0 - ys:a1 - ys, b0 - xs:b1 - xs]
    return out, lat_o - (np.arange(r0, r1) + .5)*psy, lon_o + (np.arange(c0, c1) + .5)*psx


def _dem_tiles(lat0, lat1, lon0, lon1):
    tiles = []
    for la in range(math.floor(lat0), math.floor(lat1) + 1):
        for lo in range(math.floor(lon0), math.floor(lon1) + 1):
            name = f"Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM"
            tiles.append(read_cog_window(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif",
                                         max(lat0, la), min(lat1, la + 1), max(lon0, lo), min(lon1, lo + 1)))
    return tiles


def _worldcover_tiles(lat0, lat1, lon0, lon1, level=1):
    tiles = []
    for la in range(3*math.floor(lat0/3), 3*math.floor(lat1/3) + 1, 3):
        for lo in range(3*math.floor(lon0/3), 3*math.floor(lon1/3) + 1, 3):
            url = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                   f"ESA_WorldCover_10m_2021_v200_N{la:02d}E{lo:03d}_Map.tif")
            tiles.append(read_cog_window(url, max(lat0, la), min(lat1, la + 3), max(lon0, lo), min(lon1, lo + 3), level))
    return tiles


def _mosaic(tiles, fill):
    """Merge window tiles that share one pixel size into one array with row latitudes / column longitudes."""
    dlat = abs(tiles[0][1][0] - tiles[0][1][1]) if len(tiles[0][1]) > 1 else 1
    dlon = abs(tiles[0][2][1] - tiles[0][2][0]) if len(tiles[0][2]) > 1 else 1
    lat_hi = max(t[1][0] for t in tiles); lat_lo = min(t[1][-1] for t in tiles)
    lon_lo = min(t[2][0] for t in tiles); lon_hi = max(t[2][-1] for t in tiles)
    H = int(round((lat_hi - lat_lo)/dlat)) + 1; W = int(round((lon_hi - lon_lo)/dlon)) + 1
    out = np.full((H, W), fill, dtype=tiles[0][0].dtype)
    for arr, la, lo in tiles:
        r0 = int(round((lat_hi - la[0])/dlat)); c0 = int(round((lo[0] - lon_lo)/dlon))
        out[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]] = arr
    return out, lat_hi - np.arange(H)*dlat, lon_lo + np.arange(W)*dlon


def _overpass_ways(lat0, lat1, lon0, lon1, selector='["highway"]', splits=2):
    """OSM ways matching ``selector`` with tags, node ids and geometry, fetched in ``splits`` x ``splits`` boxes."""
    import requests
    ways = {}
    las, los = np.linspace(lat0, lat1, splits + 1), np.linspace(lon0, lon1, splits + 1)
    for i in range(splits):
        for j in range(splits):
            q = (f'[out:json][timeout:240];way{selector}({las[i]},{los[j]},{las[i + 1]},{los[j + 1]});'
                 'out body geom;')
            for k in range(9):
                url = OVERPASS[k % len(OVERPASS)]
                try:
                    r = requests.post(url, data={"data": q}, headers=HTTP_HEADERS, timeout=260)
                    r.raise_for_status()
                    els = r.json()["elements"]
                    break
                except Exception as e:  # server busy: try the next mirror
                    print(f"  Overpass {url.split('/')[2]}: {type(e).__name__}, retrying", flush=True)
                    time.sleep(5)
            else:
                raise RuntimeError("Overpass API unavailable")
            for el in els:
                ways[el["id"]] = {"id": el["id"], "tags": el["tags"], "nodes": el["nodes"],
                                  "geom": [(p["lat"], p["lon"]) for p in el["geometry"]]}
            print(f"  OSM box {i*splits + j + 1}/{splits*splits}: {len(els)} ways", flush=True)
    return list(ways.values())


def fetch_raw(region: Region, half_m: float = 30000.0, raw_dir: Path | None = None, refresh=False) -> Path:
    """Download (or reuse) the DEM, land cover and OSM highways of the region's box."""
    raw_dir = raw_dir or DATA/region.key/"raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    la0, la1, lo0, lo1 = bbox(region, half_m)
    meta = {"region": region.key, "lat": region.lat, "lon": region.lon, "bbox": [la0, la1, lo0, lo1],
            "sources": {"dem": "Copernicus DEM GLO-30 (DSM)", "landcover": "ESA WorldCover 2021 v200, 20 m overview",
                        "roads": "OpenStreetMap via Overpass API (ODbL)"}}
    if refresh or not (raw_dir/"dem.npz").exists():
        print("Copernicus DEM ...", flush=True)
        z, la, lo = _mosaic(_dem_tiles(la0, la1, lo0, lo1), np.float32(np.nan))
        np.savez_compressed(raw_dir/"dem.npz", z=z, lat=la, lon=lo)
    if refresh or not (raw_dir/"worldcover.npz").exists():
        print("ESA WorldCover ...", flush=True)
        lc, la, lo = _mosaic(_worldcover_tiles(la0, la1, lo0, lo1), np.uint8(0))
        np.savez_compressed(raw_dir/"worldcover.npz", lc=lc, lat=la, lon=lo)
    if refresh or not (raw_dir/"osm_highways.json").exists():
        print("OSM highways ...", flush=True)
        meta["osm_fetched"] = time.strftime("%Y-%m-%d %H:%M")
        (raw_dir/"osm_highways.json").write_text(json.dumps({**meta, "ways": _overpass_ways(la0, la1, lo0, lo1)},
                                                            ensure_ascii=False))
    if refresh or not (raw_dir/"osm_waterways.json").exists():
        print("OSM waterways ...", flush=True)
        ways = _overpass_ways(la0, la1, lo0, lo1, '["waterway"~"^(river|stream|canal)$"]', splits=1)
        (raw_dir/"osm_waterways.json").write_text(json.dumps({**meta, "osm_fetched": time.strftime("%Y-%m-%d %H:%M"),
                                                              "ways": ways}, ensure_ascii=False))
    return raw_dir


def _fill_river_gaps(water, river_lines, axis):
    """Water cells added along OSM river lines where the land cover misses at most ``RIVER_GAP_M``."""
    cell = axis[1] - axis[0]
    n = len(axis)
    add = np.zeros_like(water)
    gap_max = int(RIVER_GAP_M/cell)
    for xy in river_lines:
        r, c = _raster_lines([xy], axis, step_m=cell/3)
        if len(r) < 2:
            continue
        keep = np.r_[True, (np.diff(r) != 0) | (np.diff(c) != 0)]
        r, c = r[keep], c[keep]
        wet = np.zeros(len(r), bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = np.clip(r + dr, 0, n - 1), np.clip(c + dc, 0, n - 1)
                wet |= water[rr, cc]
        idx = np.flatnonzero(wet)
        add[r[idx], c[idx]] = True                 # the centre line itself where the river is mapped
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap_max + 1:
                add[r[a:b], c[a:b]] = True
    # two cells wide, so no diagonal step slips through
    grow = add.copy()
    grow[1:, :] |= add[:-1, :]
    grow[:, 1:] |= add[:, :-1]
    return grow & ~water


# ------------------------------------------------------------------ build
def _sample_grid(arr, lats, lons, glat, glon, order):
    """Sample a north-up lat/lon raster at the grid points (rows = glat ascending, cols = glon)."""
    rr = (lats[0] - glat)/(lats[0] - lats[1])
    cc = (glon - lons[0])/(lons[1] - lons[0])
    R, C = np.meshgrid(rr, cc, indexing="ij")
    src = arr.astype(float) if order else arr
    out = map_coordinates(src, [R.ravel(), C.ravel()], order=order, mode="nearest")
    return out.reshape(len(glat), len(glon))


def _raster_lines(lines, axis, step_m=10.0):
    """Rows and columns of the grid cells touched by polylines (points every ``step_m`` metres)."""
    cell, lo, n = axis[1] - axis[0], axis[0], len(axis)
    rows, cols = [], []
    for xy in lines:
        d = np.diff(xy, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        for p, dd, l in zip(xy[:-1], d, L):
            u = np.linspace(0, 1, max(2, int(l/step_m) + 1))
            cols.append(np.rint((p[0] + u*dd[0] - lo)/cell).astype(int))
            rows.append(np.rint((p[1] + u*dd[1] - lo)/cell).astype(int))
    if not rows:
        return np.zeros(0, int), np.zeros(0, int)
    r, c = np.concatenate(rows), np.concatenate(cols)
    ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
    return r[ok], c[ok]


def city_zones(built: np.ndarray, cell: float) -> np.ndarray:
    dens = uniform_filter(built.astype(float), max(1, int(round(CITY_WINDOW_M/cell))))
    core = binary_fill_holes(dens >= CITY_DENSITY)
    lab, _ = label(core)
    area = np.bincount(lab.ravel())*cell**2/1e6
    keep = np.flatnonzero(area >= CITY_MIN_KM2)
    return np.isin(lab, keep[keep > 0])


def road_graph(region: Region, ways, city, axis):
    """Two-tier road graph split at shared OSM nodes; edges inside city zones are cut at the boundary."""
    use = [w for w in ways if road_tier(w["tags"]["highway"])]
    count = {}
    for w in use:
        for k, nid in enumerate(w["nodes"]):
            count[nid] = count.get(nid, 0) + (2 if k in (0, len(w["nodes"]) - 1) else 1)
    cell, lo, n = axis[1] - axis[0], axis[0], len(axis)
    half = -lo
    node_id, node_xy, node_city = {}, [], []
    edges, polys = [], []

    def nid_of(osm_id, x, y):
        if osm_id not in node_id:
            node_id[osm_id] = len(node_xy)
            node_xy.append((x, y))
            r, c = int(round((y - lo)/cell)), int(round((x - lo)/cell))
            node_city.append(bool(0 <= r < n and 0 <= c < n and city[r, c]))
        return node_id[osm_id]

    for w in use:
        t = w["tags"]
        lat = np.array([g[0] for g in w["geom"]]); lon = np.array([g[1] for g in w["geom"]])
        x, y = to_xy(region, lat, lon)
        tier = road_tier(t["highway"])
        hw = HIGHWAY_CODES.index(t["highway"].removesuffix("_link"))
        oneway = 1 if t.get("oneway") in ("yes", "1", "true") else -1 if t.get("oneway") == "-1" else 0
        if t["highway"].startswith("motorway") and "oneway" not in t:
            oneway = 1
        bridge = t.get("bridge") in ("yes", "viaduct")
        start = 0
        for k in range(1, len(w["nodes"])):
            if count.get(w["nodes"][k], 0) < 2 and k < len(w["nodes"]) - 1:
                continue
            xs, ys = x[start:k + 1], y[start:k + 1]
            if (np.abs(xs) <= half).any() and (np.abs(ys) <= half).any():
                u = nid_of(w["nodes"][start], x[start], y[start]); v = nid_of(w["nodes"][k], x[k], y[k])
                L = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
                edges.append((u, v, L, tier, hw, oneway, bridge))
                polys.append(np.column_stack([xs, ys]))
            start = k
    e = np.array(edges, dtype=[("u", "i4"), ("v", "i4"), ("length_m", "f4"), ("tier", "u1"), ("highway", "u1"),
                               ("oneway", "i1"), ("bridge", "?")])
    offs = np.concatenate([[0], np.cumsum([len(p) for p in polys])])
    return dict(node_xy=np.array(node_xy, float), node_city=np.array(node_city, bool), edges=e,
                edge_poly=np.concatenate(polys).astype(np.float32), edge_poly_offset=offs.astype(np.int64),
                highway_codes=np.array(HIGHWAY_CODES))


def build_real_terrain(region: Region, raw_dir: Path, map_m: float = 60000.0, cell_m: float = 50.0,
                       max_slope_deg: float = 32.0):
    """Terrain layers and road graph for the region from the cached raw data."""
    n = int(round(map_m/cell_m)) + 1
    axis = np.linspace(-map_m/2, map_m/2, n)
    kx, ky = scale(region.lat)
    glat, glon = region.lat + axis/ky, region.lon + axis/kx

    d = np.load(raw_dir/"dem.npz")
    z = _sample_grid(np.nan_to_num(d["z"].astype(float), nan=0.0), d["lat"], d["lon"], glat, glon, order=1)
    d = np.load(raw_dir/"worldcover.npz")
    lc = _sample_grid(d["lc"], d["lat"], d["lon"], glat, glon, order=0).astype(np.uint8)
    osm = json.loads((raw_dir/"osm_highways.json").read_text())

    built_all = lc == WC_BUILT
    city = city_zones(built_all, cell_m)
    built = built_all & ~city
    forest = lc == WC_TREE
    water = lc == WC_WATER

    wat = json.loads((raw_dir/"osm_waterways.json").read_text())
    river_lines, stream_lines = [], []
    for w in wat["ways"]:
        lat = np.array([g[0] for g in w["geom"]]); lon = np.array([g[1] for g in w["geom"]])
        xy = np.column_stack(to_xy(region, lat, lon))
        (river_lines if w["tags"]["waterway"] == "river" else stream_lines).append(xy)
    water |= _fill_river_gaps(water, river_lines, axis)
    stream = np.zeros((n, n), bool)
    r, c = _raster_lines(river_lines + stream_lines, axis)
    stream[r, c] = True
    stream &= ~water

    lines = {1: [], 2: []}
    bridge_lines = []
    for w in osm["ways"]:
        lat = np.array([g[0] for g in w["geom"]]); lon = np.array([g[1] for g in w["geom"]])
        xy = np.column_stack(to_xy(region, lat, lon))
        tier = road_tier(w["tags"]["highway"])
        if tier:
            lines[tier].append(xy)
        if w["tags"].get("bridge") in ("yes", "viaduct") and not w["tags"]["highway"].startswith("motorway"):
            bridge_lines.append(xy)
    surface = np.zeros((n, n), np.uint8)
    for tier in (1, 2):
        r, c = _raster_lines(lines[tier], axis)
        surface[r, c] = np.maximum(surface[r, c], tier)
    surface[city] = 0
    r, c = _raster_lines(bridge_lines, axis)
    fords = np.zeros((n, n), bool)
    fords[r, c] = True
    fords &= water
    water &= ~fords
    stream &= ~fords

    cfg = TerrainConfig(seed=0, map_m=map_m, cell_m=cell_m, max_slope_deg=max_slope_deg, mountain_frac=0.0,
                        forest_frac=float(forest.mean()), source=region.key,
                        origin_lat=region.lat, origin_lon=region.lon)
    empty = np.zeros((n, n), bool)
    terr = build_terrain(cfg, axis, z, water, fords, empty, forest, stream, surface, city=city, built=built)
    graph = road_graph(region, osm["ways"], city, axis)
    return terr, graph, lc
