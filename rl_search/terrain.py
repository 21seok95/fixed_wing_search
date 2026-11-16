"""Fixed search-area terrain for the RL search experiments.

Procedural 60 km x 60 km terrain: elongated mountain ranges on uplands, rolling hills,
rivers that follow the drainage network of the elevation, fords along the rivers and
forest that favours high and steep ground. Rivers, mountains above the no-go elevation
and steep slopes are impassable for the target, as in battlefield_infiltrator_demo.py.
A ``surface`` layer is reserved for the vehicle target model (0 = off-road,
1 = unpaved road, 2 = paved road; all off-road for now).

Real-region terrain (``rl_search.real_terrain``) uses the same format: ``surface`` holds the road
tiers (1 = slow road, 2 = fast road), ``fords`` holds bridge cells over water, ``mountain`` is empty
and two optional layers are added: ``city`` (large built-up areas, hand-off to ground units) and
``built`` (other built-up land, which counts as cover like forest). Files without them load as zeros.

Only the base layers are stored; slopes, cover, clearance and gradients are derived on load.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import heapq
import json
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, map_coordinates

OFF_ROAD, UNPAVED, PAVED = 0, 1, 2
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class TerrainConfig:
    seed: int = 7
    map_m: float = 60000.0
    cell_m: float = 50.0
    max_slope_deg: float = 32.0
    mountain_frac: float = 0.08           # share of the map above the no-go elevation
    forest_frac: float = 0.35             # forest share of passable land
    river_area_km2: float = 150.0         # drainage area at which a stream becomes an impassable river
    stream_area_km2: float = 20.0         # drainage area of the smallest mapped stream
    ford_spacing_m: float = 5000.0
    ford_half_width_m: float = 150.0
    range_angle_deg: float | None = None  # strike of the mountain ranges; None = random from the seed
    tilt_m_per_km: float = 2.5            # regional slope that organises the drainage
    lake_depth_m: float = 80.0            # deeper filled depressions are lakes, shallower ones plains
    lake_min_km2: float = 3.0
    hydro_cell_m: float = 200.0           # grid of the drainage computation
    source: str = "synthetic"             # "synthetic" or the real region key
    origin_lat: float | None = None       # real regions: map centre (x = east, y = north, metres)
    origin_lon: float | None = None


@dataclass
class Terrain:
    cfg: TerrainConfig
    axis: np.ndarray
    z: np.ndarray
    water: np.ndarray
    fords: np.ndarray
    mountain: np.ndarray
    forest: np.ndarray
    stream: np.ndarray
    surface: np.ndarray
    cover: np.ndarray
    cover_gx: np.ndarray
    cover_gy: np.ndarray
    tpi: np.ndarray
    forbidden: np.ndarray
    slope_deg: np.ndarray
    clearance: np.ndarray
    clearance_gx: np.ndarray
    clearance_gy: np.ndarray
    z_gx: np.ndarray
    z_gy: np.ndarray
    cell: float
    mountain_z_m: float
    city: np.ndarray | None = None
    built: np.ndarray | None = None
    _float_cache: dict = field(default_factory=dict, repr=False)

    def sample(self, fld: np.ndarray, x, y) -> np.ndarray:
        """Bilinear map lookup; scalar or vector locations, metric coordinates."""
        if fld.dtype != np.float64:
            if id(fld) not in self._float_cache:
                self._float_cache[id(fld)] = fld.astype(float)
            fld = self._float_cache[id(fld)]
        xx, yy = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        rows = (yy - self.axis[0]) / self.cell
        cols = (xx - self.axis[0]) / self.cell
        vals = map_coordinates(fld, [rows.ravel(), cols.ravel()], order=1,
                               mode="constant", cval=np.nan)
        return vals.reshape(xx.shape)

    def inside(self, x, y, margin_m: float = 0.0) -> np.ndarray:
        lo, hi = self.axis[0] + margin_m, self.axis[-1] - margin_m
        return (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)


def spectral_noise(n, cell, rng, beta, l_min, l_max, stretch=1.0, angle=0.0) -> np.ndarray:
    """Zero-mean, unit-std field with power spectrum ~ k^-beta between wavelengths l_min and l_max (m).

    ``stretch`` > 1 elongates features along ``angle`` (rad). The FFT grid is padded so the
    periodic seams fall outside the map.
    """
    m = int(np.ceil(1.25*n/64))*64
    ky = np.fft.fftfreq(m, d=cell)[:, None]
    kx = np.fft.rfftfreq(m, d=cell)[None, :]
    c, s = np.cos(angle), np.sin(angle)
    k = np.hypot((kx*c + ky*s)*stretch, -kx*s + ky*c)
    k = np.where(k == 0, np.inf, k)
    amp = k**(-beta/2)*np.exp(-(k*l_min)**2)*(1 - np.exp(-(k*l_max)**2))
    spec = amp*(rng.normal(size=amp.shape) + 1j*rng.normal(size=amp.shape))
    f = np.fft.irfft2(spec, s=(m, m))[:n, :n]
    return (f - f.mean())/f.std()


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x*x*(3 - 2*x)


def fill_depressions(h: np.ndarray, eps: float = 1e-4, jitter=None) -> np.ndarray:
    """Priority-flood fill from the map edge so every cell drains to the boundary.

    ``jitter`` (per-cell random increments, same shape as h) is added inside filled flats; the
    result is then only a routing surface whose flood order, and so the flow paths across flats,
    is irregular instead of straight.
    """
    rows, cols = h.shape
    filled = h.astype(float).ravel().tolist()
    jit = [0.0]*h.size if jitter is None else jitter.ravel().tolist()
    seen = np.zeros(h.shape, bool)
    seen[0, :] = seen[-1, :] = seen[:, 0] = seen[:, -1] = True
    heap = [(filled[k], k) for k in np.flatnonzero(seen.ravel()).tolist()]
    heapq.heapify(heap)
    seen = seen.ravel().tolist()
    while heap:
        e, k = heapq.heappop(heap)
        i, j = divmod(k, cols)
        for di, dj in NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < rows and 0 <= b < cols:
                q = a*cols + b
                if not seen[q]:
                    seen[q] = True
                    if filled[q] <= e:
                        filled[q] = e + eps + jit[q]
                    heapq.heappush(heap, (filled[q], q))
    return np.array(filled).reshape(h.shape)


def d8_receivers(h: np.ndarray) -> np.ndarray:
    """Flat index of the steepest-descent neighbour of every cell; -1 on the map edge (outlets)."""
    rows, cols = h.shape
    pad = np.pad(h, 1, constant_values=np.inf)
    ii, jj = np.indices(h.shape)
    best = np.zeros(h.shape)
    recv = np.full(h.shape, -1)
    for di, dj in NEIGHBOURS:
        drop = (h - pad[1+di:1+di+rows, 1+dj:1+dj+cols])/np.hypot(di, dj)
        take = drop > best
        best[take] = drop[take]
        recv[take] = ((ii + di)*cols + jj + dj)[take]
    recv[0, :] = recv[-1, :] = recv[:, 0] = recv[:, -1] = -1
    return recv.ravel()


def flow_accumulation(h: np.ndarray, recv: np.ndarray) -> np.ndarray:
    """Number of upstream cells (including itself) draining through each cell."""
    acc = [1.0]*h.size
    r = recv.tolist()
    for k in np.argsort(-h.ravel(), kind="stable").tolist():
        if r[k] >= 0:
            acc[r[k]] += acc[k]
    return np.array(acc)


def make_terrain(cfg: TerrainConfig) -> Terrain:
    """Procedural terrain (synthetic, not a DEM); deterministic for a given config."""
    rng = np.random.default_rng(cfg.seed)
    n = int(round(cfg.map_m/cfg.cell_m)) + 1
    axis = np.linspace(-cfg.map_m/2, cfg.map_m/2, n)
    cell = float(axis[1] - axis[0])
    theta = np.deg2rad(cfg.range_angle_deg if cfg.range_angle_deg is not None else rng.uniform(0, 180))

    # Elevation: broad uplands and lowlands, rolling hills, ridged ranges on the uplands and a
    # regional tilt so the drainage organises into a few river systems
    x, y = np.meshgrid(axis, axis)
    macro = spectral_noise(n, cell, rng, 3.0, 8000, 60000)
    hills = spectral_noise(n, cell, rng, 2.6, 200, 5000)
    ridge_src = spectral_noise(n, cell, rng, 2.6, 1500, 20000, stretch=2.5, angle=theta)
    ridges = np.clip(1 - np.abs(ridge_src)/2.2, 0, 1)**3
    upland = smoothstep((macro + .3)/1.5)
    phi = rng.uniform(0, 2*np.pi)
    tilt = cfg.tilt_m_per_km*(x*np.cos(phi) + y*np.sin(phi))/1000
    z = 150 + 50*macro + 22*hills + upland*(40 + 900*ridges) + tilt

    # Drainage on a coarser grid; rivers follow the steepest-descent network to the map edge and
    # deep filled depressions become lakes
    f = max(1, int(round(cfg.hydro_cell_m/cell)))
    zc = z[::f, ::f]
    filled = fill_depressions(zc)
    lake_c = (filled - zc) > cfg.lake_depth_m
    lab, n_lab = label(lake_c)
    if n_lab:
        size_km2 = np.bincount(lab.ravel())*(f*cell)**2/1e6
        lake_c = (size_km2 >= cfg.lake_min_km2)[lab] & lake_c
    up = np.minimum(np.rint(np.arange(n)/f).astype(int), zc.shape[0]-1)
    lake = gaussian_filter(lake_c[np.ix_(up, up)].astype(float), f) > .5
    # Shallower basins are sediment-filled plains: raise the fine grid to the filled surface
    rr, cc = np.meshgrid(np.arange(n)/f, np.arange(n)/f, indexing="ij")
    z = np.maximum(z, map_coordinates(filled, [rr, cc], order=1, mode="nearest"))
    route = fill_depressions(zc, jitter=rng.uniform(0, .5, zc.shape))
    recv = d8_receivers(route)
    acc_km2 = flow_accumulation(route, recv)*(f*cell)**2/1e6
    ci, cj = np.divmod(np.arange(zc.size), zc.shape[1])
    edge = (ci == 0) | (ci == zc.shape[0]-1) | (cj == 0) | (cj == zc.shape[1]-1)
    jitter = np.where(edge[:, None], 0.0, rng.uniform(-.35, .35, (zc.size, 2))*f*cell)
    node_x = axis[np.minimum(cj*f, n-1)] + jitter[:, 0]
    node_y = axis[np.minimum(ci*f, n-1)] + jitter[:, 1]

    reach = np.flatnonzero((acc_km2 >= cfg.stream_area_km2) & (recv >= 0))
    t = np.linspace(0, 1, 2*f + 1)[None, :]
    px = node_x[reach, None] + t*(node_x[recv[reach], None] - node_x[reach, None])
    py = node_y[reach, None] + t*(node_y[recv[reach], None] - node_y[reach, None])
    pr = np.clip(np.rint((py - axis[0])/cell).astype(int), 0, n-1).ravel()
    pc = np.clip(np.rint((px - axis[0])/cell).astype(int), 0, n-1).ravel()
    pacc = np.repeat(acc_km2[reach], t.shape[1])
    stream_c = np.zeros((n, n), bool)
    stream_c[pr, pc] = True
    centre_acc = np.zeros((n, n))
    np.maximum.at(centre_acc, (pr, pc), pacc)
    river_c = centre_acc >= cfg.river_area_km2

    water = np.zeros((n, n), bool)
    d_river = np.full((n, n), np.inf)
    if river_c.any():
        d_cells, (ri, rj) = distance_transform_edt(~river_c, return_indices=True)
        d_river = d_cells*cell
        half_w = np.clip(35*np.sqrt(centre_acc[ri, rj]/cfg.river_area_km2), 35, 110)
        water = gaussian_filter((d_river <= half_w).astype(float), 1.0) > .5
    d_stream = distance_transform_edt(~stream_c)*cell
    z = z - 18*np.exp(-(d_river/350)**2) - 6*np.exp(-(d_stream/150)**2)
    z = z - z.min() + 20
    water |= lake

    mountain = (z >= np.quantile(z, 1 - cfg.mountain_frac)) & ~water

    fords = np.zeros((n, n), bool)
    if river_c.any():
        cand = np.argwhere(river_c & ~mountain & (distance_transform_edt(~lake)*cell > 500))
        cand = cand[rng.permutation(len(cand))]
        picked = cand[:0]
        for p in cand:
            if not len(picked) or np.hypot(*(picked - p).T).min()*cell >= cfg.ford_spacing_m:
                picked = np.vstack([picked, p])
        if len(picked):
            centres = np.zeros((n, n), bool)
            centres[picked[:, 0], picked[:, 1]] = True
            fords = water & (distance_transform_edt(~centres)*cell <= cfg.ford_half_width_m)
    water &= ~fords
    stream = stream_c & ~water & ~fords

    # Forest on high and steep ground; floodplains along the rivers are cleared farmland
    dzdy, dzdx = np.gradient(z, cell, cell)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    z_lo, z_hi = np.quantile(z, .3), np.quantile(z, 1 - cfg.mountain_frac)
    elev = np.clip((z - z_lo)/(z_hi - z_lo), 0, 1)
    farm = np.exp(-(d_river/1500)**2)*(1 - np.clip(slope/12, 0, 1))
    score = spectral_noise(n, cell, rng, 2.0, 80, 3000) + 1.8*elev + .6*np.clip(slope/15, 0, 1) - 1.2*farm
    passable = ~(water | fords | mountain)
    forest = (score > np.quantile(score[passable], 1 - cfg.forest_frac)) & passable
    surface = np.full((n, n), OFF_ROAD, np.uint8)
    return build_terrain(cfg, axis, z, water, fords, mountain, forest, stream, surface)


def build_terrain(cfg: TerrainConfig, axis, z, water, fords, mountain, forest, stream, surface,
                  city=None, built=None) -> Terrain:
    """Derive cover, slope, no-go mask, clearance and gradients from the base layers."""
    cell = float(axis[1] - axis[0])
    city = np.zeros(z.shape, bool) if city is None else city.astype(bool)
    built = np.zeros(z.shape, bool) if built is None else built.astype(bool)
    hide = forest | built
    tpi = z - gaussian_filter(z, 500/cell)
    cover = gaussian_filter(hide.astype(float), 40/cell)
    # Wide-scale cover field so agents far from forest still sense where it is
    cover_far = gaussian_filter(hide.astype(float), 320/cell)
    cgy_c, cgx_c = np.gradient(cover_far, cell, cell)
    dzdy, dzdx = np.gradient(z, cell, cell)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    forbidden = water | mountain | (slope_deg >= cfg.max_slope_deg)
    clearance = distance_transform_edt(~forbidden)*cell
    cgy, cgx = np.gradient(clearance, cell, cell)
    mountain_z = float(z[mountain].min()) if mountain.any() else float("inf")
    return Terrain(cfg, axis, z, water, fords, mountain, forest, stream, surface, cover, cgx_c, cgy_c,
                   tpi, forbidden, slope_deg, clearance, cgx, cgy, dzdx, dzdy, cell, mountain_z, city, built)


BASE_LAYERS = ("z", "water", "fords", "mountain", "forest", "stream", "surface")
EXTRA_LAYERS = ("city", "built")   # optional; saved only when any cell is set


def save_terrain(path: Path, terr: Terrain) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {k: getattr(terr, k) for k in EXTRA_LAYERS if getattr(terr, k) is not None and getattr(terr, k).any()}
    np.savez_compressed(path, config=json.dumps(asdict(terr.cfg)), axis=terr.axis,
                        **{k: getattr(terr, k) for k in BASE_LAYERS}, **extra)


def load_terrain(path: Path) -> Terrain:
    d = np.load(path)
    cfg = TerrainConfig(**json.loads(str(d["config"])))
    return build_terrain(cfg, d["axis"], *(d[k] for k in BASE_LAYERS),
                         **{k: d[k] for k in EXTRA_LAYERS if k in d.files})
