"""Georeference and file list of a real-region map, written next to the map so that the same world can be set up later in PX4 SITL / Gazebo.

The experiments use a local frame: x east, y north in metres from the region centre (equirectangular projection at the centre latitude),
which is the ENU convention of Gazebo. The manifest holds the origin, the conversion, the home position for PX4 and the data licences.
Nothing here depends on PX4 being installed; it has not been run against PX4 yet.

    python -m rl_search.scripts.real_map.world_manifest --region mojave
    python -m rl_search.scripts.real_map.world_manifest --all
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from rl_search.real_scenario import terrain_path
from rl_search.real_terrain import DATA, REGIONS, scale
from rl_search.terrain import load_terrain

ATTRIBUTION = {
    "roads_waterways": "OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
    "elevation": "Copernicus DEM GLO-30, ESA / European Union; surface model, heights relative to the EGM2008 geoid",
    "land_cover": "ESA WorldCover 2021 v200, CC BY 4.0",
}


def manifest(key: str, cell_m: float = 25.0) -> dict:
    region, sfx = REGIONS[key], f"_{cell_m:g}m"
    t = load_terrain(terrain_path(key, cell_m))
    kx, ky = scale(region.lat)
    mid = len(t.axis)//2
    z0 = float(t.z[mid, mid])
    d = DATA/key
    simp = d/f"road_simplify{sfx}.json"
    files = {"terrain": str(terrain_path(key, cell_m)), "road_graph": str(d/f"road_graph{sfx}.npz"), "land_cover": str(d/f"landcover{sfx}.npz"), "raw_data_cache": str(d/"raw")}
    if (d/f"road_graph_full{sfx}.npz").exists():
        files["road_graph_before_simplification"] = str(d/f"road_graph_full{sfx}.npz")
    return dict(
        region=key, name=region.name_ko,
        origin=dict(lat_deg=region.lat, lon_deg=region.lon, elevation_m=round(z0, 2)),
        frame=dict(x="east (m)", y="north (m)", z="up (m, DEM elevation)", convention="ENU, origin at the region centre",
                   projection="equirectangular at the centre latitude", m_per_deg_lon=kx, m_per_deg_lat=ky,
                   to_latlon="lat = lat0 + y/m_per_deg_lat; lon = lon0 + x/m_per_deg_lon"),
        grid=dict(size_m=float(t.axis[-1] - t.axis[0]), cell_m=float(t.cell), cells_per_side=len(t.axis), row0="south", elevation_min_m=float(np.min(t.z)), elevation_max_m=float(np.max(t.z))),
        px4_sitl=dict(PX4_HOME_LAT=region.lat, PX4_HOME_LON=region.lon, PX4_HOME_ALT=round(z0, 2)),
        gazebo=dict(spherical_coordinates=dict(surface_model="EARTH_WGS84", world_frame_orientation="ENU", latitude_deg=region.lat, longitude_deg=region.lon, elevation=round(z0, 2), heading_deg=0),
                    heightmap=dict(source=files["terrain"], layer="z", size_m=float(t.axis[-1] - t.axis[0]))),
        files=files, road_simplification=json.loads(simp.read_text()) if simp.exists() else None, attribution=ATTRIBUTION, tested_with_px4=False)


def main():
    from rl_search.scripts.real_map.vehicle_web import available_regions
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=sorted(REGIONS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for key in (available_regions() if args.all else [args.region]):
        p = DATA/key/"world_manifest.json"
        m = manifest(key)
        p.write_text(json.dumps(m, ensure_ascii=False, indent=1))
        print(p, m["px4_sitl"])


if __name__ == "__main__":
    main()
