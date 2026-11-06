"""Offline web viewer for tuning the vehicle evader model (replaces the slow matplotlib tuner window).

Run from the repository root, then open the printed address in a browser:
    python -m rl_search.scripts.real_map.vehicle_web            # http://127.0.0.1:8000
    python -m rl_search.scripts.real_map.vehicle_web --port 8001

Only the Python standard library is used for the server, and the page (vehicle_web.html) draws with plain
canvas code, so nothing is fetched from the internet. The simulation is rl_search.vehicle, unchanged; the
browser only draws and plays back what the server returns. Settings are saved to the same file as the
matplotlib tuner (rl_search/data/real/vehicle_params.json). The server listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from dataclasses import asdict, fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.scripts.real_map.vehicle_tuner import GROUPS, PARAMS, STATE_COLORS, load_settings, map_image, summarize
from rl_search.vehicle import STATE_NAMES, VehicleConfig, load_net, simulate_vehicle

HTML = Path(__file__).with_name("vehicle_web.html")
POS_UNIT_M = 10.0          # positions are sent as int16 in 10 m units (+-327 km)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


class App:
    def __init__(self, region: str, px: int):
        t0 = time.time()
        self.net = load_net(region, 25.0)
        img, ext = map_image(self.net, px)
        buf = io.BytesIO()
        plt.imsave(buf, img, format="png")
        self.map_png = buf.getvalue()
        self.extent = [float(v) for v in ext]                 # km: x0, x1, y0, y1
        self.cfg, self.view = load_settings()
        self.lock = threading.Lock()            # the road network object is shared and not thread-safe
        print(f"Map {px} px ready in {time.time() - t0:.1f} s ({len(self.map_png)/1e6:.1f} MB)")

    def meta(self):
        groups = {g: [{"name": n, "label": lab, "min": lo, "max": hi, "step": st} for n, lab, lo, hi, st in items]
                  for g, items in GROUPS.items()}
        values = {f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)}
        values.update(heading=self.view["heading"], seed=self.view["seed"], duration_min=self.cfg.duration_s/60)
        return {"groups": groups, "values": values, "lkp": self.view["lkp"], "extent": self.extent,
                "states": list(STATE_NAMES), "colors": [STATE_COLORS[s] for s in range(len(STATE_NAMES))],
                "params_file": str(PARAMS)}

    def config_from(self, d: dict):
        known = {f.name for f in fields(VehicleConfig)}
        kw = {k: v for k, v in d.items() if k in known}
        if "duration_min" in d:
            kw["duration_s"] = float(d["duration_min"])*60
        if "particles" in kw:
            kw["particles"] = int(kw["particles"])
        if "respect_oneway" in kw:
            kw["respect_oneway"] = bool(kw["respect_oneway"])
        cfg = replace(self.cfg, **kw)
        view = {"heading": float(d.get("heading", self.view["heading"])), "seed": int(d.get("seed", self.view["seed"])),
                "lkp": [float(v) for v in d.get("lkp", self.view["lkp"])]}
        return cfg, view

    def run(self, d: dict):
        cfg, view = self.config_from(d)
        t0 = time.time()
        with self.lock:
            res = simulate_vehicle(self.net, cfg, view["lkp"], view["heading"], np.random.default_rng(view["seed"]))
        sim_s = time.time() - t0
        self.cfg, self.view = cfg, view
        shares, pct = summarize(res)
        pos = np.clip(np.rint(res.track/POS_UNIT_M), -32767, 32767).astype("<i2")
        return {"T": int(res.track.shape[0]), "N": int(res.track.shape[1]), "dt_s": cfg.save_dt_s,
                "pos_unit_m": POS_UNIT_M, "pos": b64(pos), "state": b64(res.state.astype(np.uint8)),
                "speed": b64(np.clip(np.rint(res.speed_kmh.astype(float)), 0, 255).astype(np.uint8)),
                "shares": shares.round(4).tolist(), "pct": np.nan_to_num(pct, nan=-1).round(3).tolist(),
                "lkp_road": [float(v) for v in res.lkp_road], "heading": view["heading"], "sim_s": round(sim_s, 2),
                "speed_cap_kmh": cfg.speed_cap_kmh}

    def save(self, d: dict):
        cfg, view = self.config_from(d)
        self.cfg, self.view = cfg, view
        PARAMS.write_text(json.dumps({"config": asdict(cfg), "view": view}, ensure_ascii=False, indent=1))
        return {"saved": str(PARAMS)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, code=200):
            self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, HTML.read_bytes(), "text/html; charset=utf-8")    # read each time: edit and reload
            elif self.path == "/map.png":
                self.send(200, app.map_png, "image/png")
            elif self.path == "/api/meta":
                self.send_json(app.meta())
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/run":
                    self.send_json(app.run(d))
                elif self.path == "/api/save":
                    self.send_json(app.save(d))
                else:
                    self.send(404, b"not found", "text/plain")
            except Exception as e:  # report the error to the page instead of dropping the connection
                self.send_json({"error": f"{type(e).__name__}: {e}"}, code=500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="chungju")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--map-px", type=int, default=2400, help="Size of the prerendered map image")
    args = ap.parse_args()
    app = App(args.region, args.map_px)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
