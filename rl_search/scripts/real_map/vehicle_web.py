"""Offline web viewer for tuning the vehicle evader model (replaces the slow matplotlib tuner window).

Run from the repository root, then open the printed address in a browser:
    python -m rl_search.scripts.real_map.vehicle_web            # http://127.0.0.1:8000
    python -m rl_search.scripts.real_map.vehicle_web --port 8001

Only the Python standard library is used for the server, and the page (vehicle_web.html) draws with plain
canvas code, so nothing is fetched from the internet. The simulation is rl_search.vehicle, unchanged; the
browser only draws and plays back what the server returns. Search runs (SAROPS, web_optimize.py) go through a queue,
each in its own process; finished runs are saved in rl_search/data/real/chungju/web_runs/ and can be replayed later. Settings are saved to the same file as the
matplotlib tuner (rl_search/data/real/vehicle_params.json). The server listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import multiprocessing as mp
import queue
import shutil
import threading
import time
from dataclasses import asdict, fields, replace
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_search.real_terrain import DATA, REGIONS
from rl_search.scripts.real_map.web_optimize import AREA_DEFS, CAPS, FIELDS, SCORE_GROUP, SCORES, particles_payload, worker
from rl_search.scripts.real_map.vehicle_tuner import DEFAULT_LKP, GROUPS, PARAMS, STATE_COLORS, load_settings, map_image
from rl_search.real_scenario import chungju_preset
from rl_search.scenario import ScenarioSampler
from rl_search.vehicle import STATE_NAMES, VehicleConfig, load_net, simulate_vehicle

HTML = Path(__file__).with_name("vehicle_web.html")
RUNS = DATA/"chungju"/"web_runs"
DESERTS = ("mojave", "amboy")                            # regions where the off-road state is not "paddy fields"
WEB_STATE = DATA/"chungju"/"web_state.json"     # what the page is set to now (rewritten at every particle run)


def run_meta(o: dict, rid: str, label: str, created: str, view: dict, config: dict, **extra) -> dict:
    """What the run list shows for a finished run (``o``: the run_case payload's opt)."""
    return dict(id=rid, label=label, created=created, view=view, config=config, settings=o["settings"], final=o["final"], plans=sum(len(u["rects"]) for u in o["uavs"]),
                n_uav=len(o["uavs"]), area=o["area"]["A"], escaped=o["escaped"], sec=o["sec"],
                pos=[round(float(np.interp(m*60, o["pos_t"], o["pos"], left=0)), 4) for m in range(61)], **extra)


def save_run(rid: str, data: dict, meta: dict) -> None:
    """Write a run (replay payload + meta) under RUNS/<rid>/ so the page can list and replay it."""
    d = RUNS/rid
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d/"run.json.gz", "wt", encoding="utf-8", compresslevel=5) as f:
        json.dump({**data, "meta": meta}, f, ensure_ascii=False)
    (d/"meta.json").write_text(json.dumps(meta, ensure_ascii=False))


class Jobs:
    """Queue of SAROPS runs, each in its own process (about 1.5 GB, 1-3 min), at most ``max_jobs`` at once. A finished run is
    saved under RUNS (run.json.gz = particles + flight for replay, meta.json = what the list shows) and survives a restart."""

    def __init__(self, max_jobs: int):
        self.ctx, self.max = mp.get_context("spawn"), max_jobs
        self.jobs, self.count, self.lock = {}, 0, threading.RLock()
        RUNS.mkdir(parents=True, exist_ok=True)
        for f in sorted(RUNS.glob("*/meta.json")):
            m = json.loads(f.read_text())
            self.jobs[m["id"]] = {**m, "state": "done"}

    def submit(self, items: list[dict]):
        ids = []
        with self.lock:
            for it in items:
                self.count += 1
                jid = f"{time.strftime('%Y%m%d-%H%M%S')}-{self.count:03d}"
                self.jobs[jid] = dict(id=jid, label=it["label"], created=time.strftime("%m-%d %H:%M"), state="queued", params=it["params"])
                ids.append(jid)
            self._fill()
        return {"ids": ids}

    def _fill(self):
        with self.lock:
            running = sum(j["state"] == "running" for j in self.jobs.values())
            for j in self.jobs.values():
                if running >= self.max:
                    break
                if j["state"] == "queued":
                    q = self.ctx.Queue()
                    j["proc"], j["t0"], j["state"] = self.ctx.Process(target=worker, args=(j["params"], q), daemon=True), time.time(), "running"
                    j["proc"].start()
                    threading.Thread(target=self._pump, args=(j, j["proc"], q), daemon=True).start()
                    running += 1

    def _pump(self, j, proc, q):
        while True:
            try:
                m = q.get(timeout=.5)
            except queue.Empty:
                if not proc.is_alive():
                    if j.get("proc") is proc and j["state"] == "running":
                        j["state"], j["msg"] = "error", f"계산 프로세스가 끝났습니다 (exit {proc.exitcode})"
                        self._fill()
                    return
                continue
            if j.get("proc") is not proc:
                return
            if m["kind"] == "progress":
                j.update(stage=m["stage"], t_min=m["t_min"], plans=m["plans"])
            elif m["kind"] == "result":
                self._save(j, m["data"])
                self._fill()
                return
            else:
                j["state"], j["msg"] = "error", m["msg"]
                self._fill()
                return

    def _save(self, j, data):
        meta = run_meta(data["opt"], j["id"], j["label"], j["created"], j["params"]["view"], j["params"]["config"], group="web")
        save_run(j["id"], data, meta)
        with self.lock:
            for k in ("params", "proc", "t0", "stage", "t_min"):
                j.pop(k, None)
            j.update(meta, state="done")

    def _scan(self):
        """Pick up runs that other scripts saved (e.g. the RL validation references) since the server started."""
        for f in RUNS.glob("*/meta.json"):
            if f.parent.name not in self.jobs:
                try:
                    m = json.loads(f.read_text())
                except (OSError, ValueError):
                    continue
                self.jobs[m["id"]] = {**m, "state": "done"}

    def list(self):
        with self.lock:
            self._scan()
            out = []
            for j in self.jobs.values():
                v = {k: x for k, x in j.items() if k not in ("params", "proc", "t0")}
                if j["state"] == "running":
                    v["elapsed"] = round(time.time() - j["t0"], 1)
                out.append(v)
        return {"jobs": out[::-1], "max_jobs": self.max}

    def stop(self, jid: str):
        with self.lock:
            j = self.jobs[jid]
            p = j.get("proc")
            if j["state"] in ("queued", "running"):
                j["proc"], j["state"] = None, "cancelled"
                if p is not None and p.is_alive():
                    p.terminate()
            self._fill()
        return {"ok": True}

    def delete(self, jid: str):
        self.stop(jid) if self.jobs[jid]["state"] in ("queued", "running") else None
        with self.lock:
            self.jobs.pop(jid)
        shutil.rmtree(RUNS/jid, ignore_errors=True)
        return {"ok": True}

    def path(self, jid: str) -> Path:
        if jid not in self.jobs or self.jobs[jid]["state"] != "done":
            raise KeyError(jid)
        return RUNS/jid/"run.json.gz"


def available_regions() -> list[str]:
    """Regions whose 25 m terrain, road graph and land cover have been built (make_real_terrain --cell 25)."""
    from rl_search.real_scenario import terrain_path
    return [k for k in REGIONS if terrain_path(k).exists() and (DATA/k/"road_graph_25m.npz").exists() and (DATA/k/"landcover_25m.npz").exists()]


class App:
    def __init__(self, region: str | None, px: int, max_jobs: int = 4):
        self.px = px
        self.cfg, self.view = load_settings()
        self.lock, self.load_lock = threading.Lock(), threading.Lock()   # the road network object is shared and not thread-safe
        self.jobs = Jobs(max_jobs)
        self.scfg = chungju_preset().scenario        # launch points on the map edge, as the optimiser draws them
        self.available = available_regions()
        self.regions = {}
        saved = self.view.get("region", "chungju")
        self.default = region or (saved if saved in self.available else "chungju")
        self.region(self.default)

    def region(self, key: str):
        """Road network and map picture of a region, loaded the first time it is asked for."""
        with self.load_lock:
            if key not in self.regions:
                if key not in self.available:
                    raise ValueError(f"지도가 없습니다: {key}")
                t0 = time.time()
                net = load_net(key, 25.0)
                img, ext = map_image(net, self.px)
                buf = io.BytesIO()
                plt.imsave(buf, img, format="png")
                self.regions[key] = SimpleNamespace(key=key, name=REGIONS[key].name_ko, net=net, map_png=buf.getvalue(), extent=[float(v) for v in ext],
                                                    launch=SimpleNamespace(cfg=self.scfg, perimeter=ScenarioSampler.perimeter_points(net.terr.axis, self.scfg)))
                print(f"Map {key}: {self.px} px ready in {time.time() - t0:.1f} s ({len(self.regions[key].map_png)/1e6:.1f} MB)", flush=True)
            return self.regions[key]

    def default_lkp(self, key: str):
        if key == self.view.get("region", "chungju"):
            return self.view["lkp"]
        return list(DEFAULT_LKP) if key == "chungju" else [0.0, 0.0]

    def region_info(self, key: str):
        r = self.region(key)
        t = r.net.terr
        return {"key": key, "name": r.name, "extent": r.extent, "lkp": self.default_lkp(key),
                "grid": {"n": len(t.axis), "x0": float(t.axis[0] - t.cell/2), "cell": float(t.cell)}}

    def slope_bytes(self, key: str) -> bytes:
        """Per cell of the map (row 0 = south), gzip: slope in 1/4 deg, 254 = road (never closed), 255 = closed at any slope limit (forest, water, city).
        The page compares it with the slope limit, the same rule as ``RoadNet.set_drive_mask``."""
        r = self.region(key)
        if not hasattr(r, "slope_gz"):
            t = r.net.terr
            v = np.floor(np.minimum(t.slope_deg, 63.25)*4).astype(np.uint8)
            v[t.forest | t.water | t.fords] = 255
            v[t.surface > 0] = 254
            v[t.city] = 255
            r.slope_gz = gzip.compress(v.tobytes(), 5)
        return r.slope_gz

    def meta(self):
        groups = {g: [{"name": n, "label": lab, "min": lo, "max": hi, "step": st} for n, lab, lo, hi, st in items]
                  for g, items in GROUPS.items()}
        values = {f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)}
        values.update(heading=self.view["heading"], seed=self.view["seed"], duration_min=self.cfg.duration_s/60)
        return {"groups": groups, "values": values, "lkp": self.default_lkp(self.default), "extent": self.region(self.default).extent,
                "region": self.default, "regions": [dict(key=k, name=REGIONS[k].name_ko, off_name="도로 밖" if k in DESERTS else "논밭") for k in self.available],
                "states": list(STATE_NAMES), "colors": [STATE_COLORS[s] for s in range(len(STATE_NAMES))],
                "params_file": str(PARAMS), "opt_fields": FIELDS, "opt_caps": CAPS, "opt_scores": SCORES, "opt_area_defs": AREA_DEFS, "opt_score_group": SCORE_GROUP}

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
                "lkp": [float(v) for v in d.get("lkp", self.view["lkp"])], "region": d.get("region", self.view.get("region", "chungju"))}
        return cfg, view

    def run(self, d: dict):
        cfg, view = self.config_from(d)
        t0 = time.time()
        reg = self.region(view["region"])
        with self.lock:
            res = simulate_vehicle(reg.net, cfg, view["lkp"], view["heading"], np.random.default_rng(view["seed"]))
        sim_s = time.time() - t0
        self.cfg, self.view = cfg, view
        auto, _ = ScenarioSampler.sample_start(reg.launch, np.random.default_rng(view["seed"] + 10007), res.lkp_road)
        WEB_STATE.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), view=view, config=asdict(cfg)), ensure_ascii=False, indent=1))
        return particles_payload(res, cfg, view["heading"], auto, sim_s)

    def submit(self, d: dict):
        """Queue runs: ``n_seeds`` particle seeds from the seed on the page, with the same settings."""
        cfg, view = self.config_from(d)
        reg = self.region(view["region"])
        start = d.get("start")
        if start is not None:
            start = [float(v) for v in start]
            if max(abs(start[0]), abs(start[1])) > float(reg.net.terr.axis[-1]):
                raise ValueError("출발점이 지도 밖입니다")
        n = max(1, int(d.get("n_seeds", 1)))
        name = (d.get("label") or "").strip() or f"{reg.name} · LKP {view['lkp'][0]/1000:.1f}, {view['lkp'][1]/1000:.1f} km"
        items = []
        for i in range(n):
            v = {**view, "seed": view["seed"] + i, "start": start}
            items.append(dict(label=f"{name} · seed {v['seed']}", params=dict(config=asdict(cfg), view=v, opt=d.get("opt", {}))))
        return self.jobs.submit(items)

    def share(self, d: dict):
        """Save what is on screen for discussing it: the map as PNG, all settings and the note, in RUNS/_shared/<time>/."""
        out = RUNS/"_shared"/time.strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        (out/"scene.png").write_bytes(base64.b64decode(d["png"].split(",", 1)[1]))
        (out/"state.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), note=d.get("note", ""), **d["state"]), ensure_ascii=False, indent=1))
        return {"saved": str(out)}

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
            url = urlsplit(self.path)
            q = parse_qs(url.query)
            if self.path in ("/", "/index.html"):
                self.send(200, HTML.read_bytes(), "text/html; charset=utf-8")    # read each time: edit and reload
            elif url.path == "/map.png":
                self.send(200, app.region(q.get("region", [app.default])[0]).map_png, "image/png")
            elif url.path == "/slope.bin":
                body = app.slope_bytes(q["region"][0])
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/region":
                self.send_json(app.region_info(q["key"][0]))
            elif self.path == "/api/meta":
                self.send_json(app.meta())
            elif self.path == "/api/jobs":
                self.send_json(app.jobs.list())
            elif self.path.startswith("/api/jobs/"):
                try:
                    body = app.jobs.path(self.path.split("/")[3]).read_bytes()
                except (KeyError, OSError):
                    self.send(404, b"not found", "text/plain")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
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
                elif self.path == "/api/share":
                    self.send_json(app.share(d))
                elif self.path == "/api/jobs":
                    self.send_json(app.submit(d))
                elif self.path.startswith("/api/jobs/") and self.path.endswith("/stop"):
                    self.send_json(app.jobs.stop(self.path.split("/")[3]))
                elif self.path.startswith("/api/jobs/") and self.path.endswith("/delete"):
                    self.send_json(app.jobs.delete(self.path.split("/")[3]))
                else:
                    self.send(404, b"not found", "text/plain")
            except Exception as e:  # report the error to the page instead of dropping the connection
                self.send_json({"error": f"{type(e).__name__}: {e}"}, code=500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=None, help="Map shown first (default: the last saved one, or Chungju)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--map-px", type=int, default=2400, help="Size of the prerendered map image")
    ap.add_argument("--opt-jobs", type=int, default=4, help="SAROPS runs at once (each about 1.5 GB)")
    args = ap.parse_args()
    app = App(args.region, args.map_px, args.opt_jobs)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
