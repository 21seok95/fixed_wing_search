"""Bring the slim branch local-pc (what the local PC clones) up to date with feat/rl-env.

The slim branch has no shared history with feat/rl-env and is never merged (user 2026-11-16: updates for the local PC go to that branch,
made from feat/rl-env). Every file on it, except its own README.md and CLAUDE.md, gets feat/rl-env's committed version; --add brings more
paths over (a new module, a new tool) and --remove drops some. The work happens in a temporary worktree: this checkout is not touched,
and edits not yet committed here are not taken (the script warns about them).

Kept small (user 2026-11-18; local-pc replaces local-pc-r13, which stays as it was): results never go on it (model weights, training
logs, TensorBoard files, web runs, figures and reports: EXCLUDE), the OSM downloads of a map's raw data go on it gzipped (raw/osm_*.json
-> raw/osm_*.json.gz; OSM changes over time, so the local PC builds from the Mac's download), and the DEM and land cover do not (fixed
releases that make_real_terrain downloads again; scripts/terrain_fingerprint.py check tells whether the rebuilt map is the Mac's).
--init makes the branch the first time: no history, the files of --seed (another branch's file list) and --add, with these rules.

Before committing, it reads every Python file of the result and lists the rl_search modules that are imported (also inside functions)
but missing on the slim branch. Those imports would fail on the local PC, so add the files with --add. It also finds files edited
directly on the slim branch since the last update (e.g. a fix pushed from the local PC): they would be replaced by feat/rl-env's
version, so it does not commit until that edit is on feat/rl-env too (or --overwrite).

Run from the repository root on the Mac:
    python -m rl_search.scripts.sync_local_pc                                      # what would change; nothing is written
    python -m rl_search.scripts.sync_local_pc --commit -m "compare_flights fix"    # commit on local-pc-r13
    python -m rl_search.scripts.sync_local_pc --commit --push -m "..."             # and push; then git pull on the local PC
    python -m rl_search.scripts.sync_local_pc --add rl_search/scripts/rl_eval/new_tool.py --commit -m "..."
    python -m rl_search.scripts.sync_local_pc --init --seed local-pc-r13 --own-from slim_own/ --add ... --commit -m "..."   # once
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import gzip
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

OWN = {"README.md", "CLAUDE.md"}                                  # the slim branch's own files, never taken from the source branch
# never on the slim branch (user 2026-11-18): results, which come back as export_run zips, and the raw data downloaded again
EXCLUDE = ("rl_search/data/rl_real/*", "*/web_runs/*", "*.png", "*.zip", "*.log", "*/tb/*", "*/tb_eval/*", "*/ckpt/*",
           "rl_search/data/real/*/rl_eval/report/*", "rl_search/data/real/*/d_probe/*", "rl_search/data/export/*",
           "rl_search/data/real/*/raw/dem.npz", "rl_search/data/real/*/raw/worldcover.npz")
OSM_RAW = "rl_search/data/real/*/raw/osm_*.json"                  # gzipped on the slim branch


def excluded(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in EXCLUDE)


def slim_name(path: str) -> str:
    """Name of a source file on the slim branch (OSM downloads gzipped)."""
    return path + ".gz" if fnmatch.fnmatch(path, OSM_RAW) else path


def source_of(path: str, src_files: set[str]) -> str | None:
    """The source file a slim-branch file comes from, or None."""
    if path in src_files:
        return path
    if path.endswith(".json.gz") and path[:-3] in src_files and fnmatch.fnmatch(path[:-3], OSM_RAW):
        return path[:-3]
    return None


def gz_bytes(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)       # mtime 0: the same input gives the same blob
PUSH_URL = "git@github.com:21seok95/fixed_wing_search.git"      # the HTTPS origin has no login on the Mac; SSH works


def git(*args: str, cwd: Path, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args[:3])} ... failed:\n{r.stderr.strip()}")
    return r.stdout


def mod_files(mod: str) -> list[str]:
    p = mod.replace(".", "/")
    return [p + ".py", p + "/__init__.py"]


def blobs(ref: str, root: Path) -> dict[str, str]:
    """path -> blob id of every file of ``ref``."""
    out = {}
    for ln in git("ls-tree", "-r", ref, cwd=root).split("\n"):
        if ln:
            meta, path = ln.split("\t", 1)
            out[path] = meta.split()[2]
    return out


def last_source(dst: str, src: str, root: Path) -> str | None:
    """The --src commit the slim branch was last made from: "Update from <src> <hash>" or, for the first commit, "Cut from <src> (<hash>)"."""
    for msg in git("log", "--format=%B%x00", dst, cwd=root).split("\x00"):
        m = re.search(rf"(?:Update from|Cut from) {re.escape(src)} \(?([0-9a-f]{{7,40}})\b", msg)
        if m:
            return m.group(1)
    return None


def missing_modules(root: Path, src_files: set[str]) -> dict[str, list[str]]:
    """rl_search modules imported anywhere in the Python files under ``root`` (also inside functions, and ``from pkg import module``
    when that module is a file on the source branch) that have no file under ``root``: module -> where it is imported."""
    have = lambda mod: any((root/p).exists() for p in mod_files(mod))
    out: dict[str, list[str]] = {}
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            out.setdefault(f"(cannot read {rel})", []).append(str(e))
            continue
        pkg = list(rel.with_suffix("").parts[:-1])                  # the file's package, for relative imports
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = ".".join(pkg[:len(pkg) - node.level + 1]) if node.level else ""
                mod = ".".join(x for x in (base, node.module or "") if x)
                mods = [mod] + [f"{mod}.{a.name}" for a in node.names if any(p in src_files for p in mod_files(f"{mod}.{a.name}"))]
            for m in mods:
                if m.startswith("rl_search") and not have(m):
                    out.setdefault(m, []).append(f"{rel}:{node.lineno}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="feat/rl-env", help="branch the files come from (its committed state)")
    ap.add_argument("--dst", default="local-pc", help="the slim branch")
    ap.add_argument("--init", action="store_true", help="make --dst the first time: no history, files of --seed and --add")
    ap.add_argument("--seed", default="local-pc-r13", help="with --init: branch whose file list is the starting set")
    ap.add_argument("--own-from", default=None, help="folder with this branch's own README.md and CLAUDE.md (written into --dst)")
    ap.add_argument("--add", nargs="+", default=[], help="more paths to bring over from --src (files or folders)")
    ap.add_argument("--remove", nargs="+", default=[], help="paths to drop from the slim branch")
    ap.add_argument("-m", "--message", default="", help="what the update is for (the commit subject)")
    ap.add_argument("--trailer", action="append", default=[], help="line added at the end of the commit message (repeatable)")
    ap.add_argument("--commit", action="store_true", help="commit on --dst (default: only show what would change)")
    ap.add_argument("--push", action="store_true", help="commit and push --dst")
    ap.add_argument("--allow-missing", action="store_true", help="commit even if imported rl_search modules are missing")
    ap.add_argument("--overwrite", action="store_true", help="commit even if files edited directly on --dst get replaced")
    ap.add_argument("--remote", default=PUSH_URL)
    args = ap.parse_args()
    commit = args.commit or args.push
    root = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()).strip())
    src = git("rev-parse", "--short", "--verify", f"{args.src}^{{commit}}", cwd=root).strip()
    exists = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{args.dst}^{{commit}}"], cwd=root, capture_output=True).returncode == 0
    if exists == args.init:
        raise SystemExit(f"{args.dst} already exists (drop --init)" if exists else f"no branch {args.dst} (make it with --init)")
    src_files = set(git("ls-tree", "-r", "--name-only", args.src, cwd=root).split("\n")) - {""}
    last = last_source(args.dst, args.src, root) if exists else None
    base, now, new = (blobs(last, root) if last else {}), (blobs(args.dst, root) if exists else {}), blobs(args.src, root)
    tmp = Path(tempfile.mkdtemp(prefix="sync_local_pc_"))
    wt = tmp/"wt"
    if exists:
        git("worktree", "add", "-q", str(wt), args.dst, cwd=root)
        files = [f for f in git("ls-files", cwd=wt).split("\n") if f and f not in OWN]
    else:                                                    # a new branch with no history, from the seed's file list
        git("worktree", "add", "-q", "--detach", str(wt), args.src, cwd=root)
        git("checkout", "-q", "--orphan", args.dst, cwd=wt)
        git("rm", "-rf", "-q", "--cached", ".", cwd=wt)
        for f in Path(wt).iterdir():
            if f.name != ".git":
                shutil.rmtree(f) if f.is_dir() else f.unlink()
        files = [slim_name(f) for f in git("ls-tree", "-r", "--name-only", args.seed, cwd=root).split("\n") if f and f not in OWN]
    try:
        added = sorted({slim_name(f) for f in src_files for a in args.add if f == a or f.startswith(a.rstrip("/") + "/")})
        files = sorted(set(files) | set(added))
        drop = [f for f in files if any(f == r or f.startswith(r.rstrip("/") + "/") for r in args.remove) or excluded(f)
                or excluded(source_of(f, src_files) or f)]
        keep = [f for f in files if f not in drop]
        gone = [f for f in keep if source_of(f, src_files) is None]
        take = [f for f in keep if source_of(f, src_files) is not None]
        # edited on the slim branch since the last update and different from --src now: the update would replace that edit
        direct = [f for f in take if f in base and f in src_files and now.get(f) != base[f] and now.get(f) != new.get(f)]
        plain = [f for f in take if f in src_files]
        if plain:
            git("checkout", args.src, "--", *plain, cwd=wt)
        for f in take:
            if f not in src_files:                           # gzipped OSM download
                data = subprocess.run(["git", "show", f"{args.src}:{source_of(f, src_files)}"], cwd=root, capture_output=True, check=True).stdout
                (wt/f).parent.mkdir(parents=True, exist_ok=True)
                (wt/f).write_bytes(gz_bytes(data))
                git("add", "-f", "--", f, cwd=wt)
        if args.own_from:
            for name in OWN:
                (wt/name).write_text((Path(args.own_from)/name).read_text())
                git("add", "-f", "--", name, cwd=wt)
        tracked = set(git("ls-files", cwd=wt).split("\n")) - {""}
        drop = [f for f in drop if f in tracked]
        if drop:
            git("rm", "-q", "--", *drop, cwd=wt)
        # edits in this checkout that are not committed on --src are not taken
        dirty = [ln[3:].split(" -> ")[-1] for ln in git("status", "--porcelain", cwd=root).split("\n") if ln]
        watched = {source_of(f, src_files) for f in keep} - {None}
        unsaved = sorted(set(dirty) & watched)
        stat = git("diff", "--cached", "--stat", cwd=wt).rstrip()
        names = git("diff", "--cached", "--name-status", cwd=wt).split("\n")
        miss = missing_modules(wt, src_files)
        size = sum((wt/f).stat().st_size for f in git("ls-files", cwd=wt).split("\n") if f and (wt/f).exists())
        print(f"{args.dst} <- {args.src} {src}: {len(take)} files, {len(args.add)} paths added, {len(drop)} files removed; "
              f"{size/1e6:.1f} MB on the branch")
        print(stat if stat else "no change: the slim branch already has these versions")
        for f in gone:
            print(f"  not on {args.src} any more (kept; drop it with --remove): {f}")
        for f in unsaved:
            print(f"  WARNING: {f} has edits not committed on {args.src}; they are not taken. Commit them first.")
        if not last:
            print(f"  note: no \"Update from {args.src}\" commit found on {args.dst}; edits made directly there cannot be told apart")
        for f in direct:
            print(f"  EDITED ON {args.dst} since the update from {last}: {f} would be replaced by the {args.src} version. Bring the edit to "
                  f"{args.src} first (there: git checkout {args.dst} -- {f}, check, commit), or pass --overwrite")
        for m, where in sorted(miss.items()):
            hint = f"--add {next(p for p in mod_files(m) if p in src_files)}" if any(p in src_files for p in mod_files(m)) else "not on the source branch either"
            print(f"  MISSING MODULE {m} (imported at {', '.join(where[:3])}{' ...' if len(where) > 3 else ''}): {hint}")
        if not commit:
            print("nothing written (pass --commit to commit, --push to commit and push)")
            return
        if not stat and exists:
            return
        if miss and not args.allow_missing:
            raise SystemExit("not committed: imported rl_search modules are missing on the slim branch (add them, or --allow-missing)")
        if direct and not args.overwrite:
            raise SystemExit(f"not committed: files edited directly on {args.dst} would be replaced (bring them to {args.src}, or --overwrite)")
        subject = f"Update from {args.src} {src}" + (f": {args.message}" if args.message else "")
        body = "\n".join(n for n in names if n)
        msg = f"{subject}\n\nFiles (A added, M changed, D removed):\n{body}\n" + ("\n" + "\n".join(args.trailer) + "\n" if args.trailer else "")
        subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=wt, input=msg, text=True, check=True)
        print(f"committed on {args.dst}: {git('log', '--oneline', '-1', cwd=wt).strip()}")
        if args.push:
            r = subprocess.run(["git", "push", args.remote, f"{args.dst}:{args.dst}"], cwd=wt, capture_output=True, text=True)
            print(r.stderr.strip() or r.stdout.strip())
            if r.returncode:
                raise SystemExit("push failed")
            print("pushed; on the local PC: git pull")
    finally:
        git("worktree", "remove", "--force", str(wt), cwd=root, check=False)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
