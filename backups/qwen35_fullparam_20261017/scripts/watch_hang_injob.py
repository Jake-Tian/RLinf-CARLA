"""In-job stall watchdog for the 2026-10-16 six-actor session.

Polls run.log/metrics.log/TensorBoard freshness; after RLINF_HANG_SILENCE_S
(default 600s) without progress it dumps py-spy/gdb stacks, nvidia-smi and
process tables into RESULT_ROOT/hang_watchdog_<epoch>/, writes
stall_detected.json, and (unless RLINF_HANG_AUTOKILL=0) scancel's this job.
One dump per continuous silence episode; progress resets the episode.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
silence_limit = float(os.environ.get("RLINF_HANG_SILENCE_S", "600"))
autokill = os.environ.get("RLINF_HANG_AUTOKILL", "1") == "1"
job = os.environ.get("SLURM_JOB_ID", "")
pyspy = "/research/d7/gds/yztian25/RLinf/.venv-starvla-libero/bin/py-spy"
dumped_epoch = None


def progress_files():
    paths = [root / "run.log", root / "metrics.log"]
    paths += list((root / "tensorboard").rglob("events.out.tfevents.*"))
    return [p for p in paths if p.exists()]


def silence_seconds():
    files = progress_files()
    if not files:
        return None
    return time.time() - max(p.stat().st_mtime for p in files)


def candidate_processes():
    procs = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if cmd and ("train_async_carla" in cmd or "ray::" in cmd):
            procs.append((int(entry.name), cmd))
    return procs


def dump(seconds):
    stamp = int(time.time())
    out = root / f"hang_watchdog_{stamp}"
    out.mkdir(exist_ok=True)
    procs = candidate_processes()
    record = {
        "time_epoch": stamp,
        "time_beijing": time.strftime("%F %T", time.localtime(stamp)),
        "seconds_without_progress": seconds,
        "criterion": f"run.log/metrics.log/tensorboard unchanged for {silence_limit:.0f}s",
        "job_id": job,
        "processes": [{"pid": pid, "cmdline": cmd} for pid, cmd in procs],
    }
    (out / "processes.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    for pid, cmd in procs[:6]:
        with (out / f"pyspy_{pid}.txt").open("w") as stream:
            try:
                subprocess.run([pyspy, "dump", "--pid", str(pid)], stdout=stream, stderr=subprocess.STDOUT, timeout=30)
            except (OSError, subprocess.TimeoutExpired) as exc:
                stream.write(f"py-spy failed: {exc}\n")
        try:
            with (out / f"gdb_{pid}.txt").open("w") as stream:
                subprocess.run(["/usr/bin/gdb", "-q", "-batch", "-ex", "set pagination off",
                                "-ex", "thread apply all bt", "-p", str(pid)],
                               stdout=stream, stderr=subprocess.STDOUT, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
    (out / "nvidia_smi.txt").write_text(
        subprocess.run(["nvidia-smi"], text=True, capture_output=True).stdout)
    (out / "cpu.txt").write_text(
        subprocess.run(["ps", "-eo", "pid,ppid,stat,etimes,pcpu,pmem,args", "--sort=-pcpu"],
                       text=True, capture_output=True).stdout[:80000])
    for name in ("metrics.log", "run.log"):
        path = root / name
        if path.exists():
            (out / f"{name}.tail").write_text(
                subprocess.run(["tail", "-60", str(path)], text=True, capture_output=True).stdout)
    (root / "stall_detected.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({"event": "stall_detected", "dir": str(out), **{k: record[k] for k in
          ("seconds_without_progress", "job_id")}}, ensure_ascii=False), flush=True)
    if autokill and job:
        subprocess.run(["scancel", job], check=False)
        print(json.dumps({"event": "scancel_sent", "job_id": job}), flush=True)


while True:
    time.sleep(30)
    silent = silence_seconds()
    if silent is None:
        continue
    if silent < silence_limit:
        if dumped_epoch is not None:
            print(json.dumps({"event": "progress_resumed", "silence_s": round(silent, 1)}), flush=True)
        dumped_epoch = None
        continue
    if dumped_epoch == "reported":
        continue
    dump(silent)
    dumped_epoch = "reported"
