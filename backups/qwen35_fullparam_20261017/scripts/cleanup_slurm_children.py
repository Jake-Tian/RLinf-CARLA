"""Stop remaining Ray/CARLA children belonging to this exact Slurm allocation."""
import os
import signal
import time
from pathlib import Path

job = os.environ['SLURM_JOB_ID'].encode()
owned = []
ancestors = set()
pid = os.getpid()
while pid > 1:
    ancestors.add(pid)
    try:
        pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    except (OSError, ValueError):
        break
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) in ancestors:
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        entries = (proc / 'environ').read_bytes().split(b'\0')
        if b'SLURM_JOB_ID=' + job not in entries and b'SLURM_JOBID=' + job not in entries:
            continue
        owned.append((int(proc.name), (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]))
    except (OSError, ValueError, IndexError):
        continue
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid, started in owned:
        try:
            # Guard against PID reuse between discovery and signalling.
            if Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19] == started:
                os.kill(pid, sig)
        except (OSError, ValueError, IndexError):
            pass
    if sig == signal.SIGTERM:
        time.sleep(1)
print(f'cleanup job {job.decode()}: {len(owned)} owned children', flush=True)
