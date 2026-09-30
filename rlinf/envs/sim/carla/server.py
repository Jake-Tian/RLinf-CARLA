# Copyright 2026 The RLinf-CARLA Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CARLA server process lifecycle.

One CarlaServer per env worker.  A server renders on card 0 unless
``ServerConfig.gpu`` passes UE's ``-graphicsadapter``.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field

VULKAN_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"

#: 0.10.0 renamed the launcher; 0.9.x is what this project targets.
LAUNCHER_0916 = "CarlaUE4.sh"
LAUNCHER_0100 = "CarlaUnreal.sh"


def launcher_for(server_dir: str) -> str:
    """Pick the launcher name from the directory."""
    return LAUNCHER_0100 if "0100" in server_dir else LAUNCHER_0916


def run(cmd, timeout: int = 30) -> str:
    """Shell out for diagnostics, never raise."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics must not kill the run
        return f"<{exc.__class__.__name__}: {exc}>"


#: One server holds three consecutive ports, so adjacent base_ports would collide;
#: 10 rather than 3 because the neighbour count is CARLA's business.
PORT_STRIDE = 10


@dataclass
class ServerConfig:
    server_dir: str
    #: Worker i uses base_port + PORT_STRIDE * i.
    base_port: int = 2100
    #: Requested render card, reached through UE's -graphicsadapter, since
    #: DRI_PRIME reaches only 0 and 1.
    gpu: int | None = None

    #: RPC-port index; None = the process-local worker index, so under RLinf pass a unique one.
    port_index: int | None = None

    quality_level: str = "Epic"

    cache_dir: str = field(default_factory=lambda: os.path.expanduser("~/carla_cache"))
    startup_timeout: float = 240.0

    def port_for(self, worker_index: int) -> int:
        index = worker_index if self.port_index is None else self.port_index
        return self.base_port + PORT_STRIDE * index

    def argv(self, port: int, ue_log: str) -> list[str]:
        # stdbuf is load-bearing: stdout is a regular file, so the child would
        # block-buffer and a terminated server takes its log with it.
        cmd = []
        if shutil.which("stdbuf"):
            cmd += ["stdbuf", "-oL", "-eL"]
        cmd += [
            os.path.join(self.server_dir, launcher_for(self.server_dir)),
            "-RenderOffScreen",
            "-vulkan",
            "-nosound",
            f"-quality-level={self.quality_level}",
            f"-carla-rpc-port={port}",
            "-log",
            "-stdout",
            "-FullStdOutLogOutput",
            f"-abslog={ue_log}",
        ]
        if self.gpu is not None:
            # Here rather than in start(): UE reads its own argv for this, and it
            # is the only selector that reaches cards above 1.
            cmd.append(f"-graphicsadapter={self.gpu}")
        return cmd


class CarlaServer:
    """Owns one CARLA server process.  Idempotent start/stop.

    Use as a context manager so a crash between start and stop cannot leak a
    CarlaUE4 holding the GPU against the next job.
    """

    def __init__(self, cfg: ServerConfig, worker_index: int, logdir: str) -> None:
        self.cfg = cfg
        self.port = cfg.port_for(worker_index)
        self.logdir = os.path.abspath(logdir)
        self.proc: subprocess.Popen | None = None
        # Port in the name: worker_index is process-local, so four env processes
        # would otherwise interleave one carla-ue-0.log.
        self.ue_log = os.path.join(
            self.logdir, f"carla-ue-{worker_index}-{self.port}.log"
        )
        self.console_log = os.path.join(
            self.logdir, f"carla-server-{worker_index}-{self.port}.log"
        )
        self._fh = None

    # -- lifecycle ---------------------------------------------------------

    def _foreign_listener(self) -> str | None:
        """A pid to name if something is ALREADY on this port, else None.

        start_new_session=True puts servers out of reach of `ray stop`, and ports
        are deterministic, so one orphaned earlier still holds this port and
        wait_ready() would report it ready.
        """
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                pass
        except OSError:
            return None  # nothing listening: the port is ours
        # Best-effort owner via /proc, so no lsof; a miss still reports the leak.
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                with open(f"/proc/{entry}/cmdline", "rb") as f:
                    cmdline = f.read().decode(errors="replace")
                if f"carla-rpc-port={self.port}" in cmdline:
                    return entry
        except OSError:
            pass
        return "unknown pid"

    def start(self) -> None:
        if self.proc is not None:
            return
        leak = self._foreign_listener()
        if leak is not None:
            raise RuntimeError(
                f"port {self.port} is already held (pid {leak}), so this server "
                f"could not start. That is a leaked CARLA server from an earlier "
                f"phase or job -- `ray stop` does not reach them. Kill it first, "
                f"e.g. `pkill -9 -f CarlaUE4`."
            )
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.cfg.cache_dir, exist_ok=True)

        env = os.environ.copy()
        if os.path.exists(VULKAN_ICD):
            env["VK_ICD_FILENAMES"] = VULKAN_ICD
        env["CARLA_CACHE_DIR"] = self.cfg.cache_dir
        if self.cfg.gpu is not None:
            # The card is chosen by argv's -graphicsadapter; this only scopes the
            # CUDA runtime and does not move the renderer.
            env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu)

        cmd = self.cfg.argv(self.port, self.ue_log)
        self._fh = open(self.console_log, "w")
        self._fh.write(f"# {' '.join(cmd)}\n")
        self._fh.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}\n")
        self._fh.flush()
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.cfg.server_dir,
            env=env,
            stdout=self._fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # so stop() can signal the whole tree
        )

    def wait_ready(self, timeout: float | None = None) -> tuple[bool, str]:
        """Poll the RPC port, but give up immediately if the server died."""
        if self.proc is None:
            return False, "not started"
        timeout = timeout if timeout is not None else self.cfg.startup_timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False, f"server exited early with rc={self.proc.returncode}"
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=2):
                    return True, "port open"
            except OSError:
                time.sleep(2.0)
        return False, f"port {self.port} never opened within {timeout}s"

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            # Signal the group: the launcher spawns UE as a child, so killing only
            # the shell leaves UE holding the GPU.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        # Scoped to this port so workers do not kill each other's servers.
        run(["pkill", "-f", f"carla-rpc-port={self.port}"], timeout=15)

    def __enter__(self) -> CarlaServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- diagnostics -------------------------------------------------------

    def tail(self, path: str | None = None, n: int = 40) -> str:
        path = path or self.console_log
        try:
            with open(path, errors="replace") as f:
                return "\n".join(f.read().splitlines()[-n:])
        except Exception as exc:  # noqa: BLE001
            return f"<cannot read {path}: {exc}>"

    def resolved_ue_log(self) -> str | None:
        """Where the UE log actually is, or None -- -abslog is not honoured."""
        if os.path.exists(self.ue_log):
            return self.ue_log
        default = os.path.join(self.cfg.server_dir, "CarlaUE4", "Saved", "Logs")
        try:
            cands = sorted(
                (
                    os.path.join(default, f)
                    for f in os.listdir(default)
                    if f.endswith(".log")
                ),
                key=os.path.getmtime,
            )
        except OSError:
            return None
        return cands[-1] if cands else None

    def failure_report(self) -> str:
        """Both logs: which one carries the error depends on how far startup got."""
        ue = self.resolved_ue_log()
        if ue is None:
            ue_part = (
                f"--- ue log: not at {self.ue_log}, and none under "
                f"{self.cfg.server_dir}/CarlaUE4/Saved/Logs ---\n"
                f"UE opened no log file; the console log above is the only "
                f"evidence there is."
            )
        else:
            ue_part = f"--- ue log ({ue}) ---\n{self.tail(ue)}"
        return (
            f"--- console log ({self.console_log}) ---\n"
            f"{self.tail(self.console_log)}\n{ue_part}"
        )
