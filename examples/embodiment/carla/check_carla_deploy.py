#!/usr/bin/env python3
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

"""Does DRI_PRIME index the job's visible GPUs, or the node's physical ones?

Part A asks for a mapping where the two readings disagree; part B brings up one
server per card. Memory is read per PCI bus id: nvidia-smi's index is cgrouped.

  python3 check_carla_deploy.py --server-dir DIR --part a
  python3 check_carla_deploy.py --server-dir DIR --part b --num-gpus 8
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time

from rlinf.envs.sim.carla.server import CarlaServer, ServerConfig

#: A gain above this is a loaded server; below it, another process or noise.
FOOTPRINT_MIB = 2500

#: Seconds to let the renderer allocate after the RPC port opens. The socket
#: binds well before Vulkan has finished setting up.
SETTLE_S = 18


def nvidia_smi_memory() -> dict[str, int]:
    """memory.used in MiB, keyed by PCI bus id."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    used: dict[str, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                used[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return used


def nvidia_smi_bus_order() -> list[str]:
    """PCI bus ids in nvidia-smi's index order, for the index->card mapping."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            rows.append((int(parts[0]), parts[1]))
    return [bus for _, bus in sorted(rows)]


def label(bus: str, order: list[str]) -> str:
    """`pci bus id` plus its position in nvidia-smi's ordering, when known."""
    return f"{bus} (index {order.index(bus)})" if bus in order else bus


def wait_port(port: int, proc: subprocess.Popen, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(3.0)
    return False


def start_raw(
    cfg: ServerConfig, env_extra: dict[str, str], port: int, logdir: str, name: str
) -> tuple[subprocess.Popen, str]:
    """Start a server with an env built by hand, not by CarlaServer.

    Part A needs CUDA_VISIBLE_DEVICES and DRI_PRIME to *disagree*, which
    CarlaServer is built never to do -- it sets both from one field. Only its
    argv is reused, so the launch flags stay identical to production.
    """
    env = os.environ.copy()
    env.update(env_extra)
    env["CARLA_CACHE_DIR"] = cfg.cache_dir
    path = os.path.join(logdir, f"{name}.log")
    fh = open(path, "w")
    fh.write(f"# env {env_extra}\n")
    fh.flush()
    proc = subprocess.Popen(
        cfg.argv(port, os.path.join(logdir, f"{name}-ue.log")),
        cwd=cfg.server_dir,
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, path


def kill(proc: subprocess.Popen, port: int) -> None:
    import signal

    if proc.poll() is None:
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
    subprocess.run(
        ["pkill", "-f", f"carla-rpc-port={port}"], capture_output=True, timeout=15
    )
    time.sleep(3)


# -- part A ------------------------------------------------------------------


def part_a(cfg: ServerConfig, logdir: str, base_port: int) -> str:
    """Start servers under a CVD mapping where the two readings disagree.

    Returns 'physical' or 'visible' -- which reading the observations support,
    or 'unknown'. It does not guess: if the memory does not land where either
    reading predicts, that is reported as unknown rather than rounded to the
    nearer one.
    """
    order = nvidia_smi_bus_order()
    print(f"nvidia-smi index order (physical anchors): {order}")

    remap = remap_for(len(order))
    if remap is None:
        print(
            f"\nVERDICT_A: UNKNOWN -- only {len(order)} card(s) visible. "
            f"Settling this needs a remap where visible 0 is not physical 0, "
            f"and that requires at least 3 cards. Job 14302 had exactly 2 "
            f"and was ambiguous for exactly this reason."
        )
        return "unknown"
    remap_idx = [int(t) for t in remap.split(",")]
    print(
        f"\n=== A: CUDA_VISIBLE_DEVICES={remap} -> visible 0 is physical "
        f"{remap_idx[0]}, visible 1 is physical {remap_idx[1]} ==="
    )
    print(
        f"    visible reading predicts DRI_PRIME=n -> physical "
        f"{remap_idx[0]} / {remap_idx[1]};"
    )
    print("    physical reading predicts DRI_PRIME=n -> physical 0 / 1\n")

    verdicts = []
    for dri_prime, port in (("0", base_port), ("1", base_port + 100)):
        before = nvidia_smi_memory()
        print(f"--- A: DRI_PRIME={dri_prime} ---")
        proc, path = start_raw(
            cfg,
            {"CUDA_VISIBLE_DEVICES": remap, "DRI_PRIME": dri_prime},
            port,
            logdir,
            f"A_driprime{dri_prime}",
        )
        if not wait_port(port, proc):
            print(f"  server did not come up (rc={proc.returncode})")
            print(f"  log: {path}")
            kill(proc, port)
            verdicts.append("unknown")
            continue
        time.sleep(SETTLE_S)
        after = nvidia_smi_memory()
        gained = sorted(
            (b for b in after if after[b] - before.get(b, 0) >= FOOTPRINT_MIB),
            key=lambda b: order.index(b) if b in order else 99,
        )
        for b in gained:
            print(f"  +{after[b] - before.get(b, 0)} MiB on {label(b, order)}")

        got_index = order.index(gained[0]) if gained and gained[0] in order else None
        want_visible = remap_idx[int(dri_prime)]
        if got_index == want_visible:
            verdicts.append("visible")
        elif got_index == int(dri_prime):
            verdicts.append("physical")
        else:
            print(
                f"  landed on index {got_index}, which neither reading for "
                f"DRI_PRIME={dri_prime} predicts (visible {want_visible}, "
                f"physical {dri_prime}) -- recording as unknown rather than "
                f"rounding"
            )
            verdicts.append("unknown")
        kill(proc, port)
        time.sleep(4)

    print(f"\nVERDICT_A dri_prime=0 -> {verdicts[0]}, dri_prime=1 -> {verdicts[1]}")
    if verdicts[0] == verdicts[1] == "visible":
        print(
            "VERDICT_A: VISIBLE-indexed. cfg.gpu = i is correct as written for "
            "any allocation."
        )
        return "visible"
    if verdicts[0] == verdicts[1] == "physical":
        print(
            "VERDICT_A: PHYSICAL-indexed. cfg.gpu = i is WRONG unless Slurm "
            "handed out exactly 0..N-1; the pin must be translated through "
            "CUDA_VISIBLE_DEVICES."
        )
        return "physical"
    print(
        f"VERDICT_A: UNKNOWN -- the two probes disagree {verdicts}. The pin is "
        f"not understood well enough to deploy."
    )
    return "unknown"


def cvd_list() -> list[str]:
    return [t for t in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if t]


def remap_for(num_visible: int) -> str | None:
    """A CVD mapping whose visible 0 is not physical 0, or None if impossible.

    Two cards cannot produce one: {0,1} is the only subset of size two, and
    both readings agree on it.
    """
    if num_visible < 3:
        return None
    if num_visible >= 8:
        return "5,7"
    return f"{num_visible - 2},{num_visible - 1}"


def expected_card(
    indexing: str, i: int, visible: list[str], order: list[str]
) -> str | None:
    """Which PCI bus id server ``i`` is predicted to land on, or None.

    Pure so the prediction can be tested without a GPU -- it is the one piece
    of part B with a right answer, and the two readings differ on it.
    """
    if indexing == "visible":
        if i >= len(visible):
            return None
        tok = visible[i]
        # Slurm may express the allocation as indices or as GPU UUIDs; only indices resolve.
        if not tok.isdigit() or int(tok) >= len(order):
            return None
        return order[int(tok)]
    if indexing == "physical":
        return order[i] if i < len(order) else None
    return None


# -- part B ------------------------------------------------------------------


def part_b(
    cfg: ServerConfig, logdir: str, num_gpus: int, indexing: str, base_port: int
) -> int:
    """Bring up one server per card, the layout phase3 needs.

    Sequential on purpose: if the pin is broken and every server lands on one
    card, starting all eight at once dies of VRAM exhaustion instead of
    reporting the misplacement. One at a time, the second server already shows
    it and the run stops there with a usable diagnosis.
    """
    order = nvidia_smi_bus_order()
    visible = cvd_list()
    print(f"Slurm CUDA_VISIBLE_DEVICES={','.join(visible) or '<unset>'}")
    print(f"nvidia-smi index order: {order}")
    if indexing == "visible":
        print(
            "indexing: VISIBLE -- server i is expected on the card at "
            "position i of the allocation"
        )
    elif indexing == "physical":
        print(
            "indexing: PHYSICAL -- server i is expected on physical card i, "
            "which is only right if the allocation is 0..N-1"
        )
    else:
        print("indexing: UNKNOWN -- expectations below may be wrong; run part A")

    def expect(i: int) -> str | None:
        return expected_card(indexing, i, visible, order)

    servers: list[CarlaServer] = []
    for i in range(num_gpus):
        srv = CarlaServer(
            ServerConfig(
                server_dir=cfg.server_dir,
                cache_dir=cfg.cache_dir,
                base_port=base_port,
                gpu=i,
            ),
            worker_index=i,
            logdir=logdir,
        )
        print(
            f"\n--- B: server {i} -> port {srv.port} gpu={i} "
            f"(expect {expect(i) or '<unknown>'}) ---"
        )
        before = nvidia_smi_memory()
        srv.start()
        # wait_ready returns (ok, why), and a 2-tuple is always truthy -- so
        # this must unpack rather than test the result directly.
        ok, why = srv.wait_ready(timeout=200)
        if not ok:
            print(f"  server {i} never became ready: {why}")
            print(srv.failure_report())
            for s in servers + [srv]:
                s.stop()
            return 1
        time.sleep(SETTLE_S)
        after = nvidia_smi_memory()
        gained = sorted(
            (b for b in after if after[b] - before.get(b, 0) >= FOOTPRINT_MIB),
            key=lambda b: order.index(b) if b in order else 99,
        )
        if not gained:
            print(
                f"  no card gained a footprint -- server {i} is running but "
                f"not rendering"
            )
            for s in servers + [srv]:
                s.stop()
            return 1
        landed = ", ".join(
            f"{label(b, order)} +{after[b] - before.get(b, 0)} MiB" for b in gained
        )
        print(f"  landed on: {landed}")
        servers.append(srv)

        exp = expect(i)
        if exp is not None and gained[0] != exp:
            print(
                f"  MISPLACED: expected {label(exp, order)}, "
                f"got {label(gained[0], order)}"
            )
            print(
                f"  Stopping here rather than starting the remaining "
                f"{num_gpus - i - 1} servers: if this repeats, they all "
                f"stack on one card and the job dies of VRAM exhaustion "
                f"instead of reporting."
            )
            for s in servers:
                s.stop()
            return 1

    print(f"\n=== all {num_gpus} servers up; distribution ===")
    used = nvidia_smi_memory()
    for b in order:
        flag = "  <-- server" if used.get(b, 0) >= FOOTPRINT_MIB else ""
        print(f"  {label(b, order)}: {used.get(b, 0)} MiB{flag}")

    total_free = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        .stdout.strip()
        .splitlines()
    )
    for line in total_free:
        total, used_mib = (int(x.strip()) for x in line.split(","))
        print(f"  card: {used_mib}/{total} MiB used ({total - used_mib} MiB free)")

    print(
        f"\nVERDICT_B: {num_gpus} servers, one per card, "
        f"{len([b for b in order if used.get(b, 0) >= FOOTPRINT_MIB])} cards "
        f"occupied"
    )

    for s in servers:
        s.stop()
    print("VERDICT_B: LAYOUT_OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/carla_cache"))
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--part", choices=["a", "b", "ab"], default="ab")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument(
        "--base-port",
        type=int,
        default=2100,
        help="part A uses base and base+100; part B uses "
        "base+200 and up, spaced by PORT_STRIDE",
    )
    ap.add_argument(
        "--indexing",
        choices=["visible", "physical", "unknown"],
        default=None,
        help="skip re-running part A by passing its verdict",
    )
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    cfg = ServerConfig(server_dir=args.server_dir, cache_dir=args.cache_dir)

    print(f"node={socket.gethostname()} at={time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    print(f"logs: {args.logdir}\n")

    indexing = args.indexing or "unknown"
    rc = 0
    if args.part in ("a", "ab"):
        indexing = part_a(cfg, args.logdir, args.base_port)
        if indexing == "unknown":
            print(
                "\nPart B not run: without knowing the indexing, a "
                "misplacement cannot be told from a correct result."
            )
            return 1
    if args.part in ("b", "ab"):
        if indexing == "unknown":
            print("Part B needs --indexing or a part-A run; refusing to guess.")
            return 1
        print()
        rc = part_b(cfg, args.logdir, args.num_gpus, indexing, args.base_port + 200)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
