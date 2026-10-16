"""Record CPU utilization over the Slurm job's allowed CPU set."""
import json
import os
import sys
import time

import psutil

allowed = sorted(os.sched_getaffinity(0))
with open(sys.argv[1], 'a', buffering=1) as stream:
    while True:
        percentages = psutil.cpu_percent(interval=1, percpu=True)
        selected = {str(cpu): percentages[cpu] for cpu in allowed}
        stream.write(json.dumps({'time_epoch': time.time(), 'allowed_cpus': allowed,
                                 'mean_util_pct': sum(selected.values())/len(selected),
                                 'util_pct_by_cpu': selected})+'\n')
