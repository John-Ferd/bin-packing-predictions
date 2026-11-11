"""
Real online bin packing instances from Microsoft's Azure VM packing trace.

Dataset: "Azure Trace for Packing 2020" (zone A, 14 days of VM requests),
Microsoft, CC-BY license. Paper: Hadary et al., "Protean: VM Allocation Service
at Scale", OSDI 2020. https://github.com/Azure/AzurePublicDataset

Each VM request has a VM type, and each (VM type, machine type) pair gives the
fraction of the machine's CPU and memory the VM needs. For one machine type,
packing the arriving VMs onto as few machines as possible is online bin
packing. We reduce the two resources to one size, the dominant resource
max(cpu, memory), and round it UP to a whole percent of a machine, so sizes
are integers in 1..100 with bin capacity 100. Rounding up means any packing
of the rounded sizes is also feasible for the real VMs.

Simplification: VMs are packed in arrival order and never leave (classic
online bin packing has no departures).
"""
from __future__ import annotations

import sqlite3
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/dataset-packing-2020/"
       "azurevmallocation_dataset2020_AzurePackingTraceV1.zip")
SQLITE_NAME = "packing_trace_zone_a_v1.sqlite"


def _sqlite_path(data_dir: Path) -> Path:
    db = data_dir / SQLITE_NAME
    if db.exists():
        return db
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "AzurePackingTraceV1.zip"
    if not zpath.exists():
        print(f"downloading the Azure packing trace (53 MB) to {zpath} ...")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        member = next(n for n in z.namelist() if n.endswith(SQLITE_NAME))
        with z.open(member) as src, open(db, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    return db


def vm_sizes(core, memory, k: int = 100) -> np.ndarray:
    """Dominant-resource size rounded up to an integer in 1..k."""
    dom = np.maximum(np.asarray(core, dtype=float), np.asarray(memory, dtype=float))
    return np.clip(np.ceil(dom * k - 1e-9), 1, k).astype(np.int64)


def load_azure_items(machine_id: int, data_dir: str | Path = "data", k: int = 100):
    """VM requests that arrived during the 14-day window, in arrival order, that
    can run on ``machine_id``. Returns (sizes in 1..k, arrival time in days)."""
    data_dir = Path(data_dir)
    cache = data_dir / f"azure_machine{machine_id}_k{k}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["sizes"].astype(np.int64), z["day"]
    db = _sqlite_path(data_dir)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        """SELECT v.starttime, t.core, t.memory FROM vm v JOIN vmType t ON v.vmTypeId = t.vmTypeId
           WHERE t.machineId = ? AND v.starttime >= 0 AND v.starttime < 14
           ORDER BY v.starttime, v.vmId""", (machine_id,)).fetchall()
    con.close()
    if not rows:
        raise ValueError(f"no requests for machine type {machine_id}")
    a = np.asarray(rows, dtype=float)
    sizes, day = vm_sizes(a[:, 1], a[:, 2], k), a[:, 0].astype(np.float32)
    np.savez_compressed(cache, sizes=sizes.astype(np.uint8), day=day)
    return sizes, day
