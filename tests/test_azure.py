import sqlite3

import numpy as np

from binpacking.azure import SQLITE_NAME, load_azure_items, vm_sizes


def test_vm_size_is_dominant_resource_rounded_up():
    s = vm_sizes([0.05, 0.021, 0.5, 0.0001], [0.01, 0.03, 0.2, 0.0])
    np.testing.assert_array_equal(s, [5, 3, 50, 1])      # 0.021 cpu < 0.03 mem -> 3; tiny -> at least 1


def _fake_trace(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE vm (vmId INTEGER PRIMARY KEY, tenantId INTEGER, vmTypeId INTEGER, "
                "priority INTEGER, starttime REAL, endtime REAL)")
    con.execute("CREATE TABLE vmType (id INTEGER PRIMARY KEY, vmTypeId INTEGER, machineId INTEGER, "
                "core DOUBLE, memory DOUBLE, hdd DOUBLE, ssd DOUBLE, nic DOUBLE)")
    con.executemany("INSERT INTO vmType VALUES (?,?,?,?,?,0,0,0)",
                    [(1, 10, 7, 0.25, 0.1), (2, 11, 7, 0.05, 0.08), (3, 10, 8, 0.5, 0.5)])
    con.executemany("INSERT INTO vm VALUES (?,0,?,0,?,NULL)",
                    [(1, 10, 2.5), (2, 11, 0.5), (3, 10, -1.0), (4, 11, 13.9), (5, 12, 1.0), (6, 10, 14.0)])
    con.commit()
    con.close()


def test_loader_keeps_window_and_machine_in_arrival_order(tmp_path):
    _fake_trace(tmp_path / SQLITE_NAME)
    sizes, day = load_azure_items(7, tmp_path)
    # vm 3 started before the window, vm 6 at its end, vm 5's type has no machine-7 entry
    np.testing.assert_array_equal(sizes, [8, 25, 8])
    np.testing.assert_allclose(day, [0.5, 2.5, 13.9])
    again = load_azure_items(7, tmp_path)                 # second call reads the cache
    np.testing.assert_array_equal(again[0], sizes)
