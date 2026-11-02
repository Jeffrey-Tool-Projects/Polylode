"""Keep asset importing inside a memory budget.

Decompressed Unity bundles are large and easy to accumulate: a careless import
sweep can take the whole machine down.  Every cache in :mod:`origin` is
bounded by count, but counts are a poor proxy for bytes -- one hair bundle can
outweigh twenty clothing ones.  This measures the process's real resident set
and drops caches when it climbs past the budget.
"""
from __future__ import annotations

import gc
import os

from ..core import swallowed

# The budget tracks what is *free right now*, not a fixed slice of installed
# RAM: on a machine with plenty spare there is no reason to be timid, and on a
# loaded one a generous fraction of total would be exactly wrong.
_MIN_BUDGET_MB = 384.0
_TOTAL_CEILING = 0.60           # never claim more than this much of the machine
# What the rest of the system keeps.  This is the number that actually protects
# the machine: the danger is not our own size but pushing everything else into
# swap, which is what a hard lock-up looks like.
_RESERVE_FRACTION = 0.08
_MIN_RESERVE_MB = 1024.0


def _win_rss() -> float | None:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    # HANDLE is pointer-sized; leaving ctypes to guess truncates it to 32 bits
    # on x64 and the call quietly fails, which is how this returned 0 before.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []

    for dll_name, fn_name in (("kernel32", "K32GetProcessMemoryInfo"),
                              ("psapi", "GetProcessMemoryInfo")):
        try:
            dll = kernel32 if dll_name == "kernel32" else ctypes.WinDLL("psapi")
            fn = getattr(dll, fn_name)
        except (OSError, AttributeError):
            continue
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        c = PMC()
        c.cb = ctypes.sizeof(c)
        if fn(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.WorkingSetSize / (1024 * 1024)
    return None


def rss_mb() -> float:
    """Resident set size in MB, or 0.0 if it genuinely can't be measured."""
    if os.name == "nt":
        try:
            v = _win_rss()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.memguard.rss_mb: _win_rss", exc)
            v = None
        if v is not None:
            return v
    try:                                             # psutil if it happens to be here
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.memguard.rss_mb: psutil", exc)
    try:                                             # POSIX
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024.0
    except ImportError:                              # no ``resource`` on Windows
        return 0.0


def _win_memory_status() -> tuple[float, float] | None:
    """(total MB, available MB) via GlobalMemoryStatusEx."""
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    mb = 1024.0 * 1024.0
    return st.ullTotalPhys / mb, st.ullAvailPhys / mb


def _posix_memory_status() -> tuple[float, float] | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            vals = {}
            for line in fh:
                k, _, rest = line.partition(":")
                vals[k] = float(rest.strip().split()[0]) / 1024.0
        total = vals.get("MemTotal")
        avail = vals.get("MemAvailable", vals.get("MemFree"))
        if total:
            return total, avail if avail is not None else total * 0.5
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.memguard._posix_memory_status: open", exc)
    try:
        total = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                 / (1024.0 * 1024.0))
        return total, total * 0.5
    except (ValueError, OSError, AttributeError):
        return None


def memory_status() -> tuple[float, float]:
    """(total MB, available MB) for this machine; (0, 0) if unknown."""
    try:
        st = _win_memory_status() if os.name == "nt" else _posix_memory_status()
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.memguard.memory_status: st = _win_memory_status() if os.name == 'nt' els", exc)
        st = None
    if st is None:
        try:
            import psutil
            vm = psutil.virtual_memory()
            st = (vm.total / (1024.0 * 1024.0), vm.available / (1024.0 * 1024.0))
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.memguard.memory_status: import psutil", exc)
            st = (0.0, 0.0)
    return st


def total_ram_mb() -> float:
    return memory_status()[0]


def available_ram_mb() -> float:
    return memory_status()[1]


def budget_mb() -> float:
    """How much resident memory the importer may use on *this* machine."""
    env = os.environ.get("ISLANDLIFE_ASSET_BUDGET_MB")
    if env:
        try:
            return max(_MIN_BUDGET_MB, float(env))
        except ValueError:
            pass
    total, avail = memory_status()
    if total <= 0.0:                                 # couldn't measure; be timid
        return _MIN_BUDGET_MB * 2
    # everything currently free, minus the reserve, plus what we already hold
    spendable = rss_mb() + avail - reserve_mb()
    return max(_MIN_BUDGET_MB, min(total * _TOTAL_CEILING, spendable))


def reserve_mb() -> float:
    """Free physical memory the rest of the system must keep."""
    total = total_ram_mb()
    if total <= 0.0:
        return _MIN_RESERVE_MB
    return max(_MIN_RESERVE_MB, total * _RESERVE_FRACTION)


def over_budget(margin: float = 0.0) -> bool:
    """True if allocating ``margin`` MB more would be unsafe.

    Two independent tests: our own footprint against the per-machine budget,
    and the machine's remaining free memory against its reserve.  The second
    catches the case that actually crashes a desktop -- we are within budget but
    everything else running has already used the RAM up.
    """
    total, avail = memory_status()
    if avail > 0.0 and avail - margin < reserve_mb():
        return True
    cur = rss_mb()
    return cur > 0.0 and cur + margin > budget_mb()


def relieve(hard: bool = False) -> float:
    """Drop asset caches.  ``hard`` also releases the open bundle handles."""
    from . import meshimport, prefab
    prefab.clear_loaders()
    if hard:
        meshimport.clear_cache()
    else:
        meshimport.trim_cache()
    gc.collect()
    return rss_mb()


def guard(margin: float = 0.0) -> bool:
    """Free memory if we're near the budget.  True if it had to act."""
    if not over_budget(margin):
        return False
    relieve(hard=True)
    return True


def max_bundle_bytes() -> int:
    """Largest single bundle we'll decompress, scaled to the machine."""
    return int(min(512.0, max(96.0, budget_mb() * 0.35)) * 1024 * 1024)


def describe() -> str:
    total, avail = memory_status()
    if total <= 0.0:
        return f"記憶體上限 {budget_mb():.0f} MB（無法偵測實體記憶體）"
    return (f"記憶體：實體 {total/1024:.1f} GB，可用 {avail/1024:.1f} GB；"
            f"素材上限 {budget_mb():.0f} MB，保留 {reserve_mb():.0f} MB")
