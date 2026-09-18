"""Minimal ctypes wrapper around libpapi's low-level API for real hardware performance counters.

Ported from the sibling experiments/ project's PAPI harness (same machine, same measurement
technique): named-event access to uncore memory-controller events (bdx_unc_imc*::UNC_M_CAS_COUNT)
running concurrently with a core EventSet is needed for true DRAM-level operational intensity,
which PAPI's simplified high-level API (papi_high) doesn't support -- so this binds directly
against libpapi.so instead of using a prebuilt Python package.
"""

import ctypes
import ctypes.util
from dataclasses import dataclass, field

_PAPI_VER_CURRENT = (7 << 24) | (1 << 16)  # PAPI_VERSION_NUMBER(7,1,0,0) & 0xffff0000
_PAPI_OK = 0
_PAPI_NULL = -1

_lib_path = ctypes.util.find_library("papi") or "/usr/local/lib/libpapi.so"
_papi = ctypes.CDLL(_lib_path)

_papi.PAPI_library_init.restype = ctypes.c_int
_papi.PAPI_library_init.argtypes = [ctypes.c_int]

_papi.PAPI_create_eventset.restype = ctypes.c_int
_papi.PAPI_create_eventset.argtypes = [ctypes.POINTER(ctypes.c_int)]

_papi.PAPI_add_named_event.restype = ctypes.c_int
_papi.PAPI_add_named_event.argtypes = [ctypes.c_int, ctypes.c_char_p]

_papi.PAPI_start.restype = ctypes.c_int
_papi.PAPI_start.argtypes = [ctypes.c_int]

_papi.PAPI_stop.restype = ctypes.c_int
_papi.PAPI_stop.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]

_papi.PAPI_read.restype = ctypes.c_int
_papi.PAPI_read.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]

_papi.PAPI_cleanup_eventset.restype = ctypes.c_int
_papi.PAPI_cleanup_eventset.argtypes = [ctypes.c_int]

_papi.PAPI_destroy_eventset.restype = ctypes.c_int
_papi.PAPI_destroy_eventset.argtypes = [ctypes.POINTER(ctypes.c_int)]

_papi.PAPI_strerror.restype = ctypes.c_char_p
_papi.PAPI_strerror.argtypes = [ctypes.c_int]

_PAPI_CPU_ATTACH = 27


class _PAPI_cpu_option_t(ctypes.Structure):
    _fields_ = [("eventset", ctypes.c_int), ("cpu_num", ctypes.c_uint)]


_papi.PAPI_set_opt.restype = ctypes.c_int
_papi.PAPI_set_opt.argtypes = [ctypes.c_int, ctypes.c_void_p]

_papi.PAPI_assign_eventset_component.restype = ctypes.c_int
_papi.PAPI_assign_eventset_component.argtypes = [ctypes.c_int, ctypes.c_int]

_papi.PAPI_get_component_index.restype = ctypes.c_int
_papi.PAPI_get_component_index.argtypes = [ctypes.c_char_p]

_initialized = False


def _check(code, what):
    if code < 0:
        msg = _papi.PAPI_strerror(code)
        msg = msg.decode() if msg else "unknown error"
        raise RuntimeError(f"PAPI error in {what}: {code} ({msg})")
    return code


def init_library():
    global _initialized
    if _initialized:
        return
    ret = _papi.PAPI_library_init(_PAPI_VER_CURRENT)
    if ret != _PAPI_VER_CURRENT:
        raise RuntimeError(
            f"PAPI_library_init version mismatch: got {ret:#x}, wanted {_PAPI_VER_CURRENT:#x}"
        )
    _initialized = True


@dataclass
class CounterSet:
    """A PAPI EventSet bound to a fixed list of named events.

    Not thread-safe / not reentrant: one CounterSet is one live EventSet.
    Two CounterSets (e.g. core + uncore) can be started/stopped concurrently
    since they map to independent hardware.
    """

    events: list[str]
    cpu: int | None = None  # set for uncore EventSets: they're socket-wide, not per-thread
    _eventset: int = field(default=_PAPI_NULL, init=False, repr=False)

    def __post_init__(self):
        init_library()
        eventset = ctypes.c_int(_PAPI_NULL)
        _check(_papi.PAPI_create_eventset(ctypes.byref(eventset)), "PAPI_create_eventset")
        self._eventset = eventset.value
        if self.cpu is not None:
            cidx = _check(
                _papi.PAPI_get_component_index(b"perf_event_uncore"),
                "PAPI_get_component_index(perf_event_uncore)",
            )
            _check(
                _papi.PAPI_assign_eventset_component(self._eventset, cidx),
                "PAPI_assign_eventset_component",
            )
            opt = _PAPI_cpu_option_t(eventset=self._eventset, cpu_num=self.cpu)
            _check(
                _papi.PAPI_set_opt(_PAPI_CPU_ATTACH, ctypes.cast(ctypes.byref(opt), ctypes.c_void_p)),
                "PAPI_set_opt(PAPI_CPU_ATTACH)",
            )
        for name in self.events:
            ret = _papi.PAPI_add_named_event(self._eventset, name.encode())
            if ret != _PAPI_OK:
                self.close()
                _check(ret, f"PAPI_add_named_event({name})")

    def start(self):
        _check(_papi.PAPI_start(self._eventset), "PAPI_start")

    def stop(self) -> dict[str, int]:
        values = (ctypes.c_longlong * len(self.events))()
        _check(_papi.PAPI_stop(self._eventset, values), "PAPI_stop")
        return dict(zip(self.events, values))

    def close(self):
        if self._eventset == _PAPI_NULL:
            return
        _papi.PAPI_cleanup_eventset(self._eventset)
        eventset = ctypes.c_int(self._eventset)
        _papi.PAPI_destroy_eventset(ctypes.byref(eventset))
        self._eventset = _PAPI_NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def discover_uncore_imc_events(suffix: str = ":ALL") -> list[str]:
    """Find live per-channel UNC_M_CAS_COUNT event names via papi_native_avail.

    Not hardcoded to a fixed channel count so it adapts to whatever the running machine's
    papi_native_avail reports (this Broadwell-EP shows 5 imc channel entries, only some of
    which are populated).
    """
    import re
    import subprocess

    out = subprocess.run(["papi_native_avail"], capture_output=True, text=True, check=True).stdout
    events = []
    for line in out.splitlines():
        m = re.search(r"\|\s*(\S+::UNC_M_CAS_COUNT)\s*\|", line)
        if m:
            events.append(m.group(1) + suffix)
    return events
