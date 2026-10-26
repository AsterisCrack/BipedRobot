"""Write-through cache for the PhysX articulation view's static body properties.

With the GPU pipeline, PhysX reads masses, inertias, COMs and shape materials back from the device on
every ``get_*`` call: measured at 4096 envs, ``get_material_properties`` alone cost ~28 ms per training
iteration, and the four getters together ~60 ms (the reset-time DR terms call them on ~every step,
Isaac Lab's ``body_com_pose_b`` calls ``get_coms`` every step).

These properties are static in PhysX: after startup they change only through the matching ``set_*``
call. So the first ``get_*`` reads PhysX once, every ``set_*(data, indices)`` writes through to PhysX AND
updates the cached rows of ``indices`` (the same rows PhysX applies), and later gets return a fresh
copy of the cache. Callers see the values PhysX would return, without the readback.

Only the view this is installed on is affected; joint (DOF) properties are not cached.
"""
from __future__ import annotations

import torch

PROPERTIES = ("masses", "inertias", "coms", "material_properties")


class PhysxPropertyCache:
    def __init__(self, view, properties=PROPERTIES):
        self._view = view
        self._cache: dict[str, torch.Tensor] = {}
        self.reads = {name: 0 for name in properties}      # PhysX readbacks actually done (for tests/logs)
        self._physx_get = {}
        for name in properties:
            get, put = getattr(view, f"get_{name}"), getattr(view, f"set_{name}")
            self._physx_get[name] = get
            setattr(view, f"get_{name}", self._getter(name, get))
            setattr(view, f"set_{name}", self._setter(name, put))

    def _getter(self, name, get):
        def cached_get():
            data = self._cache.get(name)
            if data is None:
                data = self._cache[name] = get().clone()
                self.reads[name] += 1
            return data.clone()           # callers modify the result in place, as with PhysX's fresh tensor
        return cached_get

    def _setter(self, name, put):
        def write_through(data, indices=None, *args, **kwargs):
            put(data, indices, *args, **kwargs)
            cached = self._cache.get(name)
            if cached is not None:
                src = data.to(device=cached.device, dtype=cached.dtype)
                if indices is None:
                    cached.copy_(src)
                else:
                    rows = torch.as_tensor(indices, device=cached.device).long()
                    # Isaac Lab passes the full (count, ...) buffer; a (len(indices), ...) one is the rows themselves
                    cached[rows] = src[rows] if src.shape[0] == cached.shape[0] else src
        return write_through

    def verify(self) -> dict[str, float]:
        """Max |cached - fresh PhysX readback| per cached property (debug check; does the slow reads)."""
        return {name: float((data - self._physx_get[name]().to(data.device, data.dtype)).abs().max())
                for name, data in self._cache.items()}

    def invalidate(self, name: str | None = None) -> None:
        """Drop the cached copy (all properties if name is None); the next get reads PhysX again."""
        if name is None:
            self._cache.clear()
        else:
            self._cache.pop(name, None)
