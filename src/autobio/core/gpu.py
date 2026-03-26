"""GPU discovery and semaphore-based allocation."""

from __future__ import annotations

import threading

from autobio.core.result import GPUNotAvailableError


class GPUManager:
    """Lightweight GPU allocator backed by pynvml.

    Falls back gracefully to an empty GPU list when the NVIDIA driver or
    ``pynvml`` is unavailable.
    """

    def __init__(self) -> None:
        self._available: list[int] = self._discover_gpus()
        self._lock = threading.Lock()
        self._in_use: set[int] = set()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_gpus() -> list[int]:
        """Return a sorted list of GPU device indices, or ``[]`` on failure."""
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return list(range(count))
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # Allocation / release
    # ------------------------------------------------------------------

    def allocate(
        self,
        count: int = 1,
        device_ids: list[int] | None = None,
    ) -> list[int]:
        """Reserve GPUs.

        Args:
            count: Number of GPUs to allocate (used when *device_ids* is
                ``None``).
            device_ids: Specific GPU indices to request.

        Returns:
            List of allocated device indices.

        Raises:
            GPUNotAvailableError: If the requested GPUs cannot be satisfied.
        """
        with self._lock:
            if device_ids is not None:
                unavailable = [
                    d for d in device_ids if d not in self._available or d in self._in_use
                ]
                if unavailable:
                    free = [d for d in self._available if d not in self._in_use]
                    raise GPUNotAvailableError(
                        f"GPU(s) {unavailable} not available. Available: {free}"
                    )
                self._in_use.update(device_ids)
                return list(device_ids)

            free = [d for d in self._available if d not in self._in_use]
            if len(free) < count:
                raise GPUNotAvailableError(
                    f"Requested {count} GPU(s) but only {len(free)} free. Available: {free}"
                )
            allocated = free[:count]
            self._in_use.update(allocated)
            return allocated

    def release(self, device_ids: list[int]) -> None:
        """Return GPUs to the available pool."""
        with self._lock:
            self._in_use -= set(device_ids)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available_gpus(self) -> list[int]:
        """Currently unallocated GPU indices."""
        with self._lock:
            return [d for d in self._available if d not in self._in_use]

    @property
    def has_gpus(self) -> bool:
        """Whether any GPUs were discovered."""
        return len(self._available) > 0
