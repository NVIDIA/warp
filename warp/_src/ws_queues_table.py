"""
Work-Stealing Queues Table for managing queues indexed by stream combinations.
"""

from typing import Optional


class WorkStealingQueuesTable:
    """
    Table for managing work-stealing queues indexed by stream combinations.

    Memory:
        - Only 3*k counters are allocated per queues (front, back, last)
        - m is specified per epoch via next_epoch(m)

    Usage:
        table = WorkStealingQueuesTable()
        streams = [wp.Stream(device="cuda:0") for _ in range(4)]

        # Get or create queues for this stream combination
        queues = table.get_or_create(streams)

        # Set bounds for this epoch
        queues.next_epoch(m=1024)
        view = queues.view()

        # ... launch kernels ...
    """

    def __init__(self):
        self._table = {}  # Maps hash(stream_ids) -> ws_queues

    @staticmethod
    def _compute_key(streams) -> int:
        """Compute a hash key from stream IDs (order-dependent)."""
        return hash(tuple(id(s) for s in streams))

    def get_or_create(self, streams, enable_instrumentation: bool = False):
        """
        Get or create work-stealing queues for a stream combination.

        Args:
            streams: List of streams (k = len(streams) deques will be created)
            enable_instrumentation: Enable instrumentation for validation

        Returns:
            WorkStealingQueues with k = len(streams) deques

        Note:
            Queues use unified memory - accessible from all GPUs.
            m is NOT set here - call next_epoch(m) to set bounds per epoch.
        """
        import warp as wp

        if not streams:
            raise ValueError("Cannot create work-stealing queues for empty stream list")

        key = self._compute_key(streams)

        # Return existing if already created
        if key in self._table:
            return self._table[key]

        # Create new: only k parameter (device-agnostic via unified memory)
        ws_queues = wp.WorkStealingQueues(k=len(streams), enable_instrumentation=enable_instrumentation)

        self._table[key] = ws_queues
        return ws_queues

    def get(self, streams) -> Optional:
        """Get queues if they exist, otherwise return None."""
        return self._table.get(self._compute_key(streams))

    def clear(self):
        """Clear all cached queues (they will be garbage collected)."""
        self._table.clear()

    def __len__(self):
        """Return number of cached stream combinations."""
        return len(self._table)

    def __repr__(self):
        return f"WorkStealingQueuesTable({len(self._table)} stream combinations)"
