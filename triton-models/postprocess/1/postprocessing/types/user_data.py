"""User data requests."""

import sys

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:
    """Data structure to gather queued requests."""

    def __init__(self):
        self._completed_requests = queue.Queue()
