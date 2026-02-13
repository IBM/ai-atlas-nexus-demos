"""
Redis streaming adapter implementation.
"""

import json
from datetime import datetime
from io import StringIO
from typing import Any, Dict, Iterator

from gaf_guard.clients.adaptors.base import StreamAdapter, StreamMessage


class JSONAdapter(StreamAdapter):
    """Adapter for Redis Streams."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize JSON adapter.
        """
        super().__init__(config)

    def connect(self) -> None:
        """Connect to Redis server."""
        try:
            stringio = StringIO(self.config["byte_data"].decode("utf-8"))
            self.input_file = json.loads(stringio.read())
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to load JSON file: {e}")

    def disconnect(self) -> None:
        """Disconnect from JSON."""
        self._connected = False

    def next(self) -> Iterator[StreamMessage]:
        """Return next message from JSON Stream."""
        if not hasattr(self, "message_gen"):
            self.message_gen = self.create_message_gen()
        try:
            return next(self.message_gen)
        except StopIteration:
            return None

    def create_message_gen(self):
        if not self._connected:
            raise RuntimeError("Not connected to JSON file. Call connect() first.")
        try:
            for index, message in enumerate(self.input_file, start=1):
                yield StreamMessage(
                    timestamp=datetime.now(),
                    prompt_index=str(index),
                    prompt=message,
                ).__dict__

        except Exception as e:
            print(f"Error consuming message from JSON file: {e}")
