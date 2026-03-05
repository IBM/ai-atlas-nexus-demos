import queue
from typing import Any, Dict, Optional

from gaf_guard.clients.stream_adaptors.base import StreamAdapter
from gaf_guard.clients.stream_adaptors.json_adapter import JSONAdapter


def get_adapter(adapter_type: str, config: Dict[str, Any]) -> Optional[StreamAdapter]:
    """Factory function to get the appropriate adapter."""
    adapters = {"JSON": JSONAdapter}

    adapter_class = adapters.get(adapter_type)
    if adapter_class:
        adapter = adapter_class(config)
        adapter.connect()
        return adapter
    return None


# def stream_worker(adapter: StreamAdapter, message_queue: queue.Queue):
#     """Background thread worker to consume streaming data."""
#     try:
#         for message in adapter.consume():
#             try:
#                 message_queue.put(message, timeout=1)
#             except queue.Full:
#                 message_queue.get()
#                 message_queue.put(message)

#     except Exception as e:
#         raise (f"Stream error: {e}")
#     finally:
#         if adapter.is_connected:
#             adapter.disconnect()
