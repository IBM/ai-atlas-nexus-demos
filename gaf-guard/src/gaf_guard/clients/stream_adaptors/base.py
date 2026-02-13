"""
Base classes and interfaces for streaming data adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, Optional


@dataclass
class StreamMessage:
    """Standardized message format for all streaming sources."""

    timestamp: datetime
    prompt_index: Optional[str]
    prompt: Any
    metadata: Optional[Dict[str, Any]] = None


class StreamAdapter(ABC):
    """Abstract base class for streaming data adapters."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with configuration.

        Args:
            config: Configuration dictionary specific to the streaming source
        """
        self.config = config
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the streaming source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the streaming source."""
        pass

    @abstractmethod
    def next(self) -> Iterator[StreamMessage]:
        """
        Consume messages from the streaming source.

        Yields:
            StreamMessage objects
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._connected
