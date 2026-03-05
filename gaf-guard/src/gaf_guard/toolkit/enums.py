from enum import Enum, StrEnum, auto


class CustomStrEnum(StrEnum):
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()


class MessageType(CustomStrEnum):
    # Client message types
    CLIENT_INPUT = auto()
    CLIENT_RESPONSE = auto()

    # Server message types
    GAF_GUARD_WF_STARTED = auto()
    GAF_GUARD_WF_COMPLETED = auto()
    GAF_GUARD_STEP_STARTED = auto()
    GAF_GUARD_STEP_DATA = auto()
    GAF_GUARD_STEP_COMPLETED = auto()
    GAF_GUARD_QUERY = auto()


class Role(StrEnum):
    USER = "user"
    AGENT = "assistant"
    SYSTEM = "system"


class Serializer(Enum):
    YAML = auto()
    JSON = auto()


class UserInputType(CustomStrEnum):

    # User can only send below message types
    USER_INTENT = auto()
    INITIAL_RISKS = auto()
    INPUT_PROMPT = auto()


class StreamStatus(CustomStrEnum):

    # User can only send below message types
    ACTIVE = auto()
    PAUSED = auto()
    STOPPED = auto()


class ServerStatus(CustomStrEnum):

    # User can only send below message types
    DISCONNECTED = auto()
    CONNECTED = auto()
    CONNECTING = auto()
    FAILED = auto()
