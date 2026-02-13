from typing import Any, Dict, Optional

from pydantic import BaseModel

from gaf_guard.toolkit.enums import MessageType, Role, UserInputType


class WorkflowMessage(BaseModel):

    name: str
    role: Role
    type: MessageType
    desc: Optional[str] = None
    content: Optional[Any] = None
    accept: Optional[UserInputType] = None
    run_configs: Optional[Dict] = None
