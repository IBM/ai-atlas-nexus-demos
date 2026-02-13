import json
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress

from gaf_guard.core.models import WorkflowMessage
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.logging import configure_logger


STATUS_DISPLAY = {}
console = Console()
LOGGER = configure_logger(__name__)


def workflow(
    name: Optional[str] = None,
    desc: Optional[str] = None,
    role: Role = Role.SYSTEM,
    log_output: bool = True,
):

    def decorator(func):

        def wrapper(*args, config: RunnableConfig, **kwargs):
            client_id = config.get("configurable", {}).get("thread_id", 1)
            console.print()
            console.print(
                Panel(
                    Group(
                        f"Incoming request:\n{json.dumps(args[0].model_dump(include=set({'user_intent', 'prompt'}), exclude_none=True), indent=2)}"
                    ),
                    title=f"{config.get('configurable', {}).get('trial_name', 'Trial_')} | Client: {client_id}",
                )
            )
            console.print()

            write_to_stream = get_stream_writer()
            message = WorkflowMessage(
                name=name or func.__name__,
                type=MessageType.WORKFLOW_STARTED,
                role=role,
                desc=desc,
                content="New Workflow Started",
            )
            write_to_stream(
                {"client": message} | ({"logger": message} if log_output else {})
            )

            # Call the actual graph node
            return func(*args, **kwargs, config=config)

        return wrapper

    return decorator


def invoke_agent(
    name: Optional[str] = None,
    desc: Optional[str] = None,
    role: Role = Role.SYSTEM,
    log_output: bool = True,
):

    def decorator(func):

        def wrapper(*args, config: RunnableConfig, **kwargs):
            client_id = config.get("configurable", {}).get("thread_id", 1)
            agent_name = args[0]._WORKFLOW_NAME
            display = STATUS_DISPLAY.setdefault(
                client_id,
                {
                    "live": Live(console=Console()),
                    "progress": Progress(),
                    "current_task": None,
                },
            )
            if display["current_task"]:
                display["progress"].update(
                    display["current_task"]["task_id"],
                    completed=100,
                    description=f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{display['current_task']['name']}[/bold white][bold yellow]...Completed[/bold yellow]",
                    refresh=True,
                )

            display["current_task"] = {
                "task_id": display["progress"].add_task(
                    f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{agent_name}[/bold white]",
                    total=None,
                ),
                "name": agent_name,
            }

            # display["live"].start()
            # display["live"].update(
            #     Panel(
            #         Group(
            #             f"Incoming request:\n{json.dumps(args[1].model_dump(include=set({'user_intent', 'prompt'}), exclude_none=True), indent=2)}\n",
            #             display["progress"],
            #         ),
            #         title=f"{config.get('configurable', {}).get('trial_name', 'Trial_')} | Client: {client_id}",
            #     )
            # )
            console.log(
                f"[bold yellow]Invoking Agent[/bold yellow][bold white]...{agent_name}[/bold white]"
            )

            # Call the actual graph node
            return func(*args, **kwargs, config=config)

        return wrapper

    return decorator


def workflow_step(
    step_name: Optional[str] = None,
    step_desc: Optional[str] = None,
    step_role: Role = Role.AGENT,
    log_output: bool = True,
    **step_kwargs,
):
    def decorator(func):

        def wrapper(*args, config: RunnableConfig, **kwargs):

            write_to_stream = get_stream_writer()
            message = WorkflowMessage(
                type=MessageType.STEP_STARTED,
                role=Role.SYSTEM,
                name=step_name or func.__name__,
                desc=step_desc,
                kwargs=step_kwargs,
            )
            write_to_stream({"client": message})

            # Call the actual graph node
            event = func(*args, **kwargs, config=config)

            event_message = message.model_copy(
                update={
                    "role": step_role,
                    "type": MessageType.STEP_DATA,
                    "content": event,
                }
            )
            write_to_stream(
                {"client": event_message}
                | ({"logger": event_message} if log_output else {})
            )
            write_to_stream(
                {
                    "client": message.model_copy(
                        update={
                            "type": MessageType.STEP_COMPLETED,
                            "role": Role.SYSTEM,
                        }
                    )
                }
            )

            return event

        return wrapper

    return decorator
