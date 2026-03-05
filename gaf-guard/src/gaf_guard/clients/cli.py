import asyncio
import json

#!/usr/bin/env python
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List

import typer
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt

from gaf_guard.clients.login_handler import LoginHandler
from gaf_guard.clients.stream_adaptors import get_adapter
from gaf_guard.core.models import UserInputType, WorkflowMessage
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.file_utils import resolve_file_paths


app = typer.Typer()


def signal_handler(sig, frame):
    print("Exiting...")
    for task in asyncio.tasks.all_tasks():
        task.cancel()
    sys.exit(0)


def pprint(key, value):
    if isinstance(value, List) or isinstance(value, Dict):
        return json.dumps(value, indent=2)
    elif isinstance(value, str) and key.endswith("alert"):
        return f"[red]{value}[/red]"
    else:
        return value


STREAM_ADAPTORS = {}
GAF_GUARD_ROOT = Path(__file__).parent.parent.absolute()
signal.signal(signal.SIGINT, signal_handler)

console = Console(log_time=True)

run_configs = {
    "RiskGeneratorAgent": {
        "risk_questionnaire_cot": os.path.join(
            GAF_GUARD_ROOT, "chain_of_thought", "risk_questionnaire.json"
        )
    },
    "DriftMonitoringAgent": {
        "drift_monitoring_cot": os.path.join(
            GAF_GUARD_ROOT, "chain_of_thought", "drift_monitoring.json"
        )
    },
}
resolve_file_paths(run_configs)


async def run_cli_client(login_handler):
    status = console.status(
        f"[bold yellow] Trying to connect to [italic blue][GAF Guard][/italic blue] using host: [bold white]{login_handler.host}[/] and port: [bold white]{login_handler.port}[/]. To abort press CTRL+C",
    )
    processing = console.status(
        "[italic bold yellow]Processing...[/]",
        spinner_style="status.spinner",
    )
    with Live(Group(status), console=console, screen=True) as live:
        async with (
            login_handler.create_client() as client,
            client.session() as session,
        ):
            status.update(f"[bold yellow] :bell: Successfully connected.[/]")
            time.sleep(2)
            live.stop()
            console.print(
                Panel(
                    Group(
                        Align.center(
                            f"\nA real-time monitoring system for risk assessment and drift monitoring.\n",
                            vertical="middle",
                        ),
                    ),
                    subtitle=f"[[bold white]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/]] [italic bold white] :rocket: Connected to GAF Guard Server at[/italic bold white] [bold white]localhost:8000[/bold white]",
                    title="[bold green]GAF Guard[/]\n",
                    border_style="blue",
                )
            )
            input_message_type = MessageType.CLIENT_INPUT
            input_message_content = {
                UserInputType.USER_INTENT: Prompt.ask(
                    prompt=f"\n[bold blue]Enter your intent[/bold blue]",
                    console=console,
                )
            }
            COMPLETED = False
            while True:
                processing.start()
                async for event in session.run_stream(
                    agent="orchestrator",
                    input=[
                        Message(
                            parts=[
                                MessagePart(
                                    content=WorkflowMessage(
                                        name="GAF Guard Client",
                                        type=input_message_type,
                                        role=Role.USER,
                                        content=input_message_content,
                                        run_configs=run_configs,
                                    ).model_dump_json(),
                                    content_type="text/plain",
                                )
                            ]
                        )
                    ],
                ):
                    processing.stop()
                    if event.type == "message.part":
                        message = WorkflowMessage(**json.loads(event.part.content))

                        if message.type == MessageType.GAF_GUARD_WF_STARTED:
                            print()
                            Console(width=None).rule(
                                f"[bold blue]{message.name}[/]: {message.content}"
                            )
                        elif message.type == MessageType.GAF_GUARD_STEP_STARTED:
                            console.print(
                                f"\n[bold blue]Workflow Step: [bold white]{message.name}[/bold white]....Started"
                            )
                            if message.desc:
                                console.print(message.desc)
                        elif message.type == MessageType.GAF_GUARD_STEP_COMPLETED:
                            console.print(
                                f"[bold blue]Workflow Step: [bold white]{message.name}[/bold white]....Completed",
                            )
                        elif message.type == MessageType.GAF_GUARD_STEP_DATA:
                            if isinstance(message.content, dict):
                                for key, value in message.content.items():
                                    if key == "risk_report":
                                        for (
                                            risk_report_key,
                                            risk_report_value,
                                        ) in value.items():
                                            console.print(
                                                f"[bold yellow]Check for {risk_report_key.title()}[/bold yellow]: {pprint(risk_report_key, risk_report_value)}"
                                            )
                                    else:
                                        console.print(
                                            f"[bold yellow]{key.replace('_', ' ').title()}[/bold yellow]: {pprint(key, value)}"
                                        )
                            else:
                                console.print(message.content)
                    elif event.type == "run.awaiting":
                        if hasattr(event, "run"):
                            message = WorkflowMessage(
                                **json.loads(
                                    event.run.await_request.message.parts[0].content
                                )
                            )
                            if message.accept == UserInputType.INPUT_PROMPT:
                                prompt = None
                                if "JSON" in STREAM_ADAPTORS:
                                    prompt = STREAM_ADAPTORS["JSON"].next()
                                if not prompt or "JSON" not in STREAM_ADAPTORS:
                                    console.print(
                                        f"[bold blue]{message.content}[/bold blue]: [bold]JSON[/bold]"
                                    )
                                    prompt_file = Prompt.ask(
                                        prompt=f"[bold blue]Please enter JSON file path[/bold blue]",
                                        console=console,
                                    )
                                    STREAM_ADAPTORS["JSON"] = get_adapter(
                                        "JSON",
                                        config={
                                            "byte_data": Path(prompt_file).read_bytes()
                                        },
                                    )
                                    prompt = STREAM_ADAPTORS["JSON"].next()
                                input_message_content = {message.accept: prompt}
                            else:
                                input_message_content = {
                                    message.accept: Prompt.ask(
                                        prompt=f"[bold blue]{message.content}[/bold blue]",
                                        console=console,
                                        show_choices=False,
                                    )
                                }
                            input_message_type = MessageType.CLIENT_RESPONSE
                    elif event.type == "run.completed":
                        COMPLETED = True
                    processing.start()

                if COMPLETED:
                    processing.stop()
                    break


async def ping_server(host: str, port: int):
    try:
        await Client(
            base_url=f"http://{host}:{port}",
            verify=True,
        ).ping()
    except:
        raise Exception("Failed to connect. Please check hostname and port.")


@app.command()
def main(
    host: Annotated[
        str,
        typer.Option(
            help="Please enter GAF Guard Host.",
            rich_help_panel="Hostname",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Please enter GAF Guard Port.",
            rich_help_panel="Port",
        ),
    ] = None,
):
    os.system("clear")

    login_handler = LoginHandler(host, port)

    # ping server for health check
    asyncio.run(login_handler.health_check())

    asyncio.run(run_cli_client(login_handler))


if __name__ == "__main__":
    app()
