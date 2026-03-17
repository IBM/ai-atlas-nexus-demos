import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from gaf_guard.clients.benchmark import run_benchmark
from gaf_guard.serve import start_server
from gaf_guard.toolkit.logging import configure_logger


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.ERROR)

LOGGER = configure_logger(__name__)
PACKAGEDIR = Path(__file__).parent.absolute()

app = typer.Typer()
console = Console()


@app.callback()
def main() -> None:
    """
    GAF Guard Redirect Server
    """


@app.command()
def benchmark(
    ground_trial: Annotated[
        str,
        typer.Argument(
            help="Please enter ground truth file path.",
            rich_help_panel="Ground Truth Trial",
        ),
    ],
    host: Annotated[
        str,
        typer.Argument(help="Please enter GAF Guard Host.", rich_help_panel="Hostname"),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Argument(help="Please enter GAF Guard Port.", rich_help_panel="Port"),
    ] = 8000,
):
    asyncio.run(run_benchmark(ground_trial, host, port))


@app.command()
def serve(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            help="Please enter GAF Guard config file location.",
            rich_help_panel="Configuration File",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(help="Please enter GAF Guard Host.", rich_help_panel="Hostname"),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(help="Please enter GAF Guard Port.", rich_help_panel="Port"),
    ] = 8000,
):
    start_server(config, host, port)


@app.command()
def client(
    ctx: typer.Context,
    type: Annotated[
        str,
        typer.Option(
            help="Please enter GAF Guard client type.",
            rich_help_panel="Configuration File",
        ),
    ],
    host: Annotated[
        str,
        typer.Option(help="Please enter GAF Guard Host.", rich_help_panel="Hostname"),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(help="Please enter GAF Guard Port.", rich_help_panel="Port"),
    ] = None,
):
    os.system("clear")
    console.rule(f"[bold blue]Launching GAF Guard {type.title()} Client[/bold blue]")
    try:
        if type == "streamlit":
            process = subprocess.Popen(
                ["streamlit", "run", f"src/gaf_guard/clients/{type}.py"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        elif type == "cli":
            if port:
                args = [
                    "python",
                    f"{str(PACKAGEDIR)}/clients/{type}.py",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ]
            else:
                args = [
                    "python",
                    f"{str(PACKAGEDIR)}/clients/{type}.py",
                    "--host",
                    host,
                ]
            process = subprocess.Popen(args, stderr=subprocess.STDOUT, text=True)

        # Wait for the process to fully terminate and get the return code
        return_code = process.wait()
        return return_code
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}. Error:")
        print(e.stderr)


if __name__ == "__main__":
    app()
