import logging
import os
import subprocess
import sys

import typer
from rich.console import Console

from gaf_guard.serve import start_server
from gaf_guard.toolkit.logging import configure_logger


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.ERROR)

LOGGER = configure_logger(__name__)

app = typer.Typer()
console = Console()


@app.callback()
def main() -> None:
    """
    GAF Guard Redirect Server
    """


@app.command()
def benchmark(config_file): ...


@app.command()
def serve(config_file):
    start_server(config_file)


@app.command()
def client(client_type):
    os.system("clear")
    console.rule(f"[bold blue]Launching GAF Guard Client - {client_type}[/bold blue]")
    try:
        process = subprocess.Popen(
            ["streamlit", "run", "src/gaf_guard/clients/streamlit.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Read output line by line in real-time
        while True:
            output = process.stdout.readline()
            # Check if the process has finished and there is no more output
            if output == "" and process.poll() is not None:
                break
            if output:
                # Print the output immediately
                print(output.strip())
                # Flush the output to ensure it's displayed immediately, not buffered by Python's stdout
                sys.stdout.flush()

        # Wait for the process to fully terminate and get the return code
        return_code = process.wait()
        return return_code
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}. Error:")
        print(e.stderr)


# if __name__ == "__main__":
#     app()
