import asyncio
import json
import os
import time
from datetime import datetime
from typing import Annotated, Dict, List

import streamlit as st
import typer
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart
from rich.console import Console

from gaf_guard.core.models import WorkflowStepMessage
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.file_utils import resolve_file_paths


# Apply CSS to hide chat_input when app is running (processing)
st.markdown(
    """
<style>
.stApp[data-teststate=running] .stChatInput textarea,
.stApp[data-test-script-state=running] .stChatInput textarea {
    display: none !important;
}
.stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
.st-key-sidebar_bottom {
        position: absolute;
        bottom: 20px;
        right: 5px;
    }
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# Declare global session variables
st.session_state.priority = ["low", "medium", "high"]
st.set_page_config(
    page_title="GAF Guard - A real-time monitoring system for risk assessment and drift monitoring.",
    layout="wide",  # This sets the app to wide mode
    # initial_sidebar_state="expanded",
)
console = Console(log_time=True)
app = typer.Typer()
run_configs = {
    "RiskGeneratorAgent": {
        "risk_questionnaire_cot": "examples/data/chain_of_thought/risk_questionnaire.json",
        "risk_generation_cot": "examples/data/chain_of_thought/risk_generation.json",
    },
    "DriftMonitoringAgent": {
        "drift_monitoring_cot": "examples/data/chain_of_thought/drift_monitoring.json"
    },
}
initial_risks = ["Toxic output", "Hallucination"]
resolve_file_paths(run_configs)


def reconnect(input_host, input_port):
    # Iterate over all keys and delete them
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.session_state.host = input_host
    st.session_state.port = input_port
    st.rerun()


def set_drift_threshold(input_drift_threshold):
    st.session_state.drift_threshold = input_drift_threshold


def pprint(key, value):
    if isinstance(value, List) or isinstance(value, Dict):
        return json.dumps(value, indent=2)
    elif isinstance(value, str) and key.endswith("alert"):
        return f"[red]{value}[/red]"
    else:
        return value


def print_server_msg():
    console.print(
        f"[[bold white]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/]] [italic bold white] :rocket: Connected to GAF Guard Server at[/italic bold white] [bold white]{st.session_state.host}:{st.session_state.port}[/bold white]"
    )
    console.print(
        f"[[bold white]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/]] Client Id: {st.session_state.client_session._session.id}"
    )
    console.print(
        f"""
    You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
"""
    )
    with st.sidebar.container(key="sidebar_bottom"):
        st.markdown(
            f"Client Id: {str(st.session_state.client_session._session.id)[0:13]} \n :violet-badge[:material/rocket_launch: Connected to :yellow[GAF Guard] Server:] :orange-badge[:material/check: {st.session_state.host}:{st.session_state.port}]",
            text_alignment="center",
        )


def simulate_agent_response(role, message, json_data=None, simulate=False):
    with st.chat_message(role):
        if simulate:
            message_placeholder = st.empty()
            full_response = ""
            for chunk in message.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            st.markdown(message)

        if json_data:
            st.json(json_data, expanded=4)


def render(message: WorkflowStepMessage, simulate=False):
    if isinstance(message.content, dict):
        if message.step_name == "Input Prompt":
            # with st.chat_message(message.step_role.value):
            #     st.markdown(
            #         f":yellow[Prompt {message.content["prompt_index"]}]:  {message.content["prompt"]}"
            #     )
            simulate_agent_response(
                role=message.step_role.value,
                message=f"###### :yellow[**Prompt {message.content["prompt_index"]}**]:  {message.content["prompt"]}",
                simulate=simulate,
            )
        else:
            if len(message.content.items()) > 2:
                data = []
                for key, value in message.content.items():
                    data.append({key.title(): value})

                # st.markdown(":yellow[Risk Report]")
                simulate_agent_response(
                    role=message.step_role.value,
                    message="###### :yellow[Risk Report]",
                    json_data=data,
                    simulate=simulate,
                )
                # st.json(data)
            else:
                for key, value in message.content.items():
                    if key == "identified_risks":
                        st.session_state.risks = value
                    if isinstance(value, List) or isinstance(value, Dict):
                        # st.markdown(f":yellow[{key.replace('_', ' ').title()}]")
                        simulate_agent_response(
                            role=message.step_role.value,
                            message=f"###### :yellow[{key.replace('_', ' ').title()}]",
                            json_data=value,
                            simulate=simulate,
                        )
                    elif isinstance(value, str) and key.endswith("alert"):
                        # st.markdown(
                        #     f":yellow[{key.replace('_', ' ').title()}]: :red[{value}]"
                        # )
                        simulate_agent_response(
                            role=message.step_role.value,
                            message=f"###### :yellow[{key.replace('_', ' ').title()}]: :red[{value}]",
                            simulate=simulate,
                        )
                    else:
                        simulate_agent_response(
                            role=message.step_role.value,
                            message=f"###### :yellow[{key.replace('_', ' ').title()}]: {value}",
                            simulate=simulate,
                        )
                        # st.markdown(
                        #     f":yellow[{key.replace('_', ' ').title()}]: {value}"
                        # )
    else:
        if message.step_type == MessageType.WORKFLOW_STARTED:
            return False
            # st.markdown(f":blue[{message.step_name}]: {message.content}")
        elif message.step_type == MessageType.STEP_STARTED:
            simulate_agent_response(
                role=message.step_role.value,
                message=f"##### :blue[Workflow Step:] **{message.step_name}**",
                simulate=simulate,
            )
            # st.markdown(f"\n:blue[Workflow Step:] {message.step_name}....Started")
        elif message.step_type == MessageType.STEP_COMPLETED:
            return False
            # st.markdown(f"\n:blue[Workflow Step:] {message.step_name}....Completed")
        elif message.step_type == MessageType.HITL_QUERY:
            # st.markdown(f":blue[{message.content}]")
            simulate_agent_response(
                role=message.step_role.value,
                message=f":blue[{message.content}]",
                simulate=simulate,
            )
        else:
            # st.markdown(message.content)
            simulate_agent_response(
                role=message.step_role.value, message=message.content, simulate=simulate
            )

    return True


def add_row():
    st.session_state.setdefault("dynamic_risks", {}).update(
        {
            str(len(st.session_state.dynamic_risks)): {
                "risk": initial_risks[0],
                "priority": "low",
                "threshold": 0.01,
            }
        }
    )


@st.dialog("Initial risks", width="medium")
def vote():
    if "dynamic_risks" not in st.session_state:
        add_row()

    st.button("Add New Row", type="primary", on_click=add_row)
    with st.form("input_form"):

        # Create columns for the form inputs
        col1, col2, col3 = st.columns(3)

        for key, dynamic_risk in st.session_state.dynamic_risks.items():
            with col1:
                value = st.selectbox(
                    "Risk" if key == "0" else " ",
                    tuple(initial_risks),
                    key=f"col1{key}",
                    index=initial_risks.index(dynamic_risk["risk"]),
                )
                st.session_state.dynamic_risks[key].update({"risk": value})
            with col2:
                value = st.selectbox(
                    "Priority" if key == "0" else " ",
                    tuple(st.session_state.priority),
                    key=f"col2{key}",
                    index=st.session_state.priority.index(dynamic_risk["priority"]),
                )
                st.session_state.dynamic_risks[key].update({"priority": value})
            with col3:
                threshold = st.number_input(
                    "Threshold" if key == "0" else " ",
                    key=f"col3{key}",
                    value=dynamic_risk["threshold"],
                )
                st.session_state.dynamic_risks[key].update({"threshold": threshold})

        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state.user_input = json.dumps(
            list(st.session_state.dynamic_risks.values())
        )
        st.rerun()


with st.sidebar:
    st.sidebar.title("Settings")
    option = st.selectbox(
        "Risk Taxonomy",
        ("IBM Risk Atlas"),
    )
    input_drift_threshold = st.number_input("Drift Threshold", value=8)
    st.button(
        "Apply",
        type="primary",
        on_click=set_drift_threshold,
        args=(input_drift_threshold,),
    )

    st.divider()
    input_host = st.text_input("GAF Guard Host", value="localhost")
    input_port = st.number_input("GAF Guard Port", value=8000)
    st.button(
        "Reconnect",
        type="primary",
        on_click=reconnect,
        args=(input_host, input_port),
    )

    st.divider()
    st.markdown(":blue[Powered by:]")
    st.link_button(
        "AI Atlas Nexus",
        "https://github.com/IBM/ai-atlas-nexus",
        icon=":material/thumb_up:",
        type="secondary",
    )


async def run_app():

    if "client_session" not in st.session_state:
        client = Client(
            base_url=f"http://{st.session_state.host}:{st.session_state.port}"
        )
        st.session_state.client_session = client.session()
        st.session_state.input_message_type = MessageType.WORKFLOW_INPUT
        st.session_state.input_message_query = "Enter user intent here"
        st.session_state.response_type_needed = "None"
        st.session_state.input_message_key = "user_intent"
        st.session_state.disabled_input = False
        st.session_state.drift_threshold = 8
        st.session_state.messages = []

    run_configs["DriftMonitoringAgent"][
        "drift_threshold"
    ] = st.session_state.drift_threshold

    print_server_msg()
    st.title(
        f":yellow[GAF Guard]",
        text_alignment="center",
    )
    st.subheader(
        "A real-time monitoring system for risk assessment and drift monitoring",
        text_alignment="center",
        divider=True,
    )

    # Display chat messages from history
    for message in st.session_state.messages:
        render(message)

    async with st.session_state.client_session:

        # Accept user input
        if st.session_state.response_type_needed == "dynamic_risks":
            st.button("Add Initial Risks", on_click=vote)
        user_input = st.chat_input(
            st.session_state["input_message_query"], key="user_input"
        )

        if not user_input:
            st.stop()

        # progress = st.status(label="Loading data!")
        COMPLETED = False
        while True:
            async for event in st.session_state.client_session.run_stream(
                agent="orchestrator",
                input=[
                    Message(
                        parts=[
                            MessagePart(
                                content=WorkflowStepMessage(
                                    step_name="GAF Guard Client",
                                    step_type=st.session_state.input_message_type,
                                    step_role=Role.USER,
                                    content={
                                        st.session_state.input_message_key: user_input
                                    },
                                    run_configs=run_configs,
                                ).model_dump_json(),
                                content_type="text/plain",
                            )
                        ]
                    )
                ],
            ):
                # progress.update(
                #     label="Download complete!", state="complete", expanded=False
                # )
                if event.type == "message.part":
                    message = WorkflowStepMessage(**json.loads(event.part.content))
                    if render(message, simulate=True):
                        st.session_state.messages.append(message)
                    # progress.update(label="Loading data!", state="running")
                elif event.type == "run.awaiting":
                    if hasattr(event, "run"):
                        message = WorkflowStepMessage(
                            **json.loads(
                                event.run.await_request.message.parts[0].content
                            )
                        )
                        st.session_state.messages.append(message)
                        render(message, simulate=True)
                        st.session_state.input_message_type = MessageType.HITL_RESPONSE
                        st.session_state.input_message_key = "response"
                        st.session_state.input_message_query = message.step_kwargs[
                            "input_message_query"
                        ]
                        st.session_state.response_type_needed = message.step_kwargs[
                            "response_type_needed"
                        ]
                        st.rerun()
                        # COMPLETED = True

                elif event.type == "run.completed":
                    COMPLETED = True

            if COMPLETED:
                break


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
    ] = 8000,
):
    os.system("clear")
    st.session_state.host = host
    st.session_state.port = port
    asyncio.run(run_app())


if __name__ == "__main__":
    app()
