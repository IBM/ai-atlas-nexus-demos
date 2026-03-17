import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import streamlit as st
from acp_sdk.models import Message, MessagePart
from rich.console import Console

from gaf_guard.clients.login_handler import LoginHandler
from gaf_guard.clients.stream_adaptors import get_adapter
from gaf_guard.core.models import WorkflowMessage
from gaf_guard.toolkit.enums import (
    MessageType,
    Role,
    ServerStatus,
    StreamStatus,
    UserInputType,
)
from gaf_guard.toolkit.file_utils import resolve_file_paths


GAF_GUARD_ROOT = Path(__file__).parent.parent.absolute()


def signal_handler(sig, frame):
    print("Exiting...")
    for task in asyncio.tasks.all_tasks():
        task.cancel()
    sys.exit(0)


# Apply CSS to hide chat_input when app is running (processing)
st.markdown(
    """
<style>
.header {
    padding: 1rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.message-card {
    padding: 1rem;
    border-left: 4px solid #667eea;
    background-color: #f8f9fa;
    border-radius: 5px;
    margin: 0.5rem 0;
}
.stApp[data-teststate=running] .stChatInput textarea,
.stApp[data-test-script-state=running] .stChatInput textarea {
    display: none !important;
}
.stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
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
if "server_status" not in st.session_state:
    st.session_state.server_status = ServerStatus.DISCONNECTED
if "host" not in st.session_state:
    st.session_state.host = "http://localhost"
if "port" not in st.session_state:
    st.session_state.port = 8000
st.session_state.priority = ["low", "medium", "high"]
st.session_state.initial_risks_master = ["Toxic output", "Hallucination"]
st.set_page_config(
    page_title="GAF Guard - A real-time monitoring system for risk assessment and drift monitoring.",
    layout="wide",  # This sets the app to wide mode
    menu_items={
        "Get Help": "https://github.com/IBM/ai-atlas-nexus-demos/tree/main/gaf-guard",
        "Report a bug": "https://github.com/IBM/ai-atlas-nexus-demos/issues",
        "About": "### GAF-Guard \n GAF Guard is an AI framework that can effectively detect and manage risks associated with LLMs for a given use-case. The framework leverages agents to identify risks tailored to a specific use case, generate drift and risk monitors, and establish real-time monitoring functions for LLMs. By integrating these capabilities, our approach aims to provide a comprehensive risk management framework that addresses the unique requirements of each LLM application.",
    },
    # initial_sidebar_state="expanded",
)

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
        ),
        "drift_threshold": (
            st.session_state.drift_threshold
            if "drift_threshold" in st.session_state
            else 8
        ),
    },
}
resolve_file_paths(run_configs)


def file_uploaded():
    st.session_state.prompt_file = st.session_state.prompt_file_uploader.getvalue()
    message = WorkflowMessage(
        name="GAF Guard Client",
        type=MessageType.GAF_GUARD_QUERY,
        role=Role.SYSTEM,
        content=f"**File uploaded successfully:** {st.session_state.prompt_file_uploader.name}. Please click on **Start** to begin risk assessment.",
        accept=UserInputType.INPUT_PROMPT,
        run_configs=run_configs,
    )
    st.session_state.messages.append(message)
    render(message, simulate=True)


def play_button(adapter_type):
    if st.session_state.setdefault(
        "stream_adaptor",
        get_adapter(
            adapter_type,
            config={"byte_data": st.session_state.prompt_file},
        ),
    ):
        st.session_state.stream_status = StreamStatus.ACTIVE
    else:
        st.write("Selected adaptor is not available.")


def pause_button():
    st.session_state.stream_status = StreamStatus.PAUSED
    st.session_state.messages.append(
        WorkflowMessage(
            name="GAF Guard Client",
            type=MessageType.GAF_GUARD_QUERY,
            role=Role.SYSTEM,
            content=f"**:red[Alert:]** Current input stream is paused. Please click on **Start** to resume.",
            accept=UserInputType.INPUT_PROMPT,
            run_configs=run_configs,
        )
    )


@st.fragment
def pause_fragment(adapter_type):
    st.button(
        "⏸️  Pause",
        use_container_width=True,
        disabled=(
            adapter_type == "Select"
            or st.session_state.stream_status
            in [StreamStatus.STOPPED, StreamStatus.PAUSED]
        ),
        on_click=pause_button,
    )


def add_sidebar():

    with st.sidebar:
        st.sidebar.title("⚙️ Settings")
        if st.session_state.sidebar_display in ["settings_view", "input_prompt_source"]:
            st.subheader(f":blue[Taxonomy:] {st.session_state.taxonomy}")
            st.subheader(f":blue[Drift Threshold:] {st.session_state.drift_threshold}")
        if st.session_state.sidebar_display == "settings_edit":
            st.session_state.taxonomy = st.selectbox(
                "Risk Taxonomy",
                ("IBM Risk Atlas"),
            )
            st.session_state.drift_threshold = st.slider(
                "Drift Threshold",
                value=st.session_state.drift_threshold,
                min_value=2,
                max_value=10,
                step=1,
            )
        if st.session_state.sidebar_display == "input_prompt_source":
            st.sidebar.title("⚙️ Streaming Source")
            adapter_type = st.selectbox(
                "Select Input Prompt Source",
                ["Select", "JSON"],
                help="Choose your streaming source",
                index=0,
                disabled="stream_adaptor" in st.session_state,
            )
            if adapter_type == "JSON":
                st.subheader("JSON File Source")
                st.file_uploader(
                    "OK",
                    accept_multiple_files=False,
                    type="json",
                    label_visibility="collapsed",
                    on_change=file_uploaded,
                    key="prompt_file_uploader",
                    disabled="stream_adaptor" in st.session_state,
                )

            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                st.button(
                    "▶️  Start",
                    use_container_width=True,
                    disabled=(
                        adapter_type == "Select"
                        or "prompt_file" not in st.session_state
                        or st.session_state.stream_status == StreamStatus.ACTIVE
                    ),
                    on_click=play_button,
                    args=(adapter_type,),
                )
            with col2:
                pause_fragment(adapter_type)
            st.markdown(
                "**Note:** Pause button will temporarily halt the stream after processing the current prompt."
            )

        st.divider()
        ai_atlas_button = st.container(
            horizontal_alignment="center", vertical_alignment="bottom", height="stretch"
        )
        ai_atlas_button.markdown(":blue[Powered by:]", text_alignment="center")
        ai_atlas_button.link_button(
            "AI Atlas Nexus",
            "https://github.com/IBM/ai-atlas-nexus",
            icon=":material/thumb_up:",
            type="secondary",
        )
        if hasattr(st.session_state, "client_session"):
            ai_atlas_button.markdown(
                f"Client Id: {str(st.session_state.client_session._session.id)[0:13]} \n :violet-badge[:material/rocket_launch: Connected to :yellow[GAF Guard] Server:] :orange-badge[:material/check: {st.session_state.base_url}]",
                text_alignment="center",
            )
        else:
            ai_atlas_button.markdown(
                f":red-badge[:material/mimo_disconnect: Client Disconnected]",
                text_alignment="center",
            )


# render agent reponse from the server
def render(message: WorkflowMessage, simulate=False):

    def simulate_agent_response(
        role: Role,
        message: str,
        json_data: Dict = None,
        simulate: bool = False,
        accept: Dict = None,
    ):
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
            elif accept == UserInputType.INITIAL_RISKS:
                st.button(
                    "Add Initial Risks",
                    on_click=initial_risks_selector,
                    disabled=hasattr(st.session_state, "initial_risks"),
                )
                st.session_state.disabled_input = False
            elif accept == UserInputType.INPUT_PROMPT:
                st.session_state.sidebar_display = "input_prompt_source"
                st.session_state.disabled_input = True

    if not message.display:
        return False
    if message.type == MessageType.GAF_GUARD_WF_STARTED:
        return False
    if message.type == MessageType.GAF_GUARD_WF_COMPLETED:
        return False
    elif message.type == MessageType.GAF_GUARD_STEP_STARTED:
        simulate_agent_response(
            role=message.role.value,
            message=f"##### :blue[Workflow Step:] **{message.name}**",
            simulate=simulate,
            accept=message.accept,
        )
    elif message.type == MessageType.GAF_GUARD_STEP_COMPLETED:
        # simulate_agent_response(
        #     role=message.role.value,
        #     message=f"##### :blue[Workflow Step:] **{message.name}** COMPLETED",
        #     simulate=simulate,
        #     accept=message.accept,
        # )
        return False
    elif message.type == MessageType.GAF_GUARD_STEP_DATA:
        if isinstance(message.content, dict):
            if message.name == "Input Prompt":
                simulate_agent_response(
                    role=message.role.value,
                    message=f"###### :yellow[**Prompt {message.content["prompt_index"]}**]:  {message.content["prompt"]}",
                    simulate=simulate,
                    accept=message.accept,
                )
            else:
                if len(message.content.items()) > 2:
                    data = []
                    for key, value in message.content.items():
                        data.append({key.title(): value})

                    simulate_agent_response(
                        role=message.role.value,
                        message="###### :yellow[Risk Report]",
                        json_data=data,
                        simulate=simulate,
                        accept=message.accept,
                    )
                else:
                    for key, value in message.content.items():
                        if key == "identified_risks":
                            st.session_state.risks = value
                        if isinstance(value, List) or isinstance(value, Dict):
                            simulate_agent_response(
                                role=message.role.value,
                                message=f"###### :yellow[{key.replace('_', ' ').title()}]",
                                json_data=value,
                                simulate=simulate,
                                accept=message.accept,
                            )
                        elif isinstance(value, str) and key.endswith("alert"):
                            simulate_agent_response(
                                role=message.role.value,
                                message=f"###### :yellow[{key.replace('_', ' ').title()}]: :red[{value}]",
                                simulate=simulate,
                                accept=message.accept,
                            )
                        else:
                            simulate_agent_response(
                                role=message.role.value,
                                message=f"###### :yellow[{key.replace('_', ' ').title()}]: {value}",
                                simulate=simulate,
                                accept=message.accept,
                            )
    elif message.type == MessageType.GAF_GUARD_QUERY:
        simulate_agent_response(
            role=message.role.value,
            message=f":blue[{message.content}]",
            simulate=simulate,
            accept=message.accept,
        )
    else:
        # raise Exception(f"Invalid message type: {message.type}")
        if message.content:
            simulate_agent_response(
                role=message.role.value,
                message=message.content,
                simulate=simulate,
                accept=message.accept,
            )

    return True


@st.dialog("Initial risks", width="medium")
def initial_risks_selector():

    def add_row():
        st.session_state.setdefault("initial_risks", {}).update(
            {
                str(len(st.session_state.initial_risks)): {
                    "risk": st.session_state.initial_risks_master[0],
                    "priority": "low",
                    "threshold": 0.01,
                }
            }
        )

    if "initial_risks" not in st.session_state:
        add_row()

    st.button("Add New Row", type="primary", on_click=add_row)
    with st.form("input_form"):

        # Create columns for the form inputs
        col1, col2, col3 = st.columns(3)

        for key, initial_risk in st.session_state.initial_risks.items():
            with col1:
                value = st.selectbox(
                    "Risk" if key == "0" else " ",
                    tuple(st.session_state.initial_risks_master),
                    key=f"col1{key}",
                    index=st.session_state.initial_risks_master.index(
                        initial_risk["risk"]
                    ),
                )
                st.session_state.initial_risks[key].update({"risk": value})
            with col2:
                value = st.selectbox(
                    "Priority" if key == "0" else " ",
                    tuple(st.session_state.priority),
                    key=f"col2{key}",
                    index=st.session_state.priority.index(initial_risk["priority"]),
                )
                st.session_state.initial_risks[key].update({"priority": value})
            with col3:
                threshold = st.number_input(
                    "Threshold" if key == "0" else " ",
                    key=f"col3{key}",
                    value=initial_risk["threshold"],
                )
                st.session_state.initial_risks[key].update({"threshold": threshold})

        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state.user_input = json.dumps(
            list(st.session_state.initial_risks.values())
        )
        st.rerun()


@st.dialog(
    "GAF Guard Connect",
    width="medium",
    dismissible=False,
    icon=":material/login:",
)
def connect_screen_dialog():

    def server_connect():
        st.session_state.server_status = ServerStatus.CONNECTING

    with st.form("login_form"):
        st.session_state.host = st.text_input(
            "**GAF Guard Host**",
            value=st.session_state.host,
            disabled=st.session_state.server_status == ServerStatus.CONNECTING,
        )
        st.session_state.port = st.number_input(
            "**GAF Guard Port**",
            value=st.session_state.port,
            step=1,
            disabled=st.session_state.server_status == ServerStatus.CONNECTING,
        )
        submitted = st.form_submit_button(
            "Connect", type="primary", on_click=server_connect
        )

    if st.session_state.server_status == ServerStatus.FAILED:
        st.error(st.session_state.error, icon="🚨")
    elif submitted:
        st.session_state.base_url = (
            st.session_state.host + f":{int(st.session_state.port)}"
        )
        with st.status(
            f"Connecting to GAF Guard using host: :blue[**{st.session_state.base_url}**]",
            expanded=True,
        ) as status:
            try:
                login_handler = LoginHandler(
                    st.session_state.host, st.session_state.port
                )

                # ping server for health check
                asyncio.run(login_handler.health_check())

                client = login_handler.create_client()
                st.write("Client created...")
                time.sleep(0.5)
                st.session_state.client_session = client.session()
                st.write("Client session created...")
            except Exception as e:
                st.session_state.server_status = ServerStatus.FAILED
                st.session_state.error = str(e)
            else:
                time.sleep(0.5)
                st.session_state.server_status = ServerStatus.CONNECTED

                st.session_state.drift_threshold = 8
                st.session_state.disabled_input = False
                st.session_state.stream_status = StreamStatus.STOPPED
                st.session_state.sidebar_display = "settings_edit"
                st.session_state.messages = [
                    WorkflowMessage(
                        name="GAF Guard Client",
                        type=MessageType.CLIENT_INPUT,
                        role=Role.USER,
                        accept=UserInputType.USER_INTENT,
                        run_configs=run_configs,
                    )
                ]
                st.write("Client initialisation done...")

                # print information in the client console window
                console.print(
                    f"[[bold white]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/]] [italic bold white] :rocket: Connected to GAF Guard Server at[/italic bold white] [bold white]{st.session_state.host}:{st.session_state.port}[/bold white]"
                )
                console.print(
                    f"[[bold white]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/]] Client Id: {st.session_state.client_session._session.id}"
                )

                status.update(
                    label=f":material/rocket_launch: Connected to :yellow[**GAF Guard**] Server: :orange-badge[:material/check: {st.session_state.base_url}]",
                    state="complete",
                    expanded=True,
                )
                time.sleep(1)
            finally:
                st.rerun()


async def app():

    def submit_input():
        st.session_state.sidebar_display = "settings_view"

    st.title(f":yellow[GAF Guard]", text_alignment="center")
    st.subheader(
        "A real-time monitoring system for risk assessment and drift monitoring",
        text_alignment="center",
        divider=True,
    )

    # add sidebar and related components
    add_sidebar()

    if len(st.session_state.messages) > 1:
        # Display chat messages from history
        for message in st.session_state.messages:
            render(message)
    else:
        message_container = st.container(height="stretch")
        with message_container:
            st.info(
                "Begin a new workflow. Please enter your intent in the text box below."
            )

    last_message: WorkflowMessage = st.session_state.messages[-1]

    if st.session_state.stream_status == StreamStatus.ACTIVE:
        user_input = st.session_state.stream_adaptor.next()
        if not user_input:
            del st.session_state["stream_adaptor"]
            st.session_state.stream_status = StreamStatus.STOPPED
            st.session_state.messages.append(
                WorkflowMessage(
                    name="GAF Guard Client",
                    type=MessageType.GAF_GUARD_QUERY,
                    role=Role.SYSTEM,
                    content=f"**The streaming input has ended. Please choose a streaming source and start again.**",
                    accept=UserInputType.INPUT_PROMPT,
                    run_configs=run_configs,
                )
            )
            st.rerun()
    else:
        # Accept user input
        user_input = st.chat_input(
            placeholder="Enter your response here",
            key="user_input",
            disabled=st.session_state.disabled_input,
            on_submit=submit_input,
        )

    if not user_input:
        st.stop()
    else:
        async for event in st.session_state.client_session.run_stream(
            agent="orchestrator",
            input=[
                Message(
                    parts=[
                        MessagePart(
                            content=WorkflowMessage(
                                name="GAF Guard Client",
                                type=(
                                    MessageType.CLIENT_RESPONSE
                                    if last_message.type == MessageType.GAF_GUARD_QUERY
                                    else MessageType.CLIENT_INPUT
                                ),
                                role=Role.USER,
                                content={last_message.accept: user_input},
                                run_configs=run_configs,
                            ).model_dump_json(),
                            content_type="text/plain",
                        )
                    ]
                )
            ],
        ):
            if event.type == "message.part":
                message = WorkflowMessage(**json.loads(event.part.content))
                if render(message, simulate=True):
                    st.session_state.messages.append(message)
            elif event.type == "run.awaiting":
                if hasattr(event, "run"):
                    message = WorkflowMessage(
                        **json.loads(event.run.await_request.message.parts[0].content)
                    )
                    if message.accept == UserInputType.INPUT_PROMPT:
                        if st.session_state.stream_status == StreamStatus.STOPPED:
                            render(message, simulate=True)
                        else:
                            message.display = False
                    else:
                        render(message, simulate=True)

                    st.session_state.messages.append(message)
                    st.session_state.disabled_input = True
                    st.rerun()


if st.session_state.server_status == ServerStatus.CONNECTED:
    asyncio.run(app())
elif st.session_state.server_status in [
    ServerStatus.DISCONNECTED,
    ServerStatus.FAILED,
]:
    connect_screen_dialog()
