import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List

import streamlit as st
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart
from rich.console import Console

from gaf_guard.clients.stream_adaptors import get_adapter
from gaf_guard.core.models import WorkflowMessage
from gaf_guard.toolkit.enums import MessageType, Role, UserInputType
from gaf_guard.toolkit.file_utils import resolve_file_paths


GAF_GUARD_ROOT = Path(__file__).parent.parent.absolute()

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
st.session_state.drift_threshold = 8
st.session_state.priority = ["low", "medium", "high"]
st.set_page_config(
    page_title="GAF Guard - A real-time monitoring system for risk assessment and drift monitoring.",
    layout="wide",  # This sets the app to wide mode
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
        )
    },
}
resolve_file_paths(run_configs)
initial_risks = ["Toxic output", "Hallucination"]


def file_uploaded():
    st.session_state.prompt_file = st.session_state.prompt_file_uploader.getvalue()
    message = WorkflowMessage(
        name="GAF Guard Client",
        type=MessageType.HITL_QUERY,
        role=Role.SYSTEM,
        content={
            "File uploaded successfully": f"{st.session_state.prompt_file_uploader.name}"
        },
        accept=UserInputType.INPUT_PROMPT,
        run_configs=run_configs,
    )
    st.session_state.messages.append(message)
    render(message, simulate=True)


def add_sidebar():

    def update_settings(input_drift_threshold):
        st.session_state.drift_threshold = input_drift_threshold

    with st.sidebar:
        if st.session_state.sidebar_display == "settings":
            st.sidebar.title("⚙️ Settings")
            option = st.selectbox(
                "Risk Taxonomy",
                ("IBM Risk Atlas"),
            )
            input_drift_threshold = st.slider(
                "Drift Threshold",
                value=st.session_state.drift_threshold,
                min_value=2,
                max_value=10,
                step=1,
            )
            st.button(
                "Apply",
                type="primary",
                on_click=update_settings,
                args=(input_drift_threshold,),
            )

        elif st.session_state.sidebar_display == "input_prompt_source":
            st.sidebar.title("⚙️ Streaming Source")

            adapter_type = st.selectbox(
                "Select Input Prompt Source",
                ["Select", "JSON"],
                help="Choose your streaming source",
                index=0,
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
                )

            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "▶️  Start",
                    use_container_width=True,
                    disabled=(
                        adapter_type == "Select" or st.session_state.stream_active
                    ),
                ):
                    if st.session_state.setdefault(
                        "stream_adaptor",
                        get_adapter(
                            adapter_type,
                            config={"byte_data": st.session_state.prompt_file},
                        ),
                    ):
                        st.session_state.stream_active = True
                        st.rerun()
                    else:
                        st.write("Selected adaptor is not available.")

            with col2:
                if st.button(
                    "⏹️  Pause",
                    use_container_width=True,
                    disabled=(
                        adapter_type == "Select" or not st.session_state.stream_active
                    ),
                ):
                    st.session_state.stream_active = False
                    message = WorkflowMessage(
                        name="GAF Guard Client",
                        type=MessageType.HITL_QUERY,
                        role=Role.SYSTEM,
                        content={"message_alert": "Input streaming is paused."},
                        accept=UserInputType.INPUT_PROMPT,
                        run_configs=run_configs,
                    )
                    st.session_state.messages.append(message)
                    st.rerun()

        st.divider()
        st.markdown(":blue[Powered by:]")
        st.link_button(
            "AI Atlas Nexus",
            "https://github.com/IBM/ai-atlas-nexus",
            icon=":material/thumb_up:",
            type="secondary",
        )

    if hasattr(st.session_state, "client_session"):
        with st.sidebar.container(key="sidebar_bottom"):
            st.markdown(
                f"Client Id: {str(st.session_state.client_session._session.id)[0:13]} \n :violet-badge[:material/rocket_launch: Connected to :yellow[GAF Guard] Server:] :orange-badge[:material/check: {st.session_state.host}:{st.session_state.port}]",
                text_alignment="center",
            )
    else:
        with st.sidebar.container(key="sidebar_bottom"):
            st.markdown(
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
    else:
        if message.type == MessageType.WORKFLOW_STARTED:
            return False
        elif message.type == MessageType.STEP_STARTED:
            simulate_agent_response(
                role=message.role.value,
                message=f"##### :blue[Workflow Step:] **{message.name}** STARTED",
                simulate=simulate,
                accept=message.accept,
            )
        elif message.type == MessageType.STEP_COMPLETED:
            simulate_agent_response(
                role=message.role.value,
                message=f"##### :blue[Workflow Step:] **{message.name}** COMPLETED",
                simulate=simulate,
                accept=message.accept,
            )
        elif message.type == MessageType.HITL_QUERY:
            simulate_agent_response(
                role=message.role.value,
                message=f":blue[{message.content}]",
                simulate=simulate,
                accept=message.accept,
            )
        else:
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
                    "risk": initial_risks[0],
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
                    tuple(initial_risks),
                    key=f"col1{key}",
                    index=initial_risks.index(initial_risk["risk"]),
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
    if hasattr(st.session_state, "error"):
        st.error(st.session_state.error, icon="🚨")
    with st.form("login_form"):
        input_host = st.text_input("GAF Guard Host", value="localhost")
        input_port = st.number_input("GAF Guard Port", value=8000)
        submitted = st.form_submit_button("Connect", type="primary")

    if submitted:
        if hasattr(st.session_state, "error"):
            del st.session_state["error"]
        st.session_state.host = input_host
        st.session_state.port = input_port
        st.rerun()


@st.dialog(
    "GAF Guard Connect",
    width="medium",
    dismissible=False,
    icon=":material/login:",
)
def connect():

    async def ping_server(client):
        await client.ping()

    with st.status(
        f"Connecting to GAF Guard using host: :blue[**{st.session_state.host}**] and port: :blue[**{st.session_state.port}**]",
        expanded=True,
    ) as status:
        try:
            client = Client(
                base_url=f"http://{st.session_state.host}:{st.session_state.port}",
                verify=True,
            )
            asyncio.run(ping_server(client))
            st.write("Client created...")
        except Exception as e:
            st.session_state.error = "Failed to connect. Check hostname and port."
            st.rerun()

        st.session_state.client_session = client.session()
        st.write("Client session created...")

        st.session_state.disabled_input = False
        st.session_state.stream_active = False
        st.session_state.sidebar_display = "settings"
        st.session_state.messages = [
            WorkflowMessage(
                name="GAF Guard Client",
                type=MessageType.WORKFLOW_INPUT,
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
        console.print(
            f"""
        You can now view your Streamlit app in your browser.

        Local URL: http://{st.session_state.host}:{st.session_state.port}
    """
        )

        status.update(
            label=f":material/rocket_launch: Connected to :yellow[**GAF Guard**] Server: :orange-badge[:material/check: {st.session_state.host}:{st.session_state.port}]",
            state="complete",
            expanded=True,
        )
        time.sleep(1)

    st.rerun()


async def app():

    run_configs["DriftMonitoringAgent"][
        "drift_threshold"
    ] = st.session_state.drift_threshold

    st.title(f":yellow[GAF Guard]", text_alignment="center")
    st.subheader(
        "A real-time monitoring system for risk assessment and drift monitoring",
        text_alignment="center",
        divider=True,
    )

    # add sidebar and related components
    add_sidebar()

    message_container = st.container(height="stretch")
    with message_container:
        st.info("No messages yet. Start streaming to see data.")

    # Display chat messages from history
    for message in st.session_state.messages:
        render(message)

    last_message: WorkflowMessage = st.session_state.messages[-1]

    async with st.session_state.client_session:

        if st.session_state.stream_active:
            user_input = st.session_state.stream_adaptor.next()
            if not user_input:
                del st.session_state["stream_adaptor"]
                st.session_state.stream_active = False
                st.rerun()
        else:
            # Accept user input
            user_input = st.chat_input(
                placeholder="Enter your response here",
                key="user_input",
                disabled=st.session_state.disabled_input,
            )

        if not user_input:
            st.stop()
        else:
            COMPLETED = False
            while True:
                async for event in st.session_state.client_session.run_stream(
                    agent="orchestrator",
                    input=[
                        Message(
                            parts=[
                                MessagePart(
                                    content=WorkflowMessage(
                                        name="GAF Guard Client",
                                        type=(
                                            MessageType.HITL_RESPONSE
                                            if last_message.type
                                            == MessageType.HITL_QUERY
                                            else MessageType.WORKFLOW_INPUT
                                        ),
                                        role=Role.USER,
                                        content={
                                            (
                                                last_message.accept.value
                                                if last_message.accept
                                                == UserInputType.USER_INTENT
                                                else last_message.accept
                                            ): user_input
                                        },
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
                                **json.loads(
                                    event.run.await_request.message.parts[0].content
                                )
                            )
                            if not (
                                message.accept == UserInputType.INPUT_PROMPT
                                and st.session_state.stream_active
                            ):
                                render(message, simulate=True)

                            st.session_state.messages.append(message)
                            st.session_state.disabled_input = True
                            st.rerun()

                    elif event.type == "run.completed":
                        COMPLETED = True

                if COMPLETED:
                    break


if hasattr(st.session_state, "client_session"):
    asyncio.run(app())
elif (
    not hasattr(st.session_state, "error")
    and hasattr(st.session_state, "host")
    and hasattr(st.session_state, "port")
):
    os.system("clear")
    connect()
else:
    connect_screen_dialog()
