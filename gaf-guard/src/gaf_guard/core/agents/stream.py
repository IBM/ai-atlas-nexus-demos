import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt
from pydantic import BaseModel

from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow_step
from gaf_guard.core.models import WorkflowStepMessage
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.exceptions import HumanInterruptionException


PROMPT_GEN = {}
RANDOM_INDICES = []


# Graph state
class StreamAgentState(BaseModel):
    prompt: Optional[str] = None
    prompt_index: Optional[int] = None
    random_indices: Optional[List[int]] = [0]


# Node
def next_prompt(state: StreamAgentState, config: RunnableConfig):
    try:
        client_id = config.get("configurable", {}).get("thread_id", "Client_1")
        index, prompt = next(PROMPT_GEN.get(client_id, iter([])))
        RANDOM_INDICES = [1, 2, 7, 9]  # sorted(random.sample(range(10),4))
        # return {"prompt_index": index, "prompt": prompt, "random_indices": RANDOM_INDICES}
        return {"prompt_index": index, "prompt": prompt}
    except StopIteration:
        return {"prompt_index": None, "prompt": None}


# Node
def is_next_prompt_available(state: StreamAgentState):
    if state.prompt:
        return True
    else:
        return False


# Node
def load_input_prompts(state: StreamAgentState, config: RunnableConfig):
    syntax_error = False
    while True:
        try:
            choice = interrupt(
                WorkflowStepMessage(
                    step_type=MessageType.HITL_QUERY,
                    content=(
                        ("\nSyntax Error, Try Again." if syntax_error else "")
                        + f"\nPlease choose one of the options for real-time Risk Assessment and Drift Monitoring\n1. Enter prompt manually\n2. Start streaming prompts from a JSON file.\nYour Choice "
                    ),
                    step_name="Stream Prompt",
                    step_role=Role.SYSTEM,
                    step_kwargs={
                        "input_message_query": "Enter prompt type (1 or 2)",
                        "response_type_needed": "prompt_input",
                    },
                ).model_dump()
            )
        except GraphInterrupt as e:
            raise HumanInterruptionException(json.dumps(e.args[0][0].value))

        try:
            if choice["response"] == "1":
                prompts = [
                    interrupt(
                        WorkflowStepMessage(
                            step_type=MessageType.HITL_QUERY,
                            content="\nEnter your prompt",
                            step_name="Stream Prompt",
                            step_role=Role.SYSTEM,
                            step_kwargs={
                                "input_message_query": "Enter prompt text here",
                                "response_type_needed": "prompt_input",
                            },
                        ).model_dump()
                    )["response"]
                ]
                break
            elif choice["response"] == "2":
                prompt_file = interrupt(
                    WorkflowStepMessage(
                        step_type=MessageType.HITL_QUERY,
                        content="\nEnter JSON file path",
                        step_name="Stream Prompt",
                        step_role=Role.SYSTEM,
                        step_kwargs={
                            "input_message_query": "Enter prompt file path here",
                            "response_type_needed": "prompt_input",
                        },
                    ).model_dump()
                )
                prompts = json.load(Path(prompt_file["response"]).open("r"))
                break
            else:
                syntax_error = True

        except GraphInterrupt as e:
            raise HumanInterruptionException(json.dumps(e.args[0][0].value))
        except:
            syntax_error = True

    global PROMPT_GEN
    PROMPT_GEN[config.get("configurable", {}).get("thread_id", "Client_1")] = (
        (index, prompt) for index, prompt in enumerate(prompts, start=1)
    )
    # global RANDOM_INDICES
    # RANDOM_INDICES = random.choices(range(10), k=4)


# Node
@workflow_step(step_name="Input Prompt", step_role=Role.USER)
def stream_input_prompt(state: StreamAgentState, config: RunnableConfig):
    if state.prompt_index == 1:
        RANDOM_INDICES = [1, 3, 7, 9]  # sorted(random.sample(range(10),4))
        # RISK_METRICS = {}
    else:
        RANDOM_INDICES = [1, 2, 7, 9]  # state.random_indices
        # RISK_METRICS = state.risk_metrics
    return {
        "prompt_index": state.prompt_index,
        "prompt": state.prompt,
        "random_indices": RANDOM_INDICES,
    }


class StreamAgent(Agent):
    """
    Initializes a new instance of the Human in the Loop Agent class.
    """

    _WORKFLOW_NAME = "Stream Agent"
    _WORKFLOW_DESC = f"[bold blue]Stream Input Prompt to the agents:"

    def __init__(self):
        super(StreamAgent, self).__init__(StreamAgentState)

    def _build_graph(self, graph: StateGraph):

        # Add nodes
        graph.add_node("Next Prompt", next_prompt)
        graph.add_node("Load Input Prompts", load_input_prompts)
        graph.add_node("Stream Input Prompt", stream_input_prompt)

        # Add edges to connect nodes
        graph.add_edge(START, "Next Prompt")
        graph.add_conditional_edges(
            source="Next Prompt",
            path=is_next_prompt_available,
            path_map={True: "Stream Input Prompt", False: "Load Input Prompts"},
        )
        graph.add_edge("Load Input Prompts", "Next Prompt")
        graph.add_edge("Stream Input Prompt", END)
