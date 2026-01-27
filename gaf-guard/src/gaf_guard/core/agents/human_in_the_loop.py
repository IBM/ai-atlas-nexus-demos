import json
import operator
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt
from pydantic import BaseModel
from rich.console import Console

from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow_step
from gaf_guard.core.models import WorkflowStepMessage
from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.exceptions import HumanInterruptionException


console = Console()

# class DynamicRisk(BaseModel):
#     risk_name: str


# Graph state
class HumanInTheLoopAgentState(BaseModel):
    identified_risks: Optional[List[str]] = None
    dynamic_identified_risks: Optional[List[Dict[str, Any]]] = None


# Node
@workflow_step(step_name="Gather AI Risks for Human Intervention")
def gather_ai_risks(state: HumanInTheLoopAgentState, config: RunnableConfig):
    return {"identified_risks": state.identified_risks}


# Node
def get_human_response(state: HumanInTheLoopAgentState, config: RunnableConfig):
    syntax_error = False
    while True:
        try:
            dynamic_updated_risks = interrupt(
                WorkflowStepMessage(
                    step_type=MessageType.HITL_QUERY,
                    content=(
                        ("\nSyntax Error, Try Again." if syntax_error else "")
                        + f"\nPlease add Risks using 'Add Dynamic Risks' button or Type Risks as a python List of dictionaries with keys risk_name, priority, threshold."
                    ),
                    step_name="Human Intervention",
                    step_role=Role.AGENT,
                    step_kwargs={
                        "input_message_query": "Enter dynamic risks here",
                        "response_type_needed": "dynamic_risks",
                    },
                ).model_dump()
            )
        except GraphInterrupt as e:
            raise HumanInterruptionException(json.dumps(e.args[0][0].value))

        try:
            if len(dynamic_updated_risks["response"]) > 0:
                dynamic_updated_risks = json.loads(dynamic_updated_risks["response"])
            else:
                dynamic_updated_risks = json.loads(
                    '[{"risk_name":"Toxic output", "priority": "low", "threshold": 0.2}, {"risk_name":"Hallucination", "priority": "high", "threshold": 0.01}]'
                )
            break
        except Exception as e:
            syntax_error = True

    return {"dynamic_identified_risks": dynamic_updated_risks}


# Node
@workflow_step(step_name="Updated AI Risks from Human Response", step_role=Role.USER)
def updated_ai_risks(state: HumanInTheLoopAgentState, config: RunnableConfig):
    return {"dynamic_identified_risks": state.dynamic_identified_risks}


class HumanInTheLoopAgent(Agent):
    """
    Initializes a new instance of the Human in the Loop Agent class.
    """

    _WORKFLOW_NAME = "Human In the Loop Agent"
    _WORKFLOW_DESC = f"[bold blue]Getting Response from the User:"

    def __init__(self):
        super(HumanInTheLoopAgent, self).__init__(HumanInTheLoopAgentState)

    def _build_graph(self, graph: StateGraph):

        # Add nodes
        graph.add_node("Gather AI Risks", gather_ai_risks)
        graph.add_node("Get Human Response on AI Risks", get_human_response)
        graph.add_node("Updated AI Risks", updated_ai_risks)

        # Add edges to connect nodes
        graph.add_edge(START, "Gather AI Risks")
        graph.add_edge("Gather AI Risks", "Get Human Response on AI Risks")
        graph.add_edge("Get Human Response on AI Risks", "Updated AI Risks")
        graph.add_edge("Updated AI Risks", END)
