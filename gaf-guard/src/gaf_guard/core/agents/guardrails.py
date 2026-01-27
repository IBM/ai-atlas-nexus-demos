import operator
import re
from functools import partial
from typing import Annotated, Any, Dict, List, Optional

from ai_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from ai_atlas_nexus.blocks.inference import InferenceEngine
from ai_atlas_nexus.library import AIAtlasNexus
from deepeval.models import OllamaModel
from deepteam.guardrails.guards.hallucination_guard import HallucinationGuard
from deepteam.guardrails.guards.toxicity_guard import ToxicityGuard
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from rich.console import Console

from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow_step


console = Console()
ai_atlas_nexus = AIAtlasNexus()


def parse_model_assessment(response):
    assessment_match = re.findall(r"<score>(.*?)</score>", response, re.DOTALL)
    return assessment_match[-1].strip().title() if assessment_match else None


# Graph state
class GuardrailState(BaseModel):
    prompt: str
    identified_risks: List[str]
    dynamic_identified_risks: Optional[List[Dict[str, Any]]] = None
    guardrails_report: Optional[List[Dict[str, str]]] = None
    prompt_index: int
    transition_risks: Optional[List[str]] = []


# Node
@workflow_step(step_name="Guardrail Assessment")
def assess_guardrail(
    inference_engine: InferenceEngine,
    taxonomy: str,
    state: GuardrailState,
    config: RunnableConfig,
):
    # guardrails = state.guardrails
    # guardrails = [{}]
    dynamic_risks_high = []
    dynamic_risks_transition = []
    dynamic_risks_low = []
    guardrails_report = []
    # dynamic_updated_risks = json.loads('[{"risk_name":"Toxic output", "priority": "low", "threshold": 0.2}, {"risk_name":"Hallucination", "priority": "high", "threshold": 0.01}]')

    for risk in state.dynamic_identified_risks:
        if risk["priority"] == "high":
            dynamic_risks_high.append(risk["risk_name"])
        else:
            # if risk not in state.transition_risks:
            dynamic_risks_low.append(risk["risk_name"])

    dynamic_risks_transition = state.transition_risks

    for risk in dynamic_risks_transition:
        dynamic_risks_low.pop(dynamic_risks_low.index(risk))

    model = OllamaModel(
        model="llama3.2", base_url="http://localhost:11434", temperature=0
    )

    # toxicity_guard = ToxicityGuard(model=model)
    # hallucination_guard = HallucinationGuard(model=model)

    guardrails_map = {
        "Toxic output": ToxicityGuard,
        "Hallucination": HallucinationGuard,
    }

    guardrails = []

    for risk in dynamic_risks_high:
        guardrails.append(guardrails_map[risk](model=model))

    for risk in dynamic_risks_transition:
        guardrails.append(guardrails_map[risk](model=model))

    high_and_transition_risks = dynamic_risks_high + dynamic_risks_transition

    for index, guardrail in enumerate(guardrails):
        output = guardrail.guard_output(input=" ", output=state.prompt)
        if output == "unsafe":
            guardrails_report.append(
                {
                    "Risk": high_and_transition_risks[index],
                    "old_msg": state.prompt,
                    "new_msg": "Output cannot be provided in this context",
                }
            )
            # guardrails_report[index]["old_message"] = state.prompt
            # guardrails_report[index]["new_message"] = "Output cannot be provided in this context"

    # toxic_output = toxicity_guard.guard_output(input=" ", output=state.prompt)
    # hallucination_output = hallucination_guard.guard_output(input=" ", output=state.prompt)

    # if toxic_output=="unsafe":

    return {"guardrails_report": guardrails_report}


class GuardrailsAgent(Agent):
    """
    Initializes a new instance of the Granite Guardian Risk Detector Agent class.
    """

    _WORKFLOW_NAME = "Guardrail Asssessment Agent"
    _WORKFLOW_DESC = (
        f"[bold blue]Real-time risk monitoring using the following workflow:"
    )

    def __init__(self):
        super(GuardrailsAgent, self).__init__(GuardrailState)

    def _build_graph(
        self,
        graph: StateGraph,
        inference_engine: InferenceEngine,
        taxonomy: str,
    ):

        # Add nodes
        graph.add_node(
            "Assess Guardrails", partial(assess_guardrail, inference_engine, taxonomy)
        )

        # Add edges
        graph.add_edge(START, "Assess Guardrails")
        graph.add_edge("Assess Guardrails", END)
