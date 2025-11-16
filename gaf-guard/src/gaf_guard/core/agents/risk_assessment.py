import re
from functools import partial
from typing import Dict, List, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from rich.console import Console
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.library import RiskAtlasNexus
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow_step


console = Console()
risk_atlas_nexus = RiskAtlasNexus()


def parse_model_assessment(response):
    assessment_match = re.findall(r"<score>(.*?)</score>", response, re.DOTALL)
    return assessment_match[-1].strip().title() if assessment_match else None


# Graph state
class RiskAssessmentState(BaseModel):
    prompt: str
    identified_risks: List[str]
    risk_report: Optional[Dict] = None


# Node
@workflow_step(step_name="Risk Assessment")
def assess_risk(
    inference_engine: InferenceEngine,
    taxonomy: str,
    state: RiskAssessmentState,
    config: RunnableConfig,
):
    # Gather risk info for the given taxonomy
    risks: List[Risk] = list(
        filter(
            lambda risk: risk.name in state.identified_risks,
            risk_atlas_nexus.get_all_risks(taxonomy=taxonomy),
        )
    )

    # Prepare messages for prompting
    messages = [
        [
            {
                "role": "system",
                "content": risk.description + " " + risk.concern,
            },
            {"role": "user", "content": state.prompt},
        ]
        for risk in risks
    ]

    # Invoke inference service
    responses = inference_engine.chat(
        messages=messages,
        verbose=False,
    )

    return {
        "risk_report": {
            risk.name: parse_model_assessment(response.prediction)
            for risk, response in zip(risks, responses)
        }
    }


# Node
@workflow_step(step_name="Incident Reporting")
def aggregate_and_report_incident(state: RiskAssessmentState, config: RunnableConfig):
    risk_report_yes = None
    if state.risk_report:
        risk_report_yes = dict(
            filter(lambda item: "Yes" in item[1], state.risk_report.items())
        )

    if risk_report_yes:
        return {
            "incident_message": f"Potential risks identified.\n{list(risk_report_yes.keys())}"
        }
    else:
        return {"incident_message": "No risks identified with the prompts."}


class RisksAssessmentAgent(Agent):
    """
    Initializes a new instance of the Granite Guardian Risk Detector Agent class.
    """

    _WORKFLOW_NAME = "Risk Asssessment Agent"
    _WORKFLOW_DESC = (
        f"[bold blue]Real-time risk monitoring using the following workflow:"
    )

    def __init__(self):
        super(RisksAssessmentAgent, self).__init__(RiskAssessmentState)

    def _build_graph(
        self,
        graph: StateGraph,
        inference_engine: InferenceEngine,
        taxonomy: str,
    ):

        # Add nodes
        graph.add_node("Assess Risk", partial(assess_risk, inference_engine, taxonomy))
        graph.add_node(
            "Aggregate and Report Risk Incidents", aggregate_and_report_incident
        )

        # Add edges
        graph.add_edge(START, "Assess Risk")
        graph.add_edge("Assess Risk", "Aggregate and Report Risk Incidents")
        graph.add_edge("Aggregate and Report Risk Incidents", END)
