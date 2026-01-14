import re
from functools import partial
from typing import Dict, List, Optional, Annotated, Any

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from rich.console import Console
from ai_atlas_nexus.blocks.inference import InferenceEngine
from ai_atlas_nexus.library import AIAtlasNexus
from ai_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow_step
import operator


console = Console()
ai_atlas_nexus = AIAtlasNexus()


def parse_model_assessment(response):
    assessment_match = re.findall(r"<score>(.*?)</score>", response, re.DOTALL)
    return assessment_match[-1].strip().title() if assessment_match else None

# class DynamicRisk(BaseModel):
#     risk_name: str

# Graph state
class DynamicRiskAssessmentState(BaseModel):
    prompt: str
    identified_risks: List[str]
    dynamic_identified_risks: Optional[List[Dict[str, Any]]] = None
    prompt_index: int 
    random_indices: List[int]
    risk_report: Optional[Dict] = None
    high_risk_report: Optional[Dict] = None
    low_risk_report: Optional[Dict] = None
    window_size: int = 4
    rolling_values: Optional[Dict[str, List[int]]] = {}
    transition_risk_report: Optional[Dict] = None
    transition_risks: Optional[List[str]] = []

# Node
@workflow_step(step_name="Dynamic Risk Assessment")
def assess_risk(
    inference_engine: InferenceEngine,
    taxonomy: str,
    state: DynamicRiskAssessmentState,
    config: RunnableConfig,
):
    dynamic_risks_high = []
    dynamic_risks_transition = []
    dynamic_risks_low = []
    
    responses_low = None
    risks_low = None
    transition_risk_report  = {}
    
    # risk_metrics = state.risk_metrics

    for risk in state.dynamic_identified_risks:
        if risk["priority"]=="high":
            dynamic_risks_high.append(risk["risk_name"])
        else:
            # if risk not in state.transition_risks: 
            dynamic_risks_low.append(risk["risk_name"])
    
    dynamic_risks_transition = state.transition_risks

    for risk in dynamic_risks_transition:
        dynamic_risks_low.pop(dynamic_risks_low.index(risk))

    rolling_values = state.rolling_values
    # dynamic_risks = [risk["risk_name"] if risk["priority"]=="high" ]

    # Gather risk info for the given taxonomy
    risks_high: List[Risk] = list(
        filter(
            # lambda risk: risk.name in state.identified_risks,
            lambda risk: risk.name in dynamic_risks_high,
            ai_atlas_nexus.get_all_risks(taxonomy=taxonomy),
        )
    )

    # Prepare messages for prompting
    messages_high = [
        [
            {
                "role": "system",
                "content": risk.description + " " + risk.concern,
            },
            {"role": "user", "content": state.prompt},
        ]
        for risk in risks_high
    ]

    # Invoke inference service
    responses_high = inference_engine.chat(
        messages=messages_high,
        verbose=False,
    )


    # Transition risks
    if len(dynamic_risks_transition) > 0:
        risks_transition: List[Risk] = list(
            filter(
                # lambda risk: risk.name in state.identified_risks,
                lambda risk: risk.name in dynamic_risks_transition,
                ai_atlas_nexus.get_all_risks(taxonomy=taxonomy),
            )
        )

        # Prepare messages for prompting
        messages_transition = [
            [
                {
                    "role": "system",
                    "content": risk.description + " " + risk.concern,
                },
                {"role": "user", "content": state.prompt},
            ]
            for risk in risks_transition
        ]

        # Invoke inference service
        responses_transition = inference_engine.chat(
            messages=messages_transition,
            verbose=False,
        )
        transition_risk_report = {
            risk.name: parse_model_assessment(response.prediction)
            for risk, response in zip(risks_transition, responses_transition)
        }

        transition_risk_report_yes = dict(
            filter(lambda item: "Yes" in item[1], transition_risk_report.items())
        )

        # if len(state.rolling_values) == 0:
        #     for risk in dynamic_risks_low:
        #         # risk_metrics[risk] =  0  
        #         rolling_values[risk] = [0]*state.window_size   

        for risk in dynamic_risks_transition:
            if risk in transition_risk_report_yes.keys():
                rolling_values[risk].append(1)
            else:
                rolling_values[risk].append(0)
            if len(rolling_values[risk]) > state.window_size:
                rolling_values[risk].pop(0) # Pop from the front
                current_average = sum(rolling_values[risk]) / state.window_size
                # Need to correct this
                if current_average < 0.2: # state.dynamic_identified_risks[risk]["threshold"]:
                    dynamic_risks_low.append(risk)
                    dynamic_risks_transition.pop(dynamic_risks_transition.index(risk))

    # Low risks
    if state.prompt_index in state.random_indices:
        # Gather risk info for the given taxonomy
        risks_low: List[Risk] = list(
            filter(
                # lambda risk: risk.name in state.identified_risks,
                lambda risk: risk.name in dynamic_risks_low,
                ai_atlas_nexus.get_all_risks(taxonomy=taxonomy),
            )
        )

        # Prepare messages for prompting
        messages_low = [
            [
                {
                    "role": "system",
                    "content": risk.description + " " + risk.concern,
                },
                {"role": "user", "content": state.prompt},
            ]
            for risk in risks_low
        ]

        # Invoke inference service
        responses_low = inference_engine.chat(
            messages=messages_low,
            verbose=False,
        )

        low_risk_report = {
            risk.name: parse_model_assessment(response.prediction)
            for risk, response in zip(risks_low, responses_low)
        }
        

        if len(state.rolling_values) == 0:
            for risk in dynamic_risks_low:
                # risk_metrics[risk] =  0  
                rolling_values[risk] = [0]*state.window_size                  

        low_risk_report_yes = dict(
            filter(lambda item: "Yes" in item[1], low_risk_report.items())
        )


        for risk in dynamic_risks_low:
            if risk in low_risk_report_yes.keys():
                rolling_values[risk].append(1)
            else:
                rolling_values[risk].append(0)
            if len(rolling_values[risk]) > state.window_size:
                rolling_values[risk].pop(0) # Pop from the front
                current_average = sum(rolling_values[risk]) / state.window_size

                # Need to correct this
                if current_average > 0.2: # state.dynamic_identified_risks[risk]["threshold"]:
                    dynamic_risks_transition.append(risk)
                    dynamic_risks_low.pop(dynamic_risks_low.index(risk))
                    
        
        dynamic_risks_transition = list(set(dynamic_risks_transition))
        
        return {
        "high_risk_report": {
            risk.name: parse_model_assessment(response.prediction)
            for risk, response in zip(risks_high, responses_high)
        },
        "low_risk_report": {
            risk.name: parse_model_assessment(response.prediction)
            for risk, response in zip(risks_low, responses_low)
        },
        "transition_risk_report" : transition_risk_report,
        "transition_risks": dynamic_risks_transition,
        "rolling_values": rolling_values,

        }
    
    else:
        return {
            "high_risk_report": {
                risk.name: parse_model_assessment(response.prediction)
                for risk, response in zip(risks_high, responses_high)
            },
            "transition_risk_report" : transition_risk_report,
            "transition_risks": dynamic_risks_transition,
            "rolling_values": rolling_values
        }


# Node
@workflow_step(step_name="Incident Reporting")
def aggregate_and_report_incident(state: DynamicRiskAssessmentState, config: RunnableConfig):
    risk_report_high_yes = None
    risk_report_low_yes = None

    if state.high_risk_report:
        risk_report_high_yes = dict(
            filter(lambda item: "Yes" in item[1], state.high_risk_report.items())
        )

    if state.low_risk_report:
        risk_report_low_yes = dict(
            filter(lambda item: "Yes" in item[1], state.low_risk_report.items())
        )

    if risk_report_high_yes:
            return {
                "incident_message": f"Potential high risks identified.\n{list(risk_report_high_yes.keys())}"
            }
    else:
        return {"incident_message": "No risks identified with the prompts."}


class DynamicRisksAssessmentAgent(Agent):
    """
    Initializes a new instance of the Granite Guardian Risk Detector Agent class.
    """

    _WORKFLOW_NAME = "Dynamic Risk Asssessment Agent"
    _WORKFLOW_DESC = (
        f"[bold blue]Real-time risk monitoring using the following workflow:"
    )

    def __init__(self):
        super(DynamicRisksAssessmentAgent, self).__init__(DynamicRiskAssessmentState)

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