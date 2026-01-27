from functools import partial
from typing import Dict, List, Optional

from ai_atlas_nexus.blocks.inference import InferenceEngine
from ai_atlas_nexus.data import load_resource
from ai_atlas_nexus.library import AIAtlasNexus
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from rich.console import Console

from gaf_guard.core.agents import Agent
from gaf_guard.core.decorators import workflow_step


console = Console()
ai_atlas_nexus = AIAtlasNexus()


# Graph state
class RiskGenerationState(BaseModel):
    user_intent: str
    domain: Optional[str] = None
    environment: Optional[str] = None
    risk_questionnaire_output: Optional[List[Dict[str, str]]] = None
    identified_risks: Optional[List[str]] = None
    identified_ai_tasks: Optional[List[str]] = None


# Node
@workflow_step(step_name="Domain Identification")
def get_usecase_domain(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # domain = ai_atlas_nexus.identify_domain_from_usecases(
    #     [state.user_intent], inference_engine, verbose=False
    # )[0]

    return {"domain": "Writing assistant"}


# Node
@workflow_step(step_name="Questionnaire Prediction")
def generate_zero_shot(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # load CoT data
    risk_questionnaire_cot = (
        config.get("configurable", {})
        .get("RiskGeneratorAgent", {})
        .get("risk_questionnaire_cot", load_resource("risk_questionnaire_cot.json"))
    )

    responses = ai_atlas_nexus.generate_zero_shot_risk_questionnaire_output(
        state.user_intent,
        risk_questionnaire_cot[1:],
        inference_engine,
        verbose=False,
    )

    risk_questionnaire_output = []
    for question_data, response in zip(risk_questionnaire_cot, responses):
        risk_questionnaire_output.append(
            {
                "question": question_data["question"],
                "answer": response.prediction["answer"],
            }
        )

    return {"risk_questionnaire_output": risk_questionnaire_output}


# Node
@workflow_step(
    step_name="Questionnaire Prediction",
    step_desc="Chain of Thought (CoT) data found, using Few-shot method...",
)
def generate_few_shot(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # load CoT data
    # risk_questionnaire_cot = (
    #     config.get("configurable", {})
    #     .get("RiskGeneratorAgent", {})
    #     .get("risk_questionnaire_cot", load_resource("risk_questionnaire_cot.json"))
    # )

    # responses = ai_atlas_nexus.generate_few_shot_risk_questionnaire_output(
    #     state.user_intent,
    #     risk_questionnaire_cot[1:],
    #     inference_engine,
    #     verbose=False,
    # )

    # risk_questionnaire_output = []
    # for question_data, response in zip(risk_questionnaire_cot[1:], responses):
    #     risk_questionnaire_output.append(
    #         {
    #             "question": question_data["question"],
    #             "answer": response.prediction["answer"],
    #         }
    #     )

    return {
        "risk_questionnaire_output": [
            {
                "question": "In which environment is the system used?",
                "answer": "Legal and Policy Research Departments or Government Policy Units",
            },
            {
                "question": "What techniques are utilised in the system? Multi-modal: {Document Question/Answering, Image and text to text, Video and text to text, visual question answering}, Natural language processing: {feature extraction, fill mask, question answering, sentence similarity, summarization, table question answering, text classification, text generation, token classification, translation, zero shot classification}, computer vision: {image classification, image segmentation, text to image, object detection}, audio:{audio classification, audio to audio, text to speech}, tabular: {tabular classification, tabular regression}, reinforcement learning",
                "answer": "Natural language processing: question answering, summarization, and text generation",
            },
            {
                "question": "Who is the intended user of the system?",
                "answer": "The intended user of the system is the government policymakers or legal experts in Ireland responsible for drafting or understanding reports on automated decision-making in the welfare system.",
            },
            {
                "question": "What is the intended purpose of the system?",
                "answer": "To generate a comprehensive government policy report on automated decision-making in Ireland's welfare system, with a specific focus on analyzing how federal courts have addressed automated penalties. The system aims to ensure the report is factually correct and provides a thorough examination of the input documents.",
            },
            {
                "question": "What is the application of the system?",
                "answer": "Natural Language Processing (NLP): Analyze legal documents, court rulings, and policy papers to extract relevant information on automated penalties in Ireland's welfare system. Machine Learning (ML): Identify patterns and trends in the data to provide insights and support evidence-based policy recommendations. Knowledge Graph: Create a structured knowledge base of legal precedents, policy documents, and stakeholder perspectives to ensure a comprehensive understanding of the topic. Fact-checking and validation: Implement AI-driven fact-checking tools to ensure the accuracy and reliability of the information presented in the report.",
            },
            {
                "question": "Who is the subject as per the intent?",
                "answer": "Ireland's welfare system and federal courts' decisions on automated penalties",
            },
        ],
        "environment": "Legal and Policy Research Departments or Government Policy Units",
    }


# Node
def if_cot_examples_found(
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # load CoT data
    risk_questionnaire_cot = (
        config.get("configurable", {})
        .get("RiskGeneratorAgent", {})
        .get("risk_questionnaire_cot", load_resource("risk_questionnaire_cot.json"))
    )

    if all(
        [
            "cot_examples" in question_data and question_data["cot_examples"]
            for question_data in risk_questionnaire_cot
        ]
    ):
        return True
    else:
        return False


# Node
@workflow_step(step_name="Risk Generation")
def identify_risks(
    inference_engine: InferenceEngine,
    taxonomy: str,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # risks = ai_atlas_nexus.identify_risks_from_usecases(
    #     [state.user_intent], inference_engine, taxonomy=taxonomy, zero_shot_only=True
    # )

    return {
        "identified_risks": [
            "Incorrect risk testing",
            "Over- or under-reliance",
            "Confidential data in prompt",
            "Prompt leaking",
            "Harmful output",
            "Lack of model transparency",
            "Data bias",
        ]
    }


# Node
@workflow_step(step_name="AI Tasks")
def identify_ai_tasks(
    inference_engine: InferenceEngine,
    state: RiskGenerationState,
    config: RunnableConfig,
):
    # ai_tasks = ai_atlas_nexus.identify_ai_tasks_from_usecases(
    #     [state.user_intent], inference_engine, verbose=False
    # )[0]

    return {
        "identified_ai_tasks": [
            "Text Classification",
            "Document Question Answering",
            "Text Generation",
        ]
    }


# Node
@workflow_step(step_name="Persisting Results")
def persist_to_memory(state: RiskGenerationState, config: RunnableConfig):
    return {"log": "The data has been saved in Memory."}


class RiskGeneratorAgent(Agent):
    """
    Initializes a new instance of the Questionnaire Agent class.
    """

    _WORKFLOW_NAME = "Risk Generation Agent"
    _WORKFLOW_DESC = (
        f"[bold blue]Gathering information using the following workflow:\n[/bold blue]"
    )

    def __init__(self):
        super(RiskGeneratorAgent, self).__init__(RiskGenerationState)

    def _build_graph(
        self,
        graph: StateGraph,
        inference_engine: InferenceEngine,
        taxonomy: str,
    ):

        # Add nodes
        graph.add_node("Get AI Domain", partial(get_usecase_domain, inference_engine))
        graph.add_node(
            "Zero Shot Risk Questionnaire Output",
            partial(generate_zero_shot, inference_engine),
        )
        graph.add_node(
            "Few Shot Risk Questionnaire Output",
            partial(generate_few_shot, inference_engine),
        )
        graph.add_node(
            "Identify AI Risks", partial(identify_risks, inference_engine, taxonomy)
        )
        graph.add_node(
            "Identify AI Tasks", partial(identify_ai_tasks, inference_engine)
        )
        graph.add_node("Persist To Memory", persist_to_memory)

        # Add edges to connect nodes
        graph.add_edge(START, "Get AI Domain")
        graph.add_conditional_edges(
            source="Get AI Domain",
            path=if_cot_examples_found,
            path_map={
                True: "Few Shot Risk Questionnaire Output",
                False: "Zero Shot Risk Questionnaire Output",
            },
        )
        graph.add_edge("Few Shot Risk Questionnaire Output", "Identify AI Risks")
        graph.add_edge("Zero Shot Risk Questionnaire Output", "Identify AI Risks")
        graph.add_edge("Identify AI Risks", "Identify AI Tasks")
        graph.add_edge("Identify AI Tasks", "Persist To Memory")
        graph.add_edge("Persist To Memory", END)
