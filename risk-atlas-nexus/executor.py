import os
import pandas as pd
import gradio as gr
import datetime
from pathlib import Path
import json

from ai_atlas_nexus.blocks.inference import WMLInferenceEngine
from ai_atlas_nexus.blocks.inference.params import WMLInferenceEngineParams
from ai_atlas_nexus.library import AIAtlasNexus

from functools import lru_cache, wraps
from dotenv import load_dotenv

load_dotenv(override=True)

# Load the taxonomies
ran = AIAtlasNexus() # type: ignore


def clear_previous_risks():
    return gr.Markdown("""<h2> Potential Risks </h2> """), [], gr.Dataset(samples=[], 
                                     sample_labels=[], 
                                     samples_per_page=50, visible=False), gr.DownloadButton("Download JSON", visible=False, ), "", gr.Dataset(samples=[], sample_labels=[], visible=False), gr.DataFrame([], wrap=True, show_copy_button=True, show_search="search", visible=False), gr.DataFrame([], wrap=True, show_copy_button=True, show_search="search", visible=False), gr.DataFrame([], wrap=True, show_copy_button=True, show_search="search", visible=False), gr.Markdown(" "), gr.Markdown(" "), 

def clear_previous_mitigations():
    return "", gr.Dataset(samples=[], sample_labels=[], visible=False), gr.DataFrame([], wrap=True, show_copy_button=True, show_search="search", visible=False), gr.DataFrame([], wrap=True, show_copy_button=True, show_search="search", visible=False), gr.DataFrame([], wrap=True, show_copy_button=True, show_search="search", visible=False),  gr.Markdown(" "), gr.Markdown(" ")

def generate_subgraph(usecase, risk):
    lines =[f'```mermaid\n', '---\n'
'config:\n'
'  theme: mc\n'
'  layout: dagre\n'
'  look: classic\n'
'---\n'
'flowchart TB\n']
    
    lines.append(f'uc_173@{{ label: "{usecase}" }} -- subClassOf --> AISystem["AISystem"]\n')
    lines.append(f'uc_173 -- hasRisk --> Risk2["{risk.name}"]\n')
    lines.append(f'Risk2 -- isPartOf --> {risk.isPartOf}\n')
    lines.append(f'Risk2 -- isDefinedByTaxonomy --> {risk.isDefinedByTaxonomy}\n')
    
    # add related risks
    rrs = ran.get_related_risks(id=risk.id)
    if len(rrs) > 0:
        r_risks = ', '.join(rr.name for rr in rrs)
        lines.append(f'Risk2 -- hasRelatedRisks --> Risk3["{r_risks}"]\n')

    # add related evals
    revals = ran.get_related_evaluations(risk_id=risk.id)
    if len(revals) > 0:
        r_evals = ', '.join(reval.name for reval in revals)
        lines.append(f'Risk2 -- hasAiEvaluations --> Risk4["{r_evals[:100]}"] \n')
    
    # add related mitigations
    rmits = get_controls_and_actions(risk.id, risk.isDefinedByTaxonomy)
    if len(rmits) > 0:
        r_mits = ', '.join(rmits)
        lines.append(f'Risk2 -- hasMitigations --> Risk5["{r_mits[:100]}"] \n')

    lines.append(f"```")
    diagram_string = "".join(lines)
    return gr.Markdown(value = diagram_string)


def custom_lru_cache(maxsize=128, exclude_values=(None,[],[[]])):
    """
    Make the LRU cache not cache result when empty result was returned
    """
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize)(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = cached_func(*args, **kwargs)
            # check for empty df of risks
            if result[2].constructor_args["samples"] in exclude_values:
                return func(*args, **kwargs)
            return result
        
        return wrapper
    return decorator


@custom_lru_cache(exclude_values=(None, []))
def risk_identifier(usecase: str, 
                    model_name_or_path: str = "meta-llama/llama-3-3-70b-instruct", 
                    taxonomy: str = "ibm-risk-atlas"): # -> List[Dict[str, Any]]: #pd.DataFrame:

    downloadable = False
    inference_engine = WMLInferenceEngine(
        model_name_or_path= model_name_or_path,
        credentials={
            "api_key": os.environ["WML_API_KEY"],
            "api_url": os.environ["WML_API_URL"],
            "project_id": os.environ["WML_PROJECT_ID"],
        },
        parameters=WMLInferenceEngineParams(
            max_new_tokens=1000, decoding_method="greedy", repetition_penalty=1
        ),  # type: ignore
    )

    risks_a = ran.identify_risks_from_usecases(# type: ignore
        usecases=[usecase],
        inference_engine=inference_engine,
        taxonomy=taxonomy,
        zero_shot_only=True,
        max_risk=5
    )

    risks = risks_a[0]
    

    sample_labels = [r.name if r else r.id for r in risks]

    out_sec = gr.Markdown("""<h2> Potential Risks </h2> """)

    # write out a JSON
    data = {'time': str(datetime.datetime.now(datetime.timezone.utc)),
                'intent': usecase,
                'model': model_name_or_path,
                'taxonomy': taxonomy,
                'risks': [json.loads(r.json()) for r in risks]
        }
    file_path = Path("static/download.json")
    with open(file_path, mode='w') as f:
        f.write(json.dumps(data, indent=4))
        downloadable = True

    # return out_df
    return out_sec, gr.State(risks), gr.Dataset(samples=[r.id for r in risks], 
                                     sample_labels=sample_labels, 
                                     samples_per_page=50, visible=True, label="Estimated by an LLM."), gr.DownloadButton("Download JSON", "static/download.json", visible=(downloadable and len(risks) > 0))
    

def _get_control_action_intrinsics(riskid, taxonomy, related_risk_ids, related_reqs):
    """Helper function to get actions, controls, and intrinsics."""
    actions = []
    controls = []
    intrinsics = []
    control_activity_recommendations = []

    if related_reqs:
        control_activity_recommendation_ids = [item for sublist in [req.hasRule for req in related_reqs] for item in sublist]
        control_activity_recommendations = [item for item in ran.get_all("ControlActivityRecommendations") if item.id in control_activity_recommendation_ids ]
    
    if taxonomy == "ibm-risk-atlas":
        # look for actions associated with related risks
        if related_risk_ids:
            for i in related_risk_ids:
                rai = ran.get_related_actions(id=i)
                if rai:
                    actions += rai

                rac = ran.get_related_risk_controls(id=i)
                if rac:
                    controls += rac

                ran_intrinsics = ran.get_related_intrinsics(risk_id=i)
                if ran_intrinsics:
                    intrinsics += ran_intrinsics
    else:
        # Use only actions related to primary risks
        actions = ran.get_related_actions(id=riskid)
        controls = ran.get_related_risk_controls(id=riskid)
        intrinsics = ran.get_related_intrinsics(risk_id=riskid)

    return actions, controls, intrinsics, control_activity_recommendations


def get_controls_and_actions(riskid, taxonomy):
    related_risk_ids = [r.id for r in ran.get_related_risks(id=riskid)]
    actions, controls, intrinsics, control_activity_recommendations = _get_control_action_intrinsics(riskid, taxonomy, related_risk_ids, None)
    return [i.name for i in actions] + [i.name for i in controls] + [i.name for i in intrinsics] + [i.name for i in control_activity_recommendations] #type: ignore


@lru_cache
def mitigations(usecase: str, riskid: str, taxonomy: str) -> tuple[gr.Markdown, gr.Dataset, gr.DataFrame, gr.DataFrame, gr.DataFrame, gr.Markdown, gr.Markdown]:
    """
    For a specific risk (riskid), returns
    (a) a risk description
    (b) related risks - as a dataset
    (c) mitigations
    (d) related AI evaluations
    (e) related AI requirements
    (f) A subgraph of risk to mitigations

    """
    try:
        selected_risk = ran.get_risk(id=riskid)
        risk_desc = selected_risk.description # type: ignore
        risk_sec = f"<h3>Description: </h3> {risk_desc}"
    except AttributeError:
        risk_sec = ""

    related_risks = ran.get_related_risks(id=riskid)
    related_risk_ids = [r.id for r in related_risks]
    related_ai_evals = ran.get_related_evaluations(risk_id=riskid)
    related_ai_eval_ids = [ai_eval.id for ai_eval in related_ai_evals ]
    related_reqs = ran.query("requirement", close_mappings=riskid) + ran.query("requirement", related_mappings=riskid)
    related_reqs_ids = [req.id for req in related_reqs]

    actions, controls, intrinsics, control_activity_recommendations = _get_control_action_intrinsics(riskid, taxonomy, related_risk_ids, related_reqs)
        

    # Sanitize outputs
    if not related_risk_ids:
        label = "No related risks found."
        samples = None
        sample_labels = None
    else:
        label = f"Risks from other taxonomies related to {riskid}"
        samples = related_risk_ids
        sample_labels = [i.name for i in related_risks] #type: ignore

    if not actions and not controls and not intrinsics and not control_activity_recommendations:
        alabel = "No mitigations found."
        asamples = None
        asample_labels = None
        mitdf = pd.DataFrame()
    else:
        alabel = f"Mitigation actions and controls related to risk {riskid}."
        asample_name = [i.name for i in actions] + [i.name for i in controls] + [i.name for i in intrinsics] + [i.name for i in control_activity_recommendations] #type: ignore
        asample_labels = [i.description for i in actions] + [i.name for i in controls] + [i.description for i in intrinsics] + [i.description for i in control_activity_recommendations] #type: ignore
        asample_tax = [i.isDefinedByTaxonomy for i in actions] + [i.isDefinedByTaxonomy for i in controls] + [i.isDefinedByTaxonomy for i in intrinsics] + [i.isDefinedByTaxonomy for i in control_activity_recommendations] #type: ignore
        asample_types = ["Action"] * len(actions) + ["Control"] * len(controls) + ["Intrinsic"] * len(intrinsics) + ["ControlActivity"] * len(control_activity_recommendations)
        mitdf = pd.DataFrame({"Type": asample_types, "Mitigation": asample_name, "Taxonomy": asample_tax, "Description": asample_labels})
    
    if not related_ai_eval_ids:
        blabel = "No related AI evaluations found."
        bsample_labels = None
        aievalsdf = pd.DataFrame()
    else:
        blabel = f"AI Evaluations related to {riskid}"
        bsample_labels = [ai_eval.description for ai_eval in related_ai_evals] # type: ignore
        bsample_name = [ai_eval.name for ai_eval in related_ai_evals] #type: ignore
        aievalsdf = pd.DataFrame({"AI Evaluation": bsample_name, "Description": bsample_labels})

    if not related_reqs_ids:
        clabel = "No related requirements found."
        csample_labels = None
        reqsdf = pd.DataFrame()
    else:
        clabel = f"Documented requirements related to {riskid}"
        csample_labels = [req.description for req in related_reqs] # type: ignore
        csample_name = [req.name for req in related_reqs] #type: ignore
        csample_tax = [req.isDefinedByTaxonomy for req in related_reqs] #type: ignore
        reqsdf = pd.DataFrame({"Requirement": csample_name, "Taxonomy": csample_tax, "Description": csample_labels})

    
    status = gr.Markdown(" ") if len(mitdf) > 0 else gr.Markdown("No mitigations found.")

    fig = gr.Markdown(" ") if not selected_risk else generate_subgraph(usecase, selected_risk)

    return (gr.Markdown(risk_sec), 
            gr.Dataset(samples=samples, label=label, sample_labels=sample_labels, visible=True),
            gr.DataFrame(mitdf, wrap=True, show_copy_button=True, show_search="search", label=alabel, visible=(len(actions + controls + intrinsics) > 0)),
            gr.DataFrame(aievalsdf, wrap=True, show_copy_button=True, show_search="search", label=blabel, visible=(len(related_ai_eval_ids) > 0)),
            gr.DataFrame(reqsdf, wrap=True, show_copy_button=True, show_search="search", label=clabel, visible=(len(related_reqs_ids) > 0)),
            status, fig) 

