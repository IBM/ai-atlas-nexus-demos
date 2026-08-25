import os
import gradio as gr
import logging
from executor import clear_previous_risks, clear_previous_mitigations, risk_identifier, mitigations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UI:

    def __init__(self):
        """Load any static assets """
        self.load_css()

    def header_block(self):
        """Title/description"""
 
        with open("static/text/header.md", "r") as f:
            content = f.read()

        gr.Markdown(content)
        gr.Markdown("---")
        gr.Markdown("<br>")

    def risk_extraction(self):

        with open("static/text/sec1.md") as f:
            content = f.read()

        with gr.Row():
            gr.Markdown(content)

        with gr.Row():
            with gr.Column(variant="compact", scale=1):
                self.usecase = gr.TextArea(
                    label="Intent",
                    interactive=True,
                    info="Describe the intent of the application, or choose from one of the examples below."
                )
                self.taxonomy = gr.Dropdown(
                    choices=["ibm-risk-atlas", "mit-ai-risk-repository", "nist-ai-rmf", "ibm-granite-guardian", 
                             "ailuminate-v1.0", "credo-ucf", "owasp-llm-2.0"],
                    multiselect=False,
                    value="ibm-risk-atlas",
                    label="Choose a risk taxonomy.",
                    info="The risk taxonomy defines a wide range of risks, their classifications, and potential mitigations.",
                    interactive=True,
                )
                self.model_name_or_path = gr.Dropdown(
                    #choices=['codellama/codellama-34b-instruct-hf', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'google/flan-ul2', 'ibm/granite-13b-instruct-v2', 'ibm/granite-3-3-8b-instruct', 'ibm/granite-20b-multilingual', 'ibm/granite-3-2-8b-instruct-preview-rc', 'ibm/granite-3-2b-instruct', 'ibm/granite-3-8b-instruct', 'ibm/granite-34b-code-instruct', 'ibm/granite-3b-code-instruct', 'ibm/granite-8b-code-instruct', 'ibm/granite-guardian-3-2b', 'ibm/granite-guardian-3-8b', 'meta-llama/llama-2-13b-chat', 'meta-llama/llama-3-1-70b-instruct', 'meta-llama/llama-3-1-8b-instruct', 'meta-llama/llama-3-2-11b-vision-instruct', 'meta-llama/llama-3-2-1b-instruct', 'meta-llama/llama-3-2-3b-instruct', 'meta-llama/llama-3-2-90b-vision-instruct', 'meta-llama/llama-3-3-70b-instruct', 'meta-llama/llama-3-405b-instruct', 'meta-llama/llama-guard-3-11b-vision', 'mistralai/mistral-large', 'mistralai/mixtral-8x7b-instruct-v01'],
                    choices=["meta-llama/llama-3-3-70b-instruct", "ibm/granite-4-h-small"],
                    value="meta-llama/llama-3-3-70b-instruct",
                    multiselect=False,
                    label="Choose language model to use",
                    info="Language model used to assess risks (This is not the model being assessed).",
                    interactive=True
                )
                examples = gr.Examples([["A medical chatbot for a triage system to assess symptoms and provide advice based on patient medical history and conditions. The chatbot will analyze the patient input, identify potential medical issues, and offer recommendations to the patient or healthcare provider.", "ibm-risk-atlas"],
                                        ["Building a customer support agent that automatically triages common problems with a company's services.", "ibm-risk-atlas"]],
                                       [self.usecase, self.taxonomy],
                                        label='Example use cases', example_labels=["Medical chatbot", "Customer service agent"] 
                                       )
                self.risk_execute = gr.Button("Submit")
        
            with gr.Column(scale=2):
            
                self.assessment_sec = gr.Markdown()
                self.assessed_risks = gr.Dataset(elem_classes="risks", label=None, visible=False)
                self.assessed_risk_definition = gr.Markdown()

                if len(self.assessed_risks.elem_classes ) > 0:
                    gr.Markdown(
                    """<h2> Related Risks </h2> 
                    Select a potential risk above to check for related risks.
                    """
                    )
                    rrtb = gr.Markdown()
                    self.relatedrisks = gr.Dataset(elem_classes="related-risks", components=[rrtb], label=None, visible=False)

                    gr.Markdown(
                    """<h2>Related Requirements </h2> 
                    Select a potential risk to see its related requirements. """
                    )
                    self.requirements = gr.DataFrame(label=None, visible=False)
                    
                    gr.Markdown(
                    """<h2> Mitigations </h2> 
                    Select a potential risk to determine possible mitigations. """
                    )
                    self.mitigations_text = gr.Markdown()
                    self.mitigations = gr.DataFrame(label=None, visible=False)
                
                    gr.Markdown(
                    """<h2>Benchmarks </h2> 
                    Select a potential risk to determine possible AI evaluations. """
                    )
                    self.benchmarks_text = gr.Markdown()
                    self.benchmarks = gr.DataFrame(label=None, visible=False)
                    
                    gr.Markdown(
                    """<h2>Network </h2> 
                    Select a potential risk to see its network. """
                    )
                    self.networks = gr.Markdown()

                 


                    self.download = gr.DownloadButton("Download JSON", visible=False)

        gr.Markdown("---")
        gr.Markdown("<br>")

    def load_css(self):
        with open("static/style.css", "r") as file: 
            self.css = file.read()

    def layout(self):
        """Assemble the overall layout""" 

        with gr.Blocks(theme=gr.themes.Default()) as demo: # type: ignore

            self.header_block()
            self.risks = gr.State()
            
            # Risk assessment based on user intents
            self.risk_extraction()

            # Register event listener
            self.risk_execute.click(
                fn=clear_previous_risks,
                inputs=[],
                outputs=[self.assessment_sec, self.risks, self.assessed_risks, self.download, self.assessed_risk_definition, self.relatedrisks, self.mitigations, self.benchmarks, self.requirements, self.mitigations_text, self.networks],
            ).then(
                fn=risk_identifier,
                inputs=[
                    self.usecase,
                    self.model_name_or_path,
                    self.taxonomy
                ],
                outputs=[self.assessment_sec, self.risks, self.assessed_risks, self.download],
                api_name="risk_identifier"
            )

            self.assessed_risks.select(
                fn=clear_previous_mitigations,
                inputs=[],
                outputs=[self.assessed_risk_definition, self.relatedrisks, self.mitigations, self.benchmarks, self.requirements, self.mitigations_text, self.networks]
            ).then(
                fn=mitigations,
                inputs=[self.usecase, self.assessed_risks, self.taxonomy], 
                # NOTETOSELF: Intent based risk is stored in self.risk (if needed)
                outputs=[self.assessed_risk_definition, self.relatedrisks, self.mitigations, self.benchmarks, self.requirements, self.mitigations_text, self.networks]
            )

        return demo

    def run(self): 
        self.ui = self.layout()
        self.ui.queue().launch(allowed_paths=["static/"], ssr_mode=False)


if __name__ == "__main__":
    ui = UI()
    ui.run()
