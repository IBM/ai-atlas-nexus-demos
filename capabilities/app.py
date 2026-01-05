"""
AI Capabilities Navigator - Gradio UI

A local Gradio interface for exploring AI capabilities:
Natural Language Intent → AI Tasks → Capabilities → Recommended Intrinsics
"""

import gradio as gr
import sys
import base64

from ai_atlas_nexus import AIAtlasNexus
from ai_atlas_nexus.blocks.inference import OllamaInferenceEngine


def load_logo_as_base64(logo_path):
    """Load SVG logo and convert to base64 for embedding"""
    try:
        with open(logo_path, 'rb') as f:
            logo_data = f.read()
        return base64.b64encode(logo_data).decode('utf-8')
    except Exception as e:
        print(f"Could not load logo: {e}")
        return None


class CapabilitiesNavigator:
    """Main application class for capabilities navigation"""

    def __init__(self):
        """Initialize AI Atlas Nexus"""
        self.aan = AIAtlasNexus()
        self.inference_engine = None

        print("✓ AI Atlas Nexus initialized")
        print(f"  Loaded {len(self.aan.get_all("capabilities"))} capabilities")
        print(f"  Loaded {len(self.aan.get_all("aitasks"))} AI tasks")

    def configure_inference(self, model_name="granite3.3:8b", api_url="http://localhost:11434"):
        """Configure Ollama inference engine"""
        try:
            self.inference_engine = OllamaInferenceEngine(
                model_name_or_path=model_name,
                credentials={"api_url": api_url},
                parameters={"temperature": 0.1, "num_predict": 500}
            )
            return f"✓ Inference engine configured: {model_name}"
        except Exception as e:
            return f"✗ Failed to configure inference: {str(e)}"

    def identify_tasks_from_intent(self, user_intent):
        """Step 1: Identify AI tasks from natural language intent"""
        if not user_intent.strip():
            return "Please enter a use case description.", []

        if not self.inference_engine:
            return "Please configure the inference engine first.", []

        try:
            task_predictions = self.aan.identify_ai_tasks_from_usecases(
                usecases=[user_intent],
                inference_engine=self.inference_engine,
            )

            identified_tasks = task_predictions[0].prediction

            # Format output
            output = "### Identified AI Tasks:\n\n"
            for i, task in enumerate(identified_tasks, 1):
                output += f"{i}. **{task}**\n"

            return output, identified_tasks

        except Exception as e:
            return f"Error identifying tasks: {str(e)}", []

    def get_capabilities_for_tasks(self, tasks):
        """Step 2: Map tasks to required capabilities"""
        if not tasks:
            return "No tasks provided.", []

        output = ""
        all_capabilities = set()
        capabilities_by_task = {}

        for task in tasks:
            # Normalize task name to ID format
            task_id = task.lower().replace(" ", "-")
            print(f"[DEBUG get_capabilities_for_tasks] Processing task: {task} -> {task_id}")

            try:
                capabilities = self.aan.query(
                    class_name="capabilities", 
                    requiredByTask=task_id,
                    isDefinedByTaxonomy="ibm-ai-capabilities"
                )

                print(f"[DEBUG get_capabilities_for_tasks] Found {len(capabilities)} capabilities")

                if capabilities:
                    for cap in capabilities:
                        output += f"- **{cap.name}** (`{cap.id}`)\n"
                        if hasattr(cap, 'description') and cap.description:
                            desc = cap.description[:150] + "..." if len(cap.description) > 150 else cap.description
                            output += f"  - {desc}\n"
                        all_capabilities.add(cap.id)
                    output += "\n"
                else:
                    output += "*No capability mappings found*\n\n"

            except Exception as e:
                print(f"[DEBUG get_capabilities_for_tasks] Error: {str(e)}")
                import traceback
                traceback.print_exc()
                output += f"*Error: {str(e)}*\n\n"

        print(f"[DEBUG get_capabilities_for_tasks] Returning output length: {len(output)}, capabilities: {len(all_capabilities)}")
        return output, list(all_capabilities)

    def get_intrinsics_for_capabilities(self, capability_ids):
        """Step 3: Find intrinsics that implement the capabilities"""
        if not capability_ids:
            return "No capabilities provided."

        output = ""
        all_intrinsics = []

        for cap_id in capability_ids:
            try:
                # Get capability details
                all_caps = self.aan.get_all("capabilities")
                capability = next((c for c in all_caps if c.id == cap_id), None)

                if not capability:
                    continue

                # Get intrinsics for this capability
                intrinsics = self.aan.query(class_name="adapters",
                    implementsCapability=cap_id,   
                )

                if intrinsics:
                    output += f"**{capability.name}** ({len(intrinsics)} implementation(s))\n\n"
                    for intr in intrinsics:
                        output += f"- **{intr.name}** (`{intr.id}`)\n"
                        if hasattr(intr, 'description') and intr.description:
                            desc = intr.description[:200] + "..." if len(intr.description) > 200 else intr.description
                            output += f"  - {desc}\n"
                        all_intrinsics.append(intr)
                    output += "\n"
                else:
                    output += f"**{capability.name}**\n\n*No intrinsics found for this capability*\n\n"

            except Exception as e:
                output += f"*Error for {cap_id}: {str(e)}*\n\n"

        return output

    def process_full_pipeline(self, user_intent, use_llm=True):
        """Complete pipeline: Intent → Tasks → Capabilities → Intrinsics"""

        results = {
            "tasks": "",
            "capabilities": "",
            "intrinsics": ""
        }

        if use_llm:
            # Step 1: Identify tasks from intent
            tasks_output, identified_tasks = self.identify_tasks_from_intent(user_intent)
            results["tasks"] = tasks_output

            if not identified_tasks:
                return results["tasks"], "", ""
        else:
            # Use all available tasks
            all_tasks = self.aan.get_all("aitasks")
            identified_tasks = [task.name for task in all_tasks]
            results["tasks"] = "### Using All Available Tasks\n\n" + "\n".join([f"- {t}" for t in identified_tasks])

        # Step 2: Map to capabilities
        capabilities_output, capability_ids = self.get_capabilities_for_tasks(identified_tasks)
        results["capabilities"] = capabilities_output

        # Step 3: Find intrinsics
        intrinsics_output = self.get_intrinsics_for_capabilities(capability_ids)
        results["intrinsics"] = intrinsics_output

        return results["tasks"], results["capabilities"], results["intrinsics"]


def create_ui():
    """Create the Gradio interface"""

    print("Initializing AI Atlas Nexus... (this may take a minute)")
    navigator = CapabilitiesNavigator()
    print("✓ Ready!")

    # Load logo
    logo_path = "ai_atlas_nexus_vector.svg"
    logo_b64 = load_logo_as_base64(logo_path)

    logo_html = ""
    if logo_b64:
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" style="height: 70px; margin-right: 20px;"/>'

    with gr.Blocks(title="AI Capabilities Navigator") as app:

        # Header with IBM Blue styling and logo
        gr.HTML(f"""
        <style>
            .header-container {{
                display: flex;
                align-items: center;
                padding: 25px 30px;
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin-bottom: 25px;
            }}
            .header-content {{
                flex: 1;
            }}
            .main-title {{
                color: #161616;
                font-size: 2em;
                font-weight: 600;
                margin: 0 0 8px 0;
            }}
            .subtitle {{
                color: #525252;
                font-size: 1em;
                margin: 0;
            }}
            .description {{
                color: #525252;
                margin: 0 0 25px 0;
                font-size: 0.95em;
                line-height: 1.5;
            }}
            .result-section {{
                background: #f4f4f4;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 20px;
                margin: 15px 0;
            }}
            .result-section h3 {{
                color: #161616;
                font-size: 1.3em;
                margin-top: 0;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #0f62fe;
            }}
            /* Style for result content groups */
            .result-content {{
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 20px;
                margin: 10px 0 20px 0;
            }}
        </style>
        <div class="header-container">
            {logo_html}
            <div class="header-content">
                <h1 class="main-title">AI Capabilities Navigator</h1>
                <p class="subtitle">Natural Language Intent → AI Tasks → Capabilities → Recommended Intrinsics</p>
            </div>
        </div>
        <p class="description">
        This tool helps you identify the AI capabilities and model intrinsics needed to implement your use case.
        Loaded <strong>{len(navigator.aan.get_all("capabilities"))} capabilities</strong> and
        <strong>{len(navigator.aan.get_all("aitasks"))} AI tasks</strong> from the IBM AI Capabilities Framework.
        </p>
        """)

        with gr.Tab("🎯 Full Pipeline"):

            intent_input = gr.Textbox(
                label="Use Case Description",
                placeholder="Example: Generate personalized responses for customer support agents. The system should summarize claims, answer questions, and provide contextual suggestions...",
                lines=6
            )

            with gr.Accordion("⚙️ Inference Configuration", open=False):
                with gr.Row():
                    model_input = gr.Textbox(
                        label="Ollama Model",
                        value="granite3.3:8b",
                        scale=2
                    )
                    api_url_input = gr.Textbox(
                        label="API URL",
                        value="http://localhost:11434",
                        scale=2
                    )

                config_btn = gr.Button("Configure Inference Engine", variant="secondary", size="sm")
                config_status = gr.Textbox(label="Status", interactive=False, lines=1)

            analyze_btn = gr.Button("Submit", variant="secondary", size="lg")

            # Loading indicator for inference
            loading_msg = gr.Markdown("", visible=False)

            # Task selection section - hidden until inference completes
            with gr.Group(visible=False) as task_section:
                gr.Markdown("---")
                gr.Markdown("### 📋 Step 1: Identified AI Tasks")
                gr.Markdown("*The following tasks were identified from your use case. Click on a task to explore its capabilities and intrinsics.*")

                # Task selector - populated after inference
                task_selector = gr.Radio(
                    choices=[],
                    label="Select a task to explore",
                    value=None,
                    interactive=True
                )

            # Cascading panels - hidden until task is selected
            with gr.Group(visible=False) as cascade_group:
                # Loading indicator for cascade
                cascade_loading_msg = gr.Markdown("", visible=False)

                gr.Markdown("---")
                gr.Markdown("### 🎯 Task → Capabilities Mapping")
                with gr.Group():
                    capabilities_output = gr.Markdown(elem_classes=["result-content"])

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Capability → Intrinsics Recommendations")
                with gr.Group():
                    intrinsics_output = gr.Markdown(elem_classes=["result-content"])

            # Wire up buttons
            config_btn.click(
                fn=navigator.configure_inference,
                inputs=[model_input, api_url_input],
                outputs=config_status
            )

            # Combined handler that shows loading and runs inference
            def handle_inference_with_loading(intent, model_name, api_url):
                print(f"[DEBUG] handle_inference_with_loading called")
                print(f"[DEBUG] Intent length: {len(intent) if intent else 0}")
                print(f"[DEBUG] Intent preview: {intent[:100] if intent else 'EMPTY'}...")
                print(f"[DEBUG] Model: {model_name}, API: {api_url}")

                # Auto-configure inference engine if not already configured
                if not navigator.inference_engine:
                    print("[DEBUG] Configuring inference engine...")
                    config_result = navigator.configure_inference(model_name, api_url)
                    print(f"[DEBUG] Config result: {config_result}")
                    if "Failed" in config_result:
                        return (
                            gr.update(value=f"⚠️ {config_result}", visible=True),
                            gr.update(choices=[], value=None),
                            gr.update(visible=False),
                            gr.update(visible=False)
                        )

                print("[DEBUG] Identifying tasks from intent...")
                _, tasks = navigator.identify_tasks_from_intent(intent)
                print(f"[DEBUG] Identified tasks: {tasks}")
                print(f"[DEBUG] Number of tasks: {len(tasks) if tasks else 0}")
                print(f"[DEBUG] Tasks type: {type(tasks)}")

                if tasks:
                    result = (
                        gr.update(value="", visible=False),  # Hide loading
                        gr.update(choices=tasks, value=None),  # Update task selector (don't set visible here)
                        gr.update(visible=True),  # Show task section (which contains the Radio)
                        gr.update(visible=False)  # Hide cascade until task selected
                    )
                    print(f"[DEBUG] Returning result with {len(tasks)} tasks to UI")
                    print(f"[DEBUG] Result tuple: {result}")
                    return result

                return (
                    gr.update(value="⚠️ No tasks identified. Please try a different use case.", visible=True),
                    gr.update(choices=[], value=None),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            # Run inference on button click
            analyze_btn.click(
                fn=handle_inference_with_loading,
                inputs=[intent_input, model_input, api_url_input],
                outputs=[loading_msg, task_selector, task_section, cascade_group]
            )

            # Step 3: Show loading message immediately when task is selected
            def show_cascade_loading(task):
                if task:
                    return (
                        gr.update(value="🔄 **Loading capabilities and intrinsics...**", visible=True),  # Show loading
                        gr.update(value=""),  # Clear capabilities
                        gr.update(value=""),  # Clear intrinsics
                        gr.update(visible=True)  # Show cascade group
                    )
                return (
                    gr.update(value="", visible=False),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(visible=False)
                )

            # Step 4: Fetch and display capabilities and intrinsics for selected task
            def handle_task_selection(task):
                if task:
                    try:
                        print(f"[DEBUG] Fetching capabilities for task: {task}")
                        caps_output, cap_ids = navigator.get_capabilities_for_tasks([task])
                        print(f"[DEBUG] Found {len(cap_ids)} capabilities")

                        # If no capabilities found, show a message
                        if not caps_output or not caps_output.strip():
                            caps_output = "*No capabilities found for this task*"

                        print(f"[DEBUG] Fetching intrinsics for {len(cap_ids)} capabilities")
                        intrinsics_output = navigator.get_intrinsics_for_capabilities(cap_ids)
                        print(f"[DEBUG] Intrinsics output length: {len(intrinsics_output) if intrinsics_output else 0}")

                        # If no intrinsics found, show a message
                        if not intrinsics_output or not intrinsics_output.strip():
                            intrinsics_output = "*No intrinsics found for the identified capabilities*"

                        print("[DEBUG] Returning results")
                        return (
                            gr.update(value="", visible=False),  # Hide loading
                            caps_output,
                            intrinsics_output
                        )
                    except Exception as e:
                        print(f"[DEBUG] Error in handle_task_selection: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return (
                            gr.update(value="", visible=False),
                            f"*Error loading capabilities: {str(e)}*",
                            ""
                        )
                return (
                    gr.update(value="", visible=False),
                    "",
                    ""
                )

            # First show loading, then fetch data
            task_selector.change(
                fn=show_cascade_loading,
                inputs=task_selector,
                outputs=[cascade_loading_msg, capabilities_output, intrinsics_output, cascade_group]
            ).then(
                fn=handle_task_selection,
                inputs=task_selector,
                outputs=[cascade_loading_msg, capabilities_output, intrinsics_output]
            )

        with gr.Tab("📋 Explore Tasks"):

            all_tasks = navigator.aan.get_all("aitasks")
            task_names = [task.name for task in all_tasks]

            task_dropdown = gr.Dropdown(
                choices=task_names,
                label="Select AI Task",
                value=task_names[0] if task_names else None
            )

            explore_btn = gr.Button("Submit", variant="secondary", size="lg")

            gr.Markdown("## Results")

            with gr.Accordion("🎯 Required Capabilities", open=True):
                task_caps_output = gr.Markdown()

            with gr.Accordion("⚙️ Available Intrinsics", open=True):
                task_intrinsics_output = gr.Markdown()

            explore_btn.click(
                fn=lambda task: (
                    navigator.get_capabilities_for_tasks([task])[0],
                    navigator.get_intrinsics_for_capabilities(
                        navigator.get_capabilities_for_tasks([task])[1]
                    )
                ),
                inputs=task_dropdown,
                outputs=[task_caps_output, task_intrinsics_output]
            )

        with gr.Tab("🔍 Browse Capabilities"):

            all_capabilities = navigator.aan.get_all("capabilities")

            cap_data = []
            for cap in sorted(all_capabilities, key=lambda x: x.isPartOf):
                desc = cap.description[:100] + "..." if hasattr(cap, 'description') and cap.description and len(cap.description) > 100 else getattr(cap, 'description', '')
                cap_data.append([cap.name, cap.id, cap.isPartOf, desc])

            gr.Dataframe(
                headers=["Capability", "ID", "Group", "Description"],
                value=cap_data,
                label="IBM AI Capabilities Framework"
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Tool

            This Gradio UI demonstrates the AI Atlas Nexus capabilities identification workflow:

            ### Workflow Steps:

            1. **Natural Language Intent** → User describes their use case in plain language
            2. **AI Tasks** → LLM identifies relevant AI tasks (e.g., Question Answering, Summarization)
            3. **Capabilities** → System maps tasks to required capabilities from IBM AI Capabilities Framework
            4. **Intrinsics** → System recommends specific model intrinsics and adapters that implement those capabilities

            ### IBM AI Capabilities Framework:

            The framework organizes capabilities into 5 domains:
            - **Code**: Code generation, understanding, explanation
            - **Comprehension**: Reading comprehension, contextual understanding
            - **Domain Knowledge**: Domain expertise, common sense reasoning
            - **Factuality**: Factual accuracy and reliability
            - **Generation**: Text generation
            - **Logical**: Logical reasoning, causal reasoning
            - **Mathematical**: Mathematical and quantitative reasoning

            ### Requirements:

            - **Ollama** running locally with a model installed (e.g., granite3.2:8b)
            - **ai-atlas-nexus** Python package

            ### Source:

            Based on the `ai_capabilities_identification.ipynb` notebook from the AI Atlas Nexus project.
            """)

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )