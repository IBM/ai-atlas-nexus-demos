# AI Atlas Nexus Intent to Capabilities demo

This Gradio UI demonstrates an AI Atlas Nexus capabilities identification workflow.

Workflow Steps:
- Natural Language Intent:  User describes their use case in plain language
- AI Tasks:  LLM identifies relevant AI tasks (e.g., Question Answering, Summarization)
- Capabilities: System maps tasks to required capabilities from IBM AI Capabilities Framework
- Intrinsics: System recommends specific model intrinsics and adapters that implement those capabilities

# Prerequisites
1. **Ollama** installed and running locally
```bash
# Install Ollama: https://ollama.ai
# Pull a model
ollama pull granite3.3:8b
```

## Installation
```
python -m venv .v-ai-atlas-nexus-cap
source .v-ai-atlas-nexus-cap/bin/activate
python -m pip install -r requirements.txt
```


## Running the UI
```commandline
python -m app
```

The UI will launch at `http://127.0.0.1:7860`


