---
title: Risk Atlas Nexus
emoji: 🛡️
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Evaluate AI risks with common risk taxonomies
python_version: 3.11.11
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

A gradio application demonstrating Risk Atlas Nexus, running with WML.

Create an env file with your following values:
```
WML_API_KEY=''
WML_API_URL=''
WML_PROJECT_ID=''
```
To run from this directory:
```
uv venv --python 3.12 .venv-risk-atlas-nexus
source .venv-risk-atlas-nexus/bin/activate
uv pip install -r requirements.txt
python app.py
```