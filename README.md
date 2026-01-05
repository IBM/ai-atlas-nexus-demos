# AI Atlas Nexus Demos

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Scope

This project provides demos applications for [ai-atlas-nexus](https://github.com/IBM/ai-atlas-nexus).  Each of the folders in the top level of this project is a self contained demonstration.

## Usage and management of this repo

This project follows the same format as the AI Atlas Nexus project in terms of [contributing](https://github.com/IBM/ai-atlas-nexus/blob/main/CONTRIBUTING.md) and developer setup. Please [submit issues here](https://github.com/IBM/ai-atlas-nexus/issues/new/choose) relating to this project.

All content provided here is 'as is' and is maintained on a best effort basis.


### Add a demonstration
[fork]: https://github.com/IBM/ai-atlas-nexus-demos/fork
[pr]: https://github.com/IBM/ai-atlas-nexus-demos/compare
[released]: https://help.github.com/articles/github-terms-of-service/
If you wish to add a demonstration project:
1. [Fork][fork] and clone this repository
2. Create a new branch: `git checkout -b my-branch-name`
3. Make your change: create the new demonstration in a **single folder within the top level project**.
    - The demonstration folder must have its own README.md
    - The list of demonstrations in this folder must be updated.
4. Push to your fork and [submit a pull request][pr]
5. Wait for your pull request to be reviewed and merged.

Note that all contributions to this project are [released][released] to the public under the project's [opensource license](https://github.com/IBM/ai-atlas-nexus-demos/blob/main/LICENSE).


## Demonstration list

| Name| Tags | Description|
| :--- |  :--- | :--- |
| [Intent to capabilities](https://github.com/IBM/ai-atlas-nexus-demos/tree/main/capabilities) | AI capabilities, AI adapters, Gradio | This Gradio UI demonstrates an AI Atlas Nexus capabilities identification workflow.|
| [Risk Policy Distillation](https://github.com/IBM/ai-atlas-nexus-demos/tree/main/risk-policy-distillation) | risk policy extraction | Pipeline to generate high-level, concept-based local or global explanations for an LLM-as-a-Judge. |
| [Auto-BenchmarkCard](https://github.com/IBM/ai-atlas-nexus-demos/tree/main/auto-benchmarkcard) | benchmark documentation, LLM, risk identification, fact-checking | Automated workflow for generating validated AI benchmark documentation with multi-agent data extraction, LLM-driven synthesis, and factuality verification.|
| [Gaf-Guard](https://github.com/IBM/ai-atlas-nexus-demos/tree/main/gaf-guard) | risk management, AI agents, | GAF-Guard is an AI framework that can effectively detect and manage risks associated with LLMs for a given use-case.|
| [Neo4j-Db ](https://github.com/IBM/ai-atlas-nexus-demos/tree/main/neo4j-db) |neo4j, Docker | Set up a docker container with an instance of neo4j community and import the data from AI Atlas Nexus into it. |
| [AI Atlas Nexus Graph Visualisation ](https://github.com/IBM/ai-atlas-nexus-demos/tree/main/ran-viz) |python, sigmajs, viz, ui | Export AI Atlas Nexus content for a sigmajs graph and display it in the UI. |


## License
AI Atlas Nexus Demos is under Apache 2.0 license.


[View the detailed LICENSE](LICENSE).


## IBM ❤️ Open Source AI

AI Atlas Nexus has been brought to you by IBM.
