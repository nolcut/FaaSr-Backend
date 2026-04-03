# Possible FAASR Agent Workflows

The FaaSr-Agent-System operates as a directed LangGraph pipeline. While the default entry point is the `clarification` node, the architecture allows for various starting points depending on the user's current artifacts (e.g., if they already have a workflow or already have generated code) or desired end state.

Here are the possible workflows and entry points into the system:

## 1. The Full Pipeline (Default)
**Entry Point:** `clarification`
**Condition:** The user only has a natural language request and wants the system to handle the entire lifecycle.
**Flow:**
1. **Clarification:** Understands and structures the natural language request.
2. **Workflow Generation:** Designs a Directed Acyclic Graph (DAG) of FaaSr tasks.
3. **Reflection:** Validates the workflow tasks and ensures input/output lineage is correct.
4. **Coding:** Generates and tests Python code for the workflow tasks.
5. **Execute:** Deploys the FaaSr workflow and functions to GitHub and runs it.

## 2. Workflow Generation Only
**Entry Point:** `clarification`
**Condition:** The user has a natural language request and only wants the workflow schema (JSON) generated, without deploying or generating code.
**Flow:**
1. **Clarification:** Understands and structures the natural language request.
2. **Workflow Generation:** Designs a Directed Acyclic Graph (DAG) of FaaSr tasks.
3. **Reflection:** Validates the workflow tasks and converts it to the FaaSr-required format.
*Implementation Note: Requires setting the graph to route to `END` after the `reflection` node instead of routing to `coding`.*

## 3. Code Generation Only
**Entry Point:** `coding`
**Condition:** The user has already provided a structured FaaSr workflow JSON (and the corresponding `workflow_tasks`), but only needs code generated without automatic deployment.
**Flow:**
1. **Coding:** Generates and tests Python code for the user's tasks.
*Implementation Note: Requires setting `workflow_tasks` and `faasr_workflow` in the initial `GlobalState`, setting the entry point to `coding`, and configuring the graph to route to `END` after `coding`.*

## 4. Workflow Generation and Code Generation
**Entry Point:** `clarification`
**Condition:** The user has a natural language request and wants the system to generate the workflow and code, but does not want to automatically deploy it.
**Flow:**
1. **Clarification:** Understands and structures the natural language request.
2. **Workflow Generation:** Designs a Directed Acyclic Graph (DAG) of FaaSr tasks.
3. **Reflection:** Validates the workflow tasks and ensures input/output lineage is correct.
4. **Coding:** Generates and tests Python code for the workflow tasks.
*Implementation Note: Requires setting the graph to route to `END` after the `coding` node instead of routing to `execute`.*

## 5. Deployment Only
**Entry Point:** `execute`
**Condition:** The user already has the `workflows/` JSON and the Python `functions/` written. They just need the system to deploy it via GitHub Actions.
**Flow:**
1. **Execute:** Pushes the existing code and workflow schema to GitHub and triggers the execution.
*Implementation Note: Requires the workflow and function files to exist locally, and the graph entry point to be set to `execute`.*

## 6. Code Generation and Deployment
**Entry Point:** `coding`
**Condition:** The user has already provided a structured FaaSr workflow JSON, and needs code generated and deployed.
**Flow:**
1. **Coding:** Generates and tests Python code for the user's tasks.
2. **Execute:** Deploys the workflow and functions to GitHub.
*Implementation Note: Requires setting `workflow_tasks` and `faasr_workflow` in the initial `GlobalState` and routing the graph to start at `coding`.*

## Modifying the Entry Point and Routing
To utilize these alternate workflows, the graph construction in `src/faasr_ai/orchestrator.py` (`build_orchestrator_graph`) must be updated to conditionally set the entry point using `g.set_entry_point(NODE_NAME)`. Alternatively, you can compile separate graphs tailored to these specific flows. You must also adjust the `add_conditional_edges` logic if you want the pipeline to stop early (e.g., routing to `END`).
