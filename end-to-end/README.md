# FaaSr Agent Project

A multi-agent system that converts user requests into validated workflows, deploys them to a FaaSr GitHub Actions repository, executes them, and iterates based on execution logs.

---

## Agents

### 1. Clarification Agent
Refines user requests by:
- Classifying user intent
- Extracting required specifications per intent
- Detecting ambiguity and asking follow-up questions until all specs are satisfied

**Intent → Required Specs Mapping:**

| Intent | Required Specs |
|--------|----------------|
| `transform-only` | `inputs`, `transformation`, `output` |
| `analysis` | `inputs`, `analysis_goal`, `output` |
| `modeling` | `inputs`, `modeling_task`, `target`, `output` |

### 2. Workflow Generation Agent
Creates step-by-step workflows as task lists (pre-FaaSr format). Each task includes dependencies, instructions, and explicit input/output files.

**Example Task:**
```json
{
  "task_id": "1",
  "dependent_task_ids": [],
  "instruction": "...",
  "inputs": ["input_file.csv"],
  "outputs": ["output_file.csv"]
}
```

### 3. Reflection Agent
Validates and improves workflows for up to `MAX_LOOP` iterations.

**Validation Criteria:**
- **Task Dependencies**: Valid DAG structure, no cycles, single entrypoint, all tasks reachable
- **Dependency Direction**: Dependencies must point backward to prerequisite tasks only
- **Redundancy Check**: No duplicate tasks
- **File Consistency**: Input/output file alignment
- **Completeness**: All required tasks present
- **Task Instructions**: Clear and executable
- **Input/Output Clarity**: Explicit file specifications

### 4. Execution Agent
Finalizes and deploys workflows:
- Uploads workflow and function files to GitHub repo (`workflows/`, `functions/`)
- Registers and invokes the workflow

### 5. Log Tracking Agent
Analyzes execution logs to:
- Identify failures and root causes
- Propose fixes (workflow edits/function updates)
- Support iterative reruns

---

## Setup

### 1. Configure Environment Variables
Copy `.env.template` to `.env` and fill in required values:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `FAASR_GH_USERNAME` | GitHub username for FaaSr |
| `FAASR_ACTION_REPO` | GitHub Actions repository name |
| `FAASR_WORKFLOW_NAME` | Name of the workflow |
| `FAASR_S3_ENDPOINT` | S3-compatible endpoint URL |
| `FAASR_S3_BUCKET` | S3 bucket name |
| `FAASR_S3_REGION` | S3 region |
| `GH_PAT` | GitHub Personal Access Token |
| `GITHUB_REPOSITORY` | GitHub repository (owner/repo format) |
| `GITHUB_REF_NAME` | Repository branch/ref name |
| `S3_AccessKey` | S3 access key |
| `S3_SecretKey` | S3 secret key |

### 2. Install Dependencies
Install all dependencies including dev group:
```bash
uv sync --all-groups
```

**Activate:**
- **macOS/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`

---

## Usage

### Run the Agent System
```bash
python -m faasr_ai.entrypoint
```

### Run Tests
Tests are located in `tests/`:
```bash
python -m faasr_ai.tests.<test_file_name>
```

---

## Project Structure
```
faasr_ai/
├── workflows/          # Generated workflow files
├── functions/          # Generated function files
├── tests/             # Test modules
└── entrypoint.py      # Main entry point
```