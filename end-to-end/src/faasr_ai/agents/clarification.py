# src/faasr_ai/agents/clarification.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Callable
import json
import re
import logging

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

Intent = Literal["transform-only", "analysis", "modeling", "agent-workflow"]

INTENTS: List[Intent] = ["transform-only", "analysis", "modeling", "agent-workflow"]

REQUIRED_SPECS_BY_INTENT: Dict[Intent, List[str]] = {
    "transform-only": ["inputs", "transformation", "output"],
    "analysis": ["inputs", "analysis_goal", "output"],
    "modeling": ["inputs", "modeling_task", "target", "output"],
    "agent-workflow": ["goal", "data_sources", "agent_steps", "outputs"],
}


class ClarificationState(TypedDict, total=False):
    """
    Specialist state for CLAM-style clarification.
    This can be embedded into GlobalState under key: state["clarification"].
    """

    # Raw user request
    question: str

    # Intent classification
    intent: Intent

    # Extracted specs
    inputs: str
    transformation: str
    analysis_goal: str
    modeling_task: str
    target: str
    output: str

    # Loop control
    missing_fields: List[str]          # treated as a queue
    pending_clarifier: str

    # Produced for the UI / coordinator to ask user
    clarifying_question: str
    suggestions: List[str]

    # Agent-workflow specs
    goal: str
    data_sources: str
    agent_steps: str
    outputs: str

    # Global inputs (deterministically captured from user)
    global_input_files: List[str]
    global_input_folders: List[str]

    # Terminal input
    user_answer: str

    # Final artifact
    final_structured_request: Dict[str, Any]
    enhanced_natural_language_request: str
    
    # Convenience flags
    ready: bool  # True when no missing_fields remain

    # One-field ambiguity checking
    last_checked_field: str
    last_checked_ambiguous: bool

    # Natural language request approval flow
    user_approved: bool
    user_feedback: str



def extract_json_block(text: str) -> Optional[dict]:
    text = text.replace("```", "").strip()

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # scan candidates
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _clean_bool(text: str) -> bool:
    ans = re.sub(r"[^a-zA-Z]", "", text).lower().strip()
    return ans == "true"


# -----------------------------
# Node implementations
# -----------------------------

def classify_intent_node(state: ClarificationState, llm: Callable[[str], str]) -> ClarificationState:
    q = state["question"]
    
    print("User request (RAW): ", q)

    prompt = f"""
        Classify the user's request into exactly ONE intent label:

        - transform-only: generate/load/transform data and produce outputs; no exploration required
        - analysis: exploratory/statistical understanding, validation, insights; no model training required
        - modeling: train/evaluate/use a predictive or clustering model
        - agent-workflow: multi-step agentic workflow where each step is described in natural language

        Return ONLY one of: transform-only, analysis, modeling, agent-workflow
        No punctuation. No explanation.

        User request:
        {q}
    """.strip()

    label = llm(prompt).lower().strip()
    if label not in INTENTS:
        label = "transform-only"  # safe default

    state["intent"] = label  
    
    # initialize queue of required specs for this intent
    state["missing_fields"] = list(REQUIRED_SPECS_BY_INTENT[state["intent"]])
    state["ready"] = False
    state["last_checked_field"] = ""
    state["last_checked_ambiguous"] = False
    return state


def extract_specs_node(state: ClarificationState, llm: Callable[[str], str]) -> ClarificationState:
    q = state["question"]

    intent = state.get("intent", "transform-only")

    if intent == "agent-workflow":
        prompt = f"""
            Extract the following fields from the user request. If a field is missing or unclear, set it to "".

            Fields:
            - goal: the overall objective of the workflow
            - data_sources: what data is used, where it comes from, or how it is generated
            - agent_steps: the sequence of agent actions or stages in the workflow
            - outputs: the final deliverables or artifacts produced

            Return ONLY valid JSON with these exact keys.
            All values must be strings.
            No backticks. No code fences. No explanations.
            Output must start with '{{' and end with '}}'.

            User request:
            {q}
        """.strip()
    else:
        prompt = f"""
            Extract the following fields from the user request. If a field is missing or unclear, set it to "".

            Fields:
            - inputs: data source OR generation of data (do not omit)
            - transformation: the data operation(s) to perform (ONLY for transform-only intent)
            - analysis_goal: what to learn/check/understand (ONLY for analysis intent)
            - modeling_task: classification/regression/clustering/forecasting (ONLY for modeling intent)
            - target: prediction target or modeling objective (ONLY for modeling intent)
            - output: what deliverables to produce (format + content)

            Return ONLY valid JSON with these exact keys.
            All values must be strings.
            No backticks. No code fences. No explanations.
            Output must start with '{{' and end with '}}'.

            User request:
            {q}
        """.strip()

    raw = llm(prompt)
    extracted = extract_json_block(raw) or {}

    if intent == "agent-workflow":
        for k in ["goal", "data_sources", "agent_steps", "outputs"]:
            state[k] = str(extracted.get(k, "") or "").strip()  # type: ignore[literal-required]

        logger.debug("\n[Specs Extracted from RAW user request]")
        logger.debug("  goal:", repr(state.get("goal", "")))
        logger.debug("  data_sources:", repr(state.get("data_sources", "")))
        logger.debug("  agent_steps:", repr(state.get("agent_steps", "")))
        logger.debug("  outputs:", repr(state.get("outputs", "")))
    else:
        for k in ["inputs", "transformation", "analysis_goal", "modeling_task", "target", "output"]:
            state[k] = str(extracted.get(k, "") or "").strip()  # type: ignore[literal-required]

        logger.debug("\n[Specs Extracted from RAW user request]")
        logger.debug("  inputs:", repr(state.get("inputs", "")))
        logger.debug("  transformation:", repr(state.get("transformation", "")))
        logger.debug("  analysis_goal:", repr(state.get("analysis_goal", "")))
        logger.debug("  modeling_task:", repr(state.get("modeling_task", "")))
        logger.debug("  target:", repr(state.get("target", "")))
        logger.debug("  output:", repr(state.get("output", "")))

    return state


def llm_is_ambiguous(
    llm: Callable[[str], str],
    user_request: str,
    intent: str,
    field: str,
    value: str,
    state: ClarificationState,
) -> bool:
    """
    Decide whether ONE field is ambiguous, given ONLY the required specs
    for the current intent.
    """

    required_fields = REQUIRED_SPECS_BY_INTENT[intent]

    context = {
        f: state.get(f, "")
        for f in required_fields
    }

    prompt = f"""
        You are a strict ambiguity checker for workflow specifications.

        Original user request:
        {user_request}

        Intent:
        {intent}

        Current extracted specification (required fields only):
        {json.dumps(context, indent=2)}

        Field under evaluation:
        {field}

        Value of this field:
        {value}

        Return ONLY True or False (no punctuation, no explanation).

        Return True if this field is missing OR does not fulfill the criteria.
        Return False if it is specific enough in context.

        Criteria reminders:
        - inputs: concrete data source OR explicit instruction to generate data
        - transformation: at least one transformation
        - analysis_goal: specific question(s) or checks
        - modeling_task: clear task type (classification/regression/clustering/forecasting)
        - target: explicit target variable or objective
        - output: deliverable type(s) AND content
        - goal: clear overall objective for the workflow
        - data_sources: specific data sources, APIs, or generation methods
        - agent_steps: distinct stages/actions the workflow should perform
        - outputs: concrete deliverables or artifacts

        If the Value is empty, return True.
    """.strip()

    return _clean_bool(llm(prompt))



def detect_ambiguity_node(state: ClarificationState, llm: Callable[[str], str]) -> ClarificationState:
    intent = state["intent"]

    if "missing_fields" not in state or state["missing_fields"] is None:
        state["missing_fields"] = list(REQUIRED_SPECS_BY_INTENT[intent])

    if len(state["missing_fields"]) == 0:
        state["ready"] = True
        state["last_checked_field"] = ""
        state["last_checked_ambiguous"] = False
        return state

    field = state["missing_fields"][0]
    val = (state.get(field, "") or "").strip()

    ambiguous = llm_is_ambiguous(llm, state["question"], intent, field, val, state)

    state["last_checked_field"] = field
    state["last_checked_ambiguous"] = ambiguous

    logger.debug(f"Checking field='{field}' val={repr(val)} ambiguous={ambiguous}")

    if not ambiguous:
        # if field not ambiguous, remove from queue
        state["missing_fields"] = state["missing_fields"][1:]
        state["ready"] = len(state["missing_fields"]) == 0
        logger.debug(f"Removed '{field}'. Remaining: {state['missing_fields']}")
    else:
        state["ready"] = False
        logger.debug(f"'{field}' is ambiguous; will ask user.")

    return state


def generate_clarifying_question_node(state: ClarificationState, llm: Callable[[str], str]) -> ClarificationState:
    intent = state["intent"]
    field = state["missing_fields"][0]
    current_value = state.get(field, "")

    prompt = f"""
        You are a clarification bot. Ask ONE natural question to fill the missing spec.

        Intent: {intent}
        Missing spec: {field}
        Current value: {current_value}

        Spec meanings:
        - inputs: data source OR how to generate the data (rows/columns/types/ranges)
        - transformation: exact operation(s) to perform on the data
        - analysis_goal: what to learn/check/understand via exploration/statistics
        - modeling_task: classification/regression/clustering/forecasting
        - target: prediction target variable or modeling objective
        - output: deliverables (format + what content should be shown)
        - goal: overall objective of the agentic workflow
        - data_sources: specific data sources, APIs, files, or generation methods
        - agent_steps: distinct stages/actions the workflow performs in sequence
        - outputs: concrete final deliverables or artifacts

        Rules:
        - Ask exactly ONE question.
        - No bullet points, no explanations, no examples.
        - Output ONLY the question text.
    """.strip()

    state["pending_clarifier"] = field
    state["clarifying_question"] = llm(prompt).replace("```", "").strip()
    return state


def generate_suggestions_node(state: ClarificationState, llm: Callable[[str], str]) -> ClarificationState:
    intent = state["intent"]
    field = state["pending_clarifier"]
    question = state["clarifying_question"]
    user_request = state["question"]

    prompt = f"""
        You generate exactly 3 candidate answers a user might give to a clarification question.

        Context:
        - Original user request: {user_request}
        - Intent: {intent}
        - Missing spec field: {field}
        - Clarifying question: {question}

        Requirements:
        - Return ONLY valid JSON in this exact format:
        {{"suggestions": ["...", "...", "..."]}}
        - Exactly 3 items.
        - Each suggestion must be specific enough to reduce ambiguity for the missing field.
        - Keep each suggestion short (one sentence).
        - No backticks. No code fences. No explanations.
    """.strip()

    raw = llm(prompt)
    obj = extract_json_block(raw) or {}
    sugg = obj.get("suggestions", [])

    if not isinstance(sugg, list):
        sugg = []
    sugg = [str(x).strip() for x in sugg if str(x).strip()]

    while len(sugg) < 3:
        sugg.append("")
    state["suggestions"] = sugg[:3]
    return state


def _reformat_string(llm: Callable[[str], str], text: str) -> str:
    prompt = f"""
        Rewrite the following text into one concise, grammatically correct spec.
        Do NOT add new details. Keep the meaning.

        Text:
        {text}

        Return ONLY the rewritten text.
    """.strip()
    return llm(prompt).replace("```", "").strip()


def user_clarification_node(state: ClarificationState) -> ClarificationState:
    """
    Terminal-based clarification.
    This blocks and asks the user for input, then stores it in state["user_answer"].
    """
    question = state.get("clarifying_question", "").strip()
    suggestions = state.get("suggestions", []) or []

    print("\nClarifying question:")
    print(question)

    if any(s.strip() for s in suggestions):
        print("\nSuggested answers (type 1/2/3 to select, or type your own):")
        for i, s in enumerate(suggestions, 1):
            if s.strip():
                print(f"  {i}. {s}")

    ans = input("\nYour answer: ").strip()

    # If user selects a suggestion
    if ans in {"1", "2", "3"}:
        idx = int(ans) - 1
        if 0 <= idx < len(suggestions) and suggestions[idx].strip():
            ans = suggestions[idx].strip()
            print(f"[Selected suggestion] {ans}")

    state["user_answer"] = ans
    return state


def apply_user_answer_node(state: ClarificationState, llm: Callable[[str], str]) -> ClarificationState:
    field = state.get("pending_clarifier", "")
    answer = (state.get("user_answer", "") or "").strip()

    if not field or not answer:
        return state  # nothing to apply

    combined = (state.get(field, "") + ", " + answer).strip(", ").strip()
    state[field] = _reformat_string(llm, combined)  # type: ignore[literal-required]
    
    # reset
    state["pending_clarifier"] = ""
    state["user_answer"] = ""
    state["clarifying_question"] = ""
    state["suggestions"] = []
    return state


def final_output_node(state: ClarificationState) -> ClarificationState:
    intent = state["intent"]

    if intent == "transform-only":
        structured = {
            "intent": intent,
            "inputs": state.get("inputs", ""),
            "transformation": state.get("transformation", ""),
            "output": state.get("output", ""),
        }
        nl = (
            "Final Request (transform-only)\n\n"
            f"Inputs:\n{structured['inputs']}\n\n"
            f"Transformation:\n{structured['transformation']}\n\n"
            f"Output:\n{structured['output']}"
        )

    elif intent == "analysis":
        structured = {
            "intent": intent,
            "inputs": state.get("inputs", ""),
            "analysis_goal": state.get("analysis_goal", ""),
            "output": state.get("output", ""),
        }
        nl = (
            "Final Request (analysis)\n\n"
            f"Inputs:\n{structured['inputs']}\n\n"
            f"Analysis goal:\n{structured['analysis_goal']}\n\n"
            f"Output:\n{structured['output']}"
        )

    elif intent == "agent-workflow":
        structured = {
            "intent": intent,
            "goal": state.get("goal", ""),
            "data_sources": state.get("data_sources", ""),
            "agent_steps": state.get("agent_steps", ""),
            "outputs": state.get("outputs", ""),
            "global_input_files": state.get("global_input_files", []),
            "global_input_folders": state.get("global_input_folders", []),
        }
        nl = (
            "Final Request (agent-workflow)\n\n"
            f"Goal:\n{structured['goal']}\n\n"
            f"Data Sources:\n{structured['data_sources']}\n\n"
            f"Agent Steps:\n{structured['agent_steps']}\n\n"
            f"Outputs:\n{structured['outputs']}"
        )

    else:
        structured = {
            "intent": intent,
            "inputs": state.get("inputs", ""),
            "modeling_task": state.get("modeling_task", ""),
            "target": state.get("target", ""),
            "output": state.get("output", ""),
        }
        nl = (
            "Final Request (modeling)\n\n"
            f"Inputs:\n{structured['inputs']}\n\n"
            f"Modeling task:\n{structured['modeling_task']}\n\n"
            f"Target:\n{structured['target']}\n\n"
            f"Output:\n{structured['output']}"
        )

    state["final_structured_request"] = structured
    state["final_structured_request_natural_language"] = nl
    return state


def generate_enhanced_natural_language_request_node(
    state: ClarificationState, llm: Callable[[str], str]
) -> ClarificationState:
    """
    Converts the structured request into a detailed, natural language workflow request
    that includes helpful context and specifics for workflow generation.
    """
    structured = state.get("final_structured_request", {})

    prompt = f"""
        You are a workflow specification writer. Convert the following structured request into a detailed,
        natural language workflow request that will be used to generate an executable workflow.

        Structured Request:
        {json.dumps(structured, indent=2)}

        Requirements:
        1. Write in clear, natural language (not bullet points or structured format)
        2. Include all technical details from the structured request
        3. Add helpful context for workflow generation, such as:
           - Expected data types and formats
           - Specific steps or operations needed
           - Output format and presentation requirements
           - Any assumptions or constraints
        4. Make it actionable and executable
        5. Be as detailed as the workflow requires — do not truncate steps or omit details to stay brief
        6. Do NOT include any file names, folder names, or paths — describe data by type and content only

        Return ONLY the natural language request text. No explanations, no preamble.
    """.strip()

    enhanced_request = llm(prompt).replace("```", "").strip()
    state["enhanced_natural_language_request"] = enhanced_request
    
    # Initialize approval state
    state["user_approved"] = False
    state["user_feedback"] = ""
    
    return state


def request_user_approval_node(state: ClarificationState) -> ClarificationState:
    """
    Displays the enhanced natural language request and asks for user approval.
    """
    enhanced_request = state.get("enhanced_natural_language_request", "")

    print("\n=== Clarification agent output (final detailed request) ===")
    print("\n" + enhanced_request + "\n")
    
    while True:
        response = input("\nDo you approve this request? (yes/no): ").strip().lower()
        
        if response in ["yes", "y", "approve", "approved"]:
            state["user_approved"] = True
            state["user_feedback"] = ""
            print("\n✓ Request approved!")
            break
        elif response in ["no", "n"]:
            feedback = input("\nPlease provide your feedback or suggestions for improvement: ").strip()
            state["user_approved"] = False
            state["user_feedback"] = feedback
            print("\n✓ Feedback received. Regenerating request...")
            break
        else:
            print("Please enter 'yes', 'no'")
    
    return state


def apply_user_feedback_node(state: ClarificationState, llm: Callable[[str], str]) -> ClarificationState:
    """
    Applies user feedback to regenerate the enhanced natural language request.
    """
    current_request = state.get("enhanced_natural_language_request", "")
    feedback = state.get("user_feedback", "")
    structured = state.get("final_structured_request", {})
    
    if not feedback:
        return state
    
    prompt = f"""
        You are a workflow specification writer. Revise the following workflow request based on user feedback.

        Current Request:
        {current_request}

        Original Structured Specification:
        {json.dumps(structured, indent=2)}

        User Feedback:
        {feedback}

        Requirements:
        1. Incorporate the user's feedback while maintaining all required specifications
        2. Keep the same intent and core requirements from the structured specification
        3. Write in clear, natural language
        4. Ensure the revised request is actionable and executable
        5. Keep it concise but comprehensive (2-4 sentences)

        Return ONLY the revised natural language request text. No explanations, no preamble.
    """.strip()

    revised_request = llm(prompt).replace("```", "").strip()
    state["enhanced_natural_language_request"] = revised_request
    
    # Reset for next approval round
    state["user_approved"] = False
    state["user_feedback"] = ""
    
    return state


def collect_global_inputs_node(state: ClarificationState) -> ClarificationState:
    """
    Ask the user for pre-existing S3 files/folders to include as global inputs.
    Fully deterministic -- no LLM. Entries ending with '/' go to global_input_folders,
    everything else goes to global_input_files.
    """
    print("\nDo you have pre-existing data files on S3 to make available to all actions?")
    print("Enter file paths or glob patterns, one per line.")
    print("  Examples: data/input.csv   data/*.txt   results/*/output.json")
    print("  Folders:  raw-data/   (trailing slash)")
    print("Press Enter on an empty line to skip.\n")

    files = []
    folders = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        if line.endswith("/"):
            folders.append(line)
        else:
            files.append(line)

    state["global_input_files"] = files
    state["global_input_folders"] = folders
    return state


# -----------------------------
# Routing
# -----------------------------
def router_after_detect(state: ClarificationState) -> str:
    # Done
    if state.get("ready", False):
        return "final_output"

    # Ask about current field if ambiguous
    if state.get("last_checked_ambiguous", False):
        return "generate_clarifying_question"

    # Otherwise, we just removed a non-ambiguous field; keep scanning
    return "detect_ambiguity"


def router_after_final_output(state: ClarificationState) -> str:
    if state.get("intent") == "agent-workflow":
        return "collect_global_inputs"
    return "generate_enhanced_nl_request"


def router_after_approval(state: ClarificationState) -> str:
    """
    Routes based on user approval.
    """
    if state.get("user_approved", False):
        return "END"
    else:
        return "apply_user_feedback"


# -----------------------------
# Build graph (in orchestrator.py)
# -----------------------------
def build_clarification_graph(llm: Callable[[str], str]):
    g = StateGraph(ClarificationState)
    
    g.add_node("classify_intent", lambda s: classify_intent_node(s, llm))
    g.add_node("extract_specs", lambda s: extract_specs_node(s, llm))
    g.add_node("detect_ambiguity", lambda s: detect_ambiguity_node(s, llm))

    g.add_node("generate_clarifying_question", lambda s: generate_clarifying_question_node(s, llm))
    g.add_node("generate_suggestions", lambda s: generate_suggestions_node(s, llm))

    g.add_node("user_clarification", user_clarification_node)

    g.add_node("apply_user_answer", lambda s: apply_user_answer_node(s, llm))
    g.add_node("final_output", final_output_node)
    g.add_node("collect_global_inputs", collect_global_inputs_node)

    g.add_node("generate_enhanced_nl_request", lambda s: generate_enhanced_natural_language_request_node(s, llm))
    g.add_node("request_user_approval", request_user_approval_node)
    g.add_node("apply_user_feedback", lambda s: apply_user_feedback_node(s, llm))

    g.set_entry_point("classify_intent")

    g.add_edge("classify_intent", "extract_specs")
    g.add_edge("extract_specs", "detect_ambiguity")

    g.add_conditional_edges(
        "detect_ambiguity",
        router_after_detect,
        {
            "detect_ambiguity": "detect_ambiguity",  # keep scanning queue
            "generate_clarifying_question": "generate_clarifying_question",
            "final_output": "final_output",
        },
    )

    g.add_edge("generate_clarifying_question", "generate_suggestions")
    g.add_edge("generate_suggestions", "user_clarification")
    g.add_edge("user_clarification", "apply_user_answer")
    g.add_edge("apply_user_answer", "detect_ambiguity")

    g.add_conditional_edges(
        "final_output",
        router_after_final_output,
        {
            "collect_global_inputs": "collect_global_inputs",
            "generate_enhanced_nl_request": "generate_enhanced_nl_request",
        },
    )
    g.add_edge("collect_global_inputs", "generate_enhanced_nl_request")
    g.add_edge("generate_enhanced_nl_request", "request_user_approval")
    
    # NEW: Conditional routing based on approval
    g.add_conditional_edges(
        "request_user_approval",
        router_after_approval,
        {
            "apply_user_feedback": "apply_user_feedback",
            "END": END,
        },
    )
    
    g.add_edge("apply_user_feedback", "request_user_approval")

    return g.compile()


# -----------------------------
# Initialize clarification state (in orchestrator.py)
# -----------------------------
def make_initial_clarification_state(
    question: str,
    intent: Intent | None = None,
) -> ClarificationState:
    state: ClarificationState = {
        "question": question,
        "intent": intent or "transform-only",
        "inputs": "",
        "transformation": "",
        "analysis_goal": "",
        "modeling_task": "",
        "target": "",
        "output": "",
        "goal": "",
        "data_sources": "",
        "agent_steps": "",
        "outputs": "",
        "missing_fields": [],
        "pending_clarifier": "",
        "clarifying_question": "",
        "suggestions": [],
        "user_answer": "",
        "final_structured_request": {},
        "final_structured_request_natural_language": "",
        "ready": False,
        "last_checked_field": "",
        "last_checked_ambiguous": False,
        "enhanced_natural_language_request": "",
        "user_approved": False,
        "user_feedback": "",
        "global_input_files": [],
        "global_input_folders": [],
    }
    return state
