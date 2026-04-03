# src/faasr_ai/entrypoint.py
from __future__ import annotations

import json
import os
import queue
import threading
from dotenv import load_dotenv
from typing import Any, Dict, Optional, Callable

try:
    import readline  # noqa: F401 — enables backspace/delete/arrow keys in input()
except ImportError:
    pass  # Windows: readline not available, fall back to raw input

from faasr_ai.orchestrator import build_orchestrator_graph

load_dotenv()

DEFAULT_SYSTEM = "Follow instructions precisely. Output valid JSON when asked."


# ---------------------------------------------------------------------------
# Thread-safe human input broker
# ---------------------------------------------------------------------------

class _HumanInputBroker:
    """
    Lets the agent thread request human input without owning stdin.
    The main thread polls for pending requests and delivers answers.
    """

    def __init__(self) -> None:
        self._req: queue.Queue[str] = queue.Queue(maxsize=1)
        self._resp: queue.Queue[str] = queue.Queue(maxsize=1)
        self._pending = threading.Event()

    def ask(self, prompt: str) -> str:
        """Called from agent thread. Blocks until main thread answers."""
        self._req.put(prompt)
        self._pending.set()
        return self._resp.get()

    def poll(self) -> tuple:
        """Called from main thread. Returns (prompt, True) if request pending."""
        if not self._pending.is_set():
            return "", False
        try:
            prompt = self._req.get_nowait()
            self._pending.clear()
            return prompt, True
        except queue.Empty:
            return "", False

    def answer(self, text: str) -> None:
        """Called from main thread to deliver answer to agent thread."""
        self._resp.put(text)


def make_llm_call(provider: str, temperature: Optional[float] = None) -> Callable[[str, Optional[str]], str]:
    """Factory that returns an llm_call function for the given provider.

    Args:
        provider: 'anthropic' or 'openai'
        temperature: explicit temperature override; falls back to the
                     ANTHROPIC_TEMPERATURE / OPENAI_TEMPERATURE env vars (default 0).
    """

    if provider == "anthropic":
        import anthropic

        if key := os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = key.strip()

        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        _temp = temperature if temperature is not None else float(os.getenv("ANTHROPIC_TEMPERATURE", "0"))
        client = anthropic.Anthropic()

        def llm_call(prompt: str, system: Optional[str] = None) -> str:
            resp = client.messages.create(
                model=model,
                temperature=_temp,
                max_tokens=4096,
                system=system or DEFAULT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.content[0].text or "").strip()

    elif provider == "openai":
        from openai import OpenAI

        if key := os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = key.strip()

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        _temp = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", "0"))
        client = OpenAI()

        def llm_call(prompt: str, system: Optional[str] = None) -> str:
            resp = client.chat.completions.create(
                model=model,
                temperature=_temp,
                messages=[
                    {"role": "system", "content": system or DEFAULT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

    else:
        raise ValueError(f"Unsupported provider: {provider!r}. Choose 'anthropic' or 'openai'.")

    return llm_call


# ---------------------------------------------------------------------------
# Oversight agent
# ---------------------------------------------------------------------------

def _has_errors(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("workflow_errors")
        or state.get("coding_errors")
        or state.get("has_issues")
        or state.get("_exception")
    )


def _oversight_agent(llm_call: Callable, state: Dict[str, Any]) -> None:
    """Interactive LLM session to diagnose agent failures."""
    print("\n=== Oversight Agent ===")
    print("Type your question about what went wrong. Type 'exit' to leave.\n")

    ctx_parts: list[str] = ["## Agent run summary"]
    if state.get("_exception"):
        ctx_parts.append(f"Unhandled exception: {state['_exception']}")
    if state.get("workflow_errors"):
        ctx_parts.append(f"Workflow errors: {state['workflow_errors']}")
    if state.get("coding_errors"):
        ctx_parts.append(f"Coding errors: {state['coding_errors']}")
    if state.get("has_issues"):
        vr = state.get("validation_result", {})
        ctx_parts.append(
            f"Reflection could not resolve all issues "
            f"(quality={vr.get('overall_quality', 'unknown')}, "
            f"issues={len(vr.get('issues', []))})"
        )
    if state.get("workflow_tasks"):
        ctx_parts.append(f"Workflow tasks generated: {len(state['workflow_tasks'])}")

    system_ctx = "\n".join(ctx_parts)
    safe_state = {k: v for k, v in state.items() if k != "_exception"}
    history: list[str] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if user_input.lower() in ("exit", "quit", "q", ""):
            break

        history.append(f"User: {user_input}")
        prompt = (
            f"You are an oversight agent helping diagnose a FaaSr workflow generation failure.\n\n"
            f"{system_ctx}\n\n"
            f"Full agent state (JSON):\n"
            f"{json.dumps(safe_state, indent=2, default=str)}\n\n"
            f"Conversation so far:\n{chr(10).join(history)}\n\n"
            f"Give a concise, actionable response."
        )
        response = llm_call(prompt)
        print(f"\nOversight: {response}\n")
        history.append(f"Oversight: {response}")

    print("Exiting oversight agent.")


# ---------------------------------------------------------------------------
# Control menu (shown after Ctrl-C detach)
# ---------------------------------------------------------------------------

def _control_menu(
    agent_thread: threading.Thread,
    done_event: threading.Event,
    result: Dict[str, Any],
    llm_call: Callable,
    broker: Optional["_HumanInputBroker"] = None,
) -> None:
    print("\nAgent still running in background.")
    print("Commands: [l] reattach to logs   [o] oversight agent   [s] status   [q] quit\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            cmd = "q"

        if cmd == "l":
            print("\nReattaching to logs...\n")
            try:
                while not done_event.is_set():
                    if broker:
                        prompt, pending = broker.poll()
                        if pending:
                            ans = input(prompt)
                            broker.answer(ans)
                            continue
                    done_event.wait(timeout=0.2)
            except KeyboardInterrupt:
                print("\n\nDetached again.")
                continue
            break

        elif cmd == "o":
            state = result.get("final_state") or {}
            if result.get("error"):
                state = dict(state)
                state["_exception"] = result["error"]
            if not done_event.is_set():
                print("Agent is still running — oversight will use partial state captured so far.")
                if broker:
                    prompt, pending = broker.poll()
                    if pending:
                        print(f"(Agent is waiting for your input: {prompt!r} — answer this first)")
                        ans = input(prompt)
                        broker.answer(ans)
                        continue
            _oversight_agent(llm_call, state)

        elif cmd == "s":
            print("Agent: running" if not done_event.is_set() else "Agent: finished")

        elif cmd == "q":
            print("Returning to shell. Agent thread will finish on its own.")
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Welcome to the FaaSr-Agent-System.")
    print("Please select an execution mode:")
    print("1. Full Pipeline (Request -> Deploy)")
    print("2. Workflow Generation Only (Request -> Workflow JSON)")
    print("3. Code Generation Only (Workflow JSON -> Code)")
    print("4. Workflow + Code Generation (Request -> Code)")
    print("5. Deployment Only (Code + JSON -> Deploy)")
    print("6. Code Generation + Deployment (Workflow JSON -> Deploy)")
    
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        if choice in ["1", "2", "3", "4", "5", "6"]:
            break
        print("Invalid choice, please enter a number from 1 to 6.")
        
    modes = {
        "1": "full",
        "2": "workflow_only",
        "3": "code_only",
        "4": "workflow_and_code",
        "5": "deploy_only",
        "6": "code_and_deploy"
    }
    
    mode = modes[choice]

    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    llm_call = make_llm_call(llm_provider)
    broker = _HumanInputBroker()
    app = build_orchestrator_graph(llm_call, mode=mode, input_fn=broker.ask)
    
    initial_state = {}

    if mode in ["full", "workflow_only", "workflow_and_code"]:
        user_request = input("\nEnter a natural language request:\n").strip()
        if not user_request:
            user_request = """
            Generate two csv files with a single column of randomly generated integers. Perform an element-wise addition of the corresponding elements from both 
            files. The resulting sums should be written into a single output csv file containing one column with the summed values.
            """
        initial_state["user_request"] = user_request

    if mode in ["code_only", "code_and_deploy"]:
        workflow_tasks_path = input("\nEnter path to workflow_tasks JSON: ").strip()
        faasr_workflow_path = input("Enter path to faasr_workflow JSON: ").strip()
        
        try:
            with open(workflow_tasks_path, "r") as f:
                initial_state["workflow_tasks"] = json.load(f)
            with open(faasr_workflow_path, "r") as f:
                initial_state["faasr_workflow"] = json.load(f)
        except Exception as e:
            print(f"Error loading JSON files: {e}")
            return
            
    if mode == "deploy_only":
        faasr_workflow_path = input("\nEnter path to faasr_workflow JSON: ").strip()
        functions_path = input("Enter path to functions folder: ").strip() # Needed if execute expects it.
        
        try:
            with open(faasr_workflow_path, "r") as f:
                initial_state["faasr_workflow"] = json.load(f)
            
            # Optionally check if functions path exists
            if not os.path.isdir(functions_path):
                print(f"Functions directory not found at: {functions_path}")
                return
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return

    print(f"\nStarting FaaSr-Agent-System in '{mode}' mode...")
    print("Press Ctrl+C to detach from logs (agent keeps running).\n")

    result: Dict[str, Any] = {"final_state": None, "error": None}
    done_event = threading.Event()

    def _run() -> None:
        try:
            result["final_state"] = app.invoke(initial_state)
        except Exception as exc:
            result["error"] = str(exc)
            result["final_state"] = {}
        finally:
            done_event.set()

    agent_thread = threading.Thread(target=_run, name="faasr-agent", daemon=False)
    agent_thread.start()

    detached = False
    try:
        while not done_event.is_set():
            prompt, pending = broker.poll()
            if pending:
                ans = input(prompt)
                broker.answer(ans)
            else:
                done_event.wait(timeout=0.2)
    except KeyboardInterrupt:
        detached = True
        print("\n\nDetached from logs.")

    if detached:
        _control_menu(agent_thread, done_event, result, llm_call, broker)
        agent_thread.join()

    final_state = result.get("final_state") or {}

    if result.get("error") or _has_errors(final_state):
        if result.get("error"):
            print(f"\nAgent error: {result['error']}")
        elif final_state.get("workflow_errors"):
            print("\nWorkflow agent errors:")
            print(final_state["workflow_errors"])
        try:
            ans = input("\nLaunch oversight agent? [y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            ans = "n"
        if ans in ("y", "yes"):
            if result.get("error"):
                final_state = dict(final_state)
                final_state["_exception"] = result["error"]
            _oversight_agent(llm_call, final_state)
    
    


if __name__ == "__main__":
    main()
