# src/faasr_ai/tui/sync_tracker.py
from __future__ import annotations

import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from framework.utils.enums import FunctionStatus
from framework.s3_client import FaaSrS3Client
from faasr_ai.utils.cache_manager import CacheStatus

logger = logging.getLogger(__name__)

STATUS_STYLE: dict[FunctionStatus, tuple[str, str]] = {
    FunctionStatus.PENDING: ("dim", "PENDING"),
    FunctionStatus.INVOKED: ("yellow", "INVOKED"),
    FunctionStatus.RUNNING: ("blue bold", "RUNNING"),
    FunctionStatus.COMPLETED: ("green bold", "COMPLETED"),
    FunctionStatus.FAILED: ("red bold", "FAILED"),
    FunctionStatus.SKIPPED: ("dim strike", "SKIPPED"),
    FunctionStatus.TIMEOUT: ("red", "TIMEOUT"),
    FunctionStatus.NOT_INVOKED: ("dim", "NOT INVOKED"),
}

_NOT_FETCHED = object()


class CodeCache:
    """Fetches artifacts from s3 in background thread"""

    def __init__(self, s3_client: FaaSrS3Client, workflow_name: str, cache_statuses: Optional[dict] = None):
        self._client = s3_client
        self._workflow_name = workflow_name
        self._cache_statuses = cache_statuses
        self._cache: dict[str, str | None] = {}
        self._fetching: set[str] = set()
        self._lock = threading.Lock()

    def get(self, fn_name: str) -> str | None | object:
        """Returns code string, None (not found), or _NOT_FETCHED (fetch pending)."""
        with self._lock:
            if fn_name in self._cache:
                return self._cache[fn_name]
            return _NOT_FETCHED

    def request_fetch(self, fn_name: str) -> None:
        """Kick off a background fetch if not already cached/fetching."""
        with self._lock:
            if fn_name in self._cache or fn_name in self._fetching:
                return
            self._fetching.add(fn_name)
        threading.Thread(target=self._fetch, args=(fn_name,), daemon=True).start()

    def _fetch(self, fn_name: str) -> None:
        code = _fetch_code_for_function(fn_name, self._client, self._cache_statuses, self._workflow_name)
        with self._lock:
            self._cache[fn_name] = code
            self._fetching.discard(fn_name)


class ActionView:
    """Tracks which action is being viewed; auto-follows unless user cycled."""

    def __init__(self, action_names: list[str]):
        self._names = action_names
        self._idx = 0
        self._auto = True
        self._lock = threading.Lock()

    @property
    def current(self) -> str | None:
        with self._lock:
            return self._names[self._idx] if self._names else None

    @property
    def index(self) -> int:
        with self._lock:
            return self._idx

    def next(self) -> None:
        with self._lock:
            if self._names:
                self._idx = (self._idx + 1) % len(self._names)
                self._auto = False

    def prev(self) -> None:
        with self._lock:
            if self._names:
                self._idx = (self._idx - 1) % len(self._names)
                self._auto = False

    def auto_select(self, statuses: dict[str, FunctionStatus]) -> None:
        """Follow the most active function unless user has manually navigated."""
        with self._lock:
            if not self._auto or not self._names:
                return
            # Prefer RUNNING
            for i, name in enumerate(self._names):
                if statuses.get(name) == FunctionStatus.RUNNING:
                    self._idx = i
                    return
            # Fall back to last COMPLETED
            for i in range(len(self._names) - 1, -1, -1):
                if statuses.get(self._names[i]) == FunctionStatus.COMPLETED:
                    self._idx = i
                    return


class ScrollState:
    """Per-action scroll offsets for both code and log panels."""

    CODE_PAGE = 30  # lines visible at once in code panel
    LOG_PAGE = 25   # lines visible at once in log panel

    def __init__(self):
        self._code: dict[str, int] = {}
        self._log: dict[str, int] = {}
        self._lock = threading.Lock()

    def get_code(self, fn_name: str) -> int:
        with self._lock:
            return self._code.get(fn_name, 0)

    def get_log(self, fn_name: str) -> int:
        with self._lock:
            return self._log.get(fn_name, 0)

    def scroll_code_up(self, fn_name: str, amount: int = 5) -> None:
        with self._lock:
            self._code[fn_name] = max(0, self._code.get(fn_name, 0) - amount)

    def scroll_code_down(self, fn_name: str, total_lines: int, amount: int = 5) -> None:
        with self._lock:
            max_offset = max(0, total_lines - self.CODE_PAGE)
            self._code[fn_name] = min(max_offset, self._code.get(fn_name, 0) + amount)

    def scroll_log_up(self, fn_name: str, amount: int = 5) -> None:
        with self._lock:
            self._log[fn_name] = max(0, self._log.get(fn_name, 0) - amount)

    def scroll_log_down(self, fn_name: str, amount: int = 5) -> None:
        """Increment log offset — clamped to actual content at render time."""
        with self._lock:
            self._log[fn_name] = self._log.get(fn_name, 0) + amount


def _keyboard_listener(
    view: ActionView, scroll: ScrollState, stop: threading.Event, code_cache: Optional[CodeCache]
) -> None:
    """Read arrow keypresses — ←/→ cycle actions, ↑/↓ scroll code."""
    try:
        import readchar
    except ImportError:
        return
    try:
        while not stop.is_set():
            key = readchar.readkey()
            fn = view.current
            if key in (readchar.key.RIGHT, "n"):
                view.next()
            elif key in (readchar.key.LEFT, "p"):
                view.prev()
            # Code scroll: ↑ / ↓
            elif key == readchar.key.UP:
                if fn:
                    scroll.scroll_code_up(fn)
            elif key == readchar.key.DOWN:
                if fn and code_cache:
                    result = code_cache.get(fn)
                    total = len(result.splitlines()) if isinstance(result, str) else 0
                    scroll.scroll_code_down(fn, total)
            # Log scroll: w / s
            elif key == "w":
                if fn:
                    scroll.scroll_log_up(fn)
            elif key == "s":
                if fn:
                    scroll.scroll_log_down(fn)
    except Exception:
        pass


def _fetch_code_for_function(
    fn_name: str,
    s3_client: FaaSrS3Client,
    cache_statuses: Optional[dict] = None,
    workflow_name: str = "",
) -> Optional[str]:
    if not cache_statuses:
        logger.debug("fetch skipped for %s: no cache_statuses provided", fn_name)
        return None
    if not workflow_name:
        logger.debug("fetch skipped for %s: no workflow_name provided", fn_name)
        return None

    entry = cache_statuses.get(fn_name)
    if entry is None:
        logger.debug("fetch skipped for %s: not present in cache_statuses", fn_name)
        return None
    if not entry.cache_key:
        logger.debug("fetch skipped for %s: cache_key is empty", fn_name)
        return None

    key = f"{workflow_name}/_cache/{fn_name}/{entry.cache_key}/code_raw.py"
    logger.debug("fetching code for %s from s3 key: %s", fn_name, key)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
            temp_path = f.name
        s3_client.download_object(key, temp_path)
        with open(temp_path, "r") as f:
            content = f.read()
        Path(temp_path).unlink()
        logger.debug("fetch succeeded for %s (%d chars)", fn_name, len(content))
        return content
    except Exception as e:
        logger.warning("fetch failed for %s (key=%s): %s", fn_name, key, e)
        return None


def _build_status_table(
    statuses: dict[str, FunctionStatus],
    workflow_name: str,
    viewed_idx: int,
    cache_statuses: Optional[dict] = None,
) -> Table:
    table = Table(
        title=f"[bold]{workflow_name}[/bold]",
        expand=True,
        show_lines=True,
    )
    table.add_column("", width=1)  # selection indicator
    table.add_column("Action", style="cyan", ratio=2)
    table.add_column("Status", justify="center", ratio=1)
    if cache_statuses:
        table.add_column("Cache", justify="center", ratio=1)

    for i, (name, status) in enumerate(statuses.items()):
        style, label = STATUS_STYLE.get(status, ("", str(status.value)))
        indicator = "▶" if i == viewed_idx else " "
        if cache_statuses:
            entry = cache_statuses.get(name)
            if entry is None or status not in (FunctionStatus.COMPLETED, FunctionStatus.FAILED):
                cache_label = Text("-", style="dim")
            elif entry.status == CacheStatus.HIT:
                cache_label = Text("HIT", style="green")
            elif entry.status == CacheStatus.INVALID:
                cache_label = Text("INVALID", style="yellow")
            else:
                cache_label = Text("MISS", style="red")
            table.add_row(indicator, name, Text(label, style=style), cache_label)
        else:
            table.add_row(indicator, name, Text(label, style=style))

    return table


def _build_detail_panel(
    fn_name: str,
    status: FunctionStatus,
    runner,
    code_cache: Optional[CodeCache],
    scroll: Optional[ScrollState] = None,
) -> Panel:
    """Build the log (+ code if completed) panel for a given action."""
    # Logs
    try:
        logs = runner.get_function_logs_content(fn_name)
        log_lines = logs.splitlines() if logs else []
    except Exception:
        log_lines = ["(unable to read logs)"]

    log_total = len(log_lines)
    log_page = ScrollState.LOG_PAGE
    if scroll:
        log_offset = min(scroll.get_log(fn_name), max(0, log_total - log_page))
    else:
        log_offset = max(0, log_total - log_page)  # default: tail
    log_visible = log_lines[log_offset: log_offset + log_page] if log_lines else ["(no logs yet)"]
    log_top = log_offset + 1
    log_bot = min(log_offset + log_page, log_total)
    log_scroll_info = f"lines {log_top}-{log_bot}/{log_total}  w/s scroll" if log_total > log_page else ""
    log_title = f"Logs: {fn_name}" + (f"  [{log_scroll_info}]" if log_scroll_info else "")
    log_panel = Panel("\n".join(log_visible), title=log_title, border_style="blue", expand=True)

    # Code (only for completed functions with a cache HIT)
    if status == FunctionStatus.COMPLETED and code_cache is not None:
        code_cache.request_fetch(fn_name)
        result = code_cache.get(fn_name)
        if result is _NOT_FETCHED:
            code_panel = Panel("⏳ Fetching code...", title=f"Code: {fn_name}", border_style="dim", expand=True)
            return Group(log_panel, code_panel)
        elif result is not None:
            lines = result.splitlines()
            total = len(lines)
            page = ScrollState.CODE_PAGE
            offset = scroll.get_code(fn_name) if scroll else 0
            visible = lines[offset: offset + page]
            top = offset + 1
            bot = min(offset + page, total)
            scroll_info = f"lines {top}-{bot}/{total}  ↑↓ scroll"
            try:
                code_renderable = Syntax(
                    "\n".join(visible), "python", theme="monokai",
                    line_numbers=True, start_line=offset + 1,
                )
            except Exception:
                code_renderable = "\n".join(visible)
            code_panel = Panel(
                code_renderable,
                title=f"Code: {fn_name}  [{scroll_info}]",
                border_style="green",
                expand=True,
            )
            return Group(log_panel, code_panel)

    return log_panel


def _progress_text(statuses: dict[str, FunctionStatus]) -> str:
    completed = sum(1 for s in statuses.values() if s == FunctionStatus.COMPLETED)
    total = len(statuses)
    return f"Progress: {completed}/{total} actions complete"


def run_sync_tracker(
    runner,
    s3_client: Optional[FaaSrS3Client] = None,
    faasr_json: Optional[dict[str, Any]] = None,
    cache_statuses: Optional[dict] = None,
) -> None:
    """Live-updating Rich TUI that polls WorkflowRunner until monitoring completes.

    Arrow keys (← →) or n/p cycle through actions in the detail panel.
    """
    console = Console()
    workflow_name = getattr(runner, "workflow_name", "Workflow")
    invocation_id = getattr(runner, "invocation_id", "N/A")

    # Snapshot action names once (order matters for cycling)
    initial_statuses = runner.get_function_statuses()
    action_names = list(initial_statuses.keys())

    view = ActionView(action_names)
    code_cache = CodeCache(s3_client, workflow_name, cache_statuses) if s3_client else None
    scroll = ScrollState()

    # Start keyboard listener
    stop_event = threading.Event()
    kb_thread = threading.Thread(
        target=_keyboard_listener, args=(view, scroll, stop_event, code_cache), daemon=True
    )
    kb_thread.start()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="status", ratio=1),
        Layout(name="detail", ratio=2),
    )

    start_time = time.time()

    try:
        with Live(layout, console=console, refresh_per_second=2, screen=True):
            while not runner.monitoring_complete:
                statuses = runner.get_function_statuses()
                elapsed = time.time() - start_time

                # Trigger background code fetch for all completed actions
                if code_cache:
                    for name, st in statuses.items():
                        if st == FunctionStatus.COMPLETED:
                            code_cache.request_fetch(name)

                # Auto-follow active function (unless user cycled)
                view.auto_select(statuses)
                viewed_fn = view.current
                viewed_idx = view.index

                # Header
                layout["header"].update(
                    Panel(
                        f"[bold]{workflow_name}[/bold]  |  "
                        f"Invocation: {invocation_id}  |  "
                        f"{_progress_text(statuses)}  |  "
                        f"[dim]← → to cycle actions[/dim]",
                        style="bold",
                    )
                )

                # Status table (with selection indicator)
                layout["status"].update(_build_status_table(statuses, workflow_name, viewed_idx, cache_statuses))

                # Detail panel (log + code for viewed action)
                if viewed_fn:
                    layout["detail"].update(
                        _build_detail_panel(
                            viewed_fn,
                            statuses.get(viewed_fn, FunctionStatus.PENDING),
                            runner,
                            code_cache,
                            scroll,
                        )
                    )
                else:
                    layout["detail"].update(
                        Panel("Waiting for a function to start...", title="Detail", border_style="dim")
                    )

                # Footer
                layout["footer"].update(
                    Panel(
                        f"Elapsed: {elapsed:.0f}s  |  "
                        f"[dim]← → cycle actions  ·  ↑↓ scroll code  ·  w/s scroll logs  ·  Ctrl+C stop[/dim]",
                        style="dim",
                    )
                )

                time.sleep(0.5)
    finally:
        stop_event.set()

    print_executive_summary(runner, s3_client=s3_client, cache_statuses=cache_statuses)


def print_executive_summary(
    runner,
    s3_client: Optional[FaaSrS3Client] = None,
    cache_statuses: Optional[dict] = None,
) -> None:
    """Print a final executive summary after the workflow completes."""
    console = Console()
    statuses = runner.get_function_statuses()
    workflow_name = getattr(runner, "workflow_name", "Workflow")

    all_ok = all(
        s in {FunctionStatus.COMPLETED, FunctionStatus.NOT_INVOKED}
        for s in statuses.values()
    )

    console.print()
    console.rule("[bold]Executive Summary[/bold]")
    console.print()
    console.print(
        _build_status_table(statuses, workflow_name, viewed_idx=-1, cache_statuses=cache_statuses)
    )
    console.print()

    if all_ok:
        console.print("[green bold]Workflow completed successfully.[/green bold]")
    else:
        console.print("[red bold]Workflow completed with failures.[/red bold]")

    # Per-function details
    for name, status in statuses.items():
        if status == FunctionStatus.FAILED:
            try:
                logs = runner.get_function_logs_content(name)
                if logs:
                    console.print(Panel(logs, title=f"Failure Logs: {name}", border_style="red"))
            except Exception:
                console.print(f"[red]Could not retrieve logs for {name}[/red]")

        elif status == FunctionStatus.COMPLETED and s3_client:
            code = _fetch_code_for_function(name, s3_client, cache_statuses, workflow_name)
            if code:
                try:
                    logs = runner.get_function_logs_content(name)
                    log_tail = "\n".join(logs.splitlines()[-20:]) if logs else ""
                except Exception:
                    log_tail = ""

                panels = []
                if log_tail:
                    panels.append(Panel(log_tail, title=f"Logs: {name}", border_style="blue", expand=True))
                try:
                    panels.append(
                        Panel(
                            Syntax(code, "python", theme="monokai", line_numbers=True),
                            title=f"Code: {name}",
                            border_style="green",
                            expand=True,
                        )
                    )
                except Exception:
                    panels.append(Panel(code, title=f"Code: {name}", border_style="green", expand=True))

                if panels:
                    console.print(Group(*panels))

    console.print()
    input("Press Enter to continue...")
