"""
Utility classes and functions shared across the teval pipeline.

Includes:
  - Timer : context manager for measuring block execution time
  - TimingRegistry : module-level collector for producing timing summaries
  - configure_timing / configure_logging : called once at startup from __main__
  - print_timing_summary : call once at the end of main()
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Global timing state. Set once at startup via configure_timing()
_TIMING_MODE: str = "none"  # "none" | "simple" | "verbose"

TIMING_CATEGORIES: Dict[str, str] = {
    "discovery":     "Domain Discovery",
    "loading":       "Input Reading",
    "output":        "Output Writing",
    "metrics":       "Metrics Calculation",
    "visualization": "Visualization",
    "general":       "General",
}


@dataclass
class _TimingEntry:
    category: str
    label: str
    elapsed: float
    failed: bool = False

class _TimingRegistry:
    """
    Stores all Timer entries for the current run.
    Produces a grouped summary table on demand.
    """

    def __init__(self):
        """Initialise the TimingRegistry."""
        self._entries: List[_TimingEntry] = []

    def record(self, entry: _TimingEntry) -> None:
        """Record a timing entry."""
        self._entries.append(entry)

    def clear(self) -> None:
        """Clear all recorded timing entries."""
        self._entries.clear()

    def print_summary(self) -> None:
        """
        Print a grouped timing summary table to the logger.
        Groups entries by category, shows per-entry and per-group totals.
        """
        if not self._entries:
            logger.info("No timing data recorded.")
            return

        # Build grouped structure
        groups: Dict[str, List[_TimingEntry]] = {}
        for e in self._entries:
            groups.setdefault(e.category, []).append(e)

        total = sum(e.elapsed for e in self._entries)
        width = 62

        lines = [
            "",
            "TIMING SUMMARY",
            "═" * width,
        ]

        # Print in a fixed category order, then any extras
        ordered_cats = list(TIMING_CATEGORIES.keys()) + [
            c for c in groups if c not in TIMING_CATEGORIES
        ]

        # Collapse categories with many entries. Categories with more
        # than COLLAPSE_THRESHOLD entries show an aggregate line instead.
        COLLAPSE_THRESHOLD = 10

        for cat_key in ordered_cats:
            if cat_key not in groups:
                continue
            entries = groups[cat_key]
            cat_label = TIMING_CATEGORIES.get(cat_key, cat_key.title())
            cat_total = sum(e.elapsed for e in entries)
            n_failed  = sum(1 for e in entries if e.failed)

            lines.append(f" {cat_label:<46} {_fmt(cat_total):>8}")

            if len(entries) > COLLAPSE_THRESHOLD:
                # Collapsed view: show count, slowest entry, failure count
                slowest = max(entries, key=lambda e: e.elapsed)
                label = (slowest.label[:38] + "...") if len(slowest.label) > 39 else slowest.label
                fail_str = f"  ({n_failed} failed)" if n_failed else ""
                lines.append(
                    f"   ├─ {len(entries)} entries — slowest: {label} "                    f"({_fmt(slowest.elapsed)}){fail_str}"                )
                lines.append(f"   └─ (use timing: verbose to see all entries)")
            else:
                for i, e in enumerate(entries):
                    connector = "└─" if i == len(entries) - 1 else "├─"
                    status = " X" if e.failed else ""
                    label = (e.label[:42] + "...") if len(e.label) > 43 else e.label
                    lines.append(f"   {connector} {label:<44}{_fmt(e.elapsed):>8}{status}")

        lines += [
            "─" * width,
            f" {'Total':<46} {_fmt(total):>8}",
            "═" * width,
            "",
        ]
        logger.info("\n".join(lines))


def _fmt(seconds: float) -> str:
    """Format seconds as 'm:ss.s' if ≥60s, else 'X.XXs'."""
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:04.1f}s"
    return f"{seconds:.2f}s"


# Module-level registry instance
_registry = _TimingRegistry()


# Configuration helpers

def configure_timing(mode: Optional[str]) -> None:
    """
    Set the global timing mode for this run.

    Parameters
    ----------
    mode : "none" | "simple" | "verbose" | None
        none    — timing is disabled entirely (Timer is a no-op)
        simple  — timings are recorded silently; a summary is printed at the end
        verbose — start/end of each block is logged immediately and recorded
    """
    global _TIMING_MODE
    _TIMING_MODE = (mode or "none").lower()
    _registry.clear()
    logger.debug(f"Timing mode set to: '{_TIMING_MODE}'")

def configure_logging(level: Optional[str]) -> None:
    """
    Set the root logger level for the entire teval run.

    Parameters
    ----------
    level : "DEBUG" | "INFO" | "WARNING" | "ERROR"
    """
    level_str = (level or "INFO").upper()
    numeric = getattr(logging, level_str, logging.INFO)
    logging.getLogger().setLevel(numeric)
    logging.getLogger("teval").setLevel(numeric)
    logger.debug(f"Logging level set to: {level_str}")

def print_timing_summary() -> None:
    """Print the collected timing summary. Call once at the end of main()."""
    if _TIMING_MODE != "none":
        _registry.print_summary()


class Timer:
    """
    Timing manager that measures execution time of code.

    Behavior depends on the global timing mode:
      none    : does nothing (zero overhead)
      simple  : records elapsed time silently for the end-of-run summary
      verbose : logs start/end immediately AND records for the summary

    Parameters
    ----------
    label    : Description of the block (shown in logs/summary)
    category : Timing category key (see TIMING_CATEGORIES). Controls grouping
               in the summary table.
               Allowed values: "discovery", "loading", "output",
                               "metrics", "visualization", "general"

    Example
    -------
    >>> with Timer("Load hydrofabric for CONUS", category="loading"):
    ...     gdf = gpd.read_file(...)
    """

    def __init__(self, label: str, category: str = "general"):
        """Initialise the TimingRegistry."""
        self.label = label
        self.category = category
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer when entering the context."""
        if _TIMING_MODE == "none":
            return self
        self._start = time.perf_counter()
        if _TIMING_MODE == "verbose":
            cat_display = TIMING_CATEGORIES.get(self.category, self.category)
            logger.info(f"  ┌─ START  [{cat_display}] {self.label}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer and record the elapsed time when exiting the context."""
        if _TIMING_MODE == "none":
            return
        self.elapsed = time.perf_counter() - self._start
        failed = exc_type is not None

        if _TIMING_MODE == "verbose":
            cat_display = TIMING_CATEGORIES.get(self.category, self.category)
            status = "FAILED" if failed else "done  "
            logger.info(
                f"  └─ {status} [{cat_display}] {self.label} "
                f"({_fmt(self.elapsed)})"
            )

        _registry.record(_TimingEntry(
            category=self.category,
            label=self.label,
            elapsed=self.elapsed,
            failed=failed,
        ))
        
        return None