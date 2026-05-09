"""
Structured step-by-step logging for ResearchPilot demo.

Logs each step's input and output in a clear, demo-friendly format.
Output: logs/{timestamp}_pipeline.log
"""

import json
import os
from datetime import datetime


class StepLogger:
    """Captures and formats step input/output for demonstration."""

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join("logs", f"{ts}_pipeline.log")
        self.step_count = 0
        self._write(f"{'='*80}\nRESEARCHPILOT PIPELINE LOG — {datetime.now().isoformat()}\n{'='*80}\n")

    def _write(self, text: str) -> None:
        """Append text to log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)

    def log_step(self, step_num: int, step_name: str, step_input: dict, step_output: dict) -> None:
        """
        Log a single pipeline step with formatted input and output.

        Args:
            step_num: Step number (1-6)
            step_name: Human-readable name (e.g., "Query Parsing")
            step_input: dict of inputs (keys extracted from state)
            step_output: dict of outputs (keys extracted from state)
        """
        self.step_count += 1

        # Format input and output as JSON for clarity
        input_json = json.dumps(step_input, indent=2, default=str)
        output_json = json.dumps(step_output, indent=2, default=str)

        log_entry = f"""
{'─'*80}
STEP {step_num}: {step_name}
{'─'*80}

INPUT:
{input_json}

OUTPUT:
{output_json}

"""
        self._write(log_entry)
        print(f"  ✓ Step {step_num} logged: {self.log_file}")

    def log_final_summary(self, state: dict) -> None:
        """Log final pipeline summary."""
        summary = {
            "total_papers_fetched": len(state.get("papers", [])),
            "top_papers_selected": len(state.get("top_papers", [])),
            "papers_deep_read": len(state.get("deep_reads", [])),
            "topic": state.get("topic", "Unknown"),
            "field": state.get("field", "Unknown"),
            "clusters": state.get("synthesis", {}).get("clusters", []),
            "report_length_chars": len(state.get("report_markdown", "")),
        }

        summary_json = json.dumps(summary, indent=2, default=str)
        log_entry = f"""
{'='*80}
FINAL SUMMARY
{'='*80}

{summary_json}

Report saved to: output/lit_review.md
Log file: {self.log_file}

{'='*80}
"""
        self._write(log_entry)
        print(f"\n✓ Pipeline complete. Full log: {self.log_file}")


def extract_step_input(state: dict, step_num: int) -> dict:
    """Extract relevant input dict for a given step."""
    if step_num == 1:
        return {"raw_query": state.get("raw_query", "")}
    elif step_num == 2:
        return {"search_strings": state.get("search_strings", [])}
    elif step_num == 3:
        return {"papers": len(state.get("papers", [])), "top_k": 10}
    elif step_num == 4:
        return {"top_papers_count": len(state.get("top_papers", []))}
    elif step_num == 5:
        return {"deep_reads_count": len(state.get("deep_reads", []))}
    elif step_num == 6:
        return {
            "synthesis_clusters": len(state.get("synthesis", {}).get("clusters", [])),
            "deep_reads": len(state.get("deep_reads", [])),
        }
    return {}


def extract_step_output(state: dict, step_num: int) -> dict:
    """Extract relevant output dict for a given step."""
    if step_num == 1:
        return {
            "topic": state.get("topic", ""),
            "field": state.get("field", ""),
            "keywords": state.get("keywords", [])[:5],
            "search_strings_count": len(state.get("search_strings", [])),
        }
    elif step_num == 2:
        return {
            "papers_fetched": len(state.get("papers", [])),
            "top_3_papers": [p.get("title", "?")[:60] for p in state.get("papers", [])[:3]],
            "authors_profiled": len(state.get("authors", {})),
        }
    elif step_num == 3:
        return {
            "top_papers_selected": len(state.get("top_papers", [])),
            "top_3_titles": [p.get("title", "?")[:60] for p in state.get("top_papers", [])[:3]],
            "researchers_profiled": len(state.get("notable_researchers", [])),
        }
    elif step_num == 4:
        reads = state.get("deep_reads", [])
        return {
            "papers_read": len(reads),
            "avg_abstract_chars": sum(len(r.get("core_argument", "")) for r in reads) // max(len(reads), 1),
            "first_paper_core_argument": reads[0].get("core_argument", "")[:100] if reads else "N/A",
        }
    elif step_num == 5:
        synth = state.get("synthesis", {})
        return {
            "agreements_extracted": len(synth.get("agreements", [])),
            "contradictions_found": len(synth.get("contradictions", [])),
            "gaps_identified": len(synth.get("gaps", [])),
            "clusters_formed": synth.get("clusters", []),
        }
    elif step_num == 6:
        return {
            "report_markdown_chars": len(state.get("report_markdown", "")),
            "report_words": len(state.get("report_markdown", "").split()),
            "first_100_chars": state.get("report_markdown", "")[:100],
        }
    return {}
