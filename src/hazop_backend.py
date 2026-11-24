"""hazop_backend.py - Haznova HAZOP study backend."""

import json
import ast
import base64
import hashlib
import io
import math
import os
import random
import re
import socket
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import webbrowser
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from hazop_prompts import (
    chair_effect_definition_prompt,
    chair_merge_prompt,
    chair_refine_prompt,
    compact_retry_prompt,
    configure_prompt_helpers,
    exhaustive_round_two_json_shape,
    expert_round_one_prompt,
    expert_round_two_exhaustive_prompt,
    expert_round_two_new_scenario_review_prompt,
    expert_round_two_refinement_prompt,
    pid_image_parser_prompt,
    pid_image_parser_system_prompt,
    planning_prompt,
    process_understanding_prompt,
    report_prompt,
    round_one_json_shape,
    round_two_new_scenario_review_json_shape,
    round_two_refinement_json_shape,
    system_prompt_for,
)

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
PERFORMANCE_DIR = ROOT / "performance"
HOST = "127.0.0.1"
DEFAULT_PORT = 8010
REQUEST_TIMEOUT_SECONDS = 600
MAX_DOCUMENT_CHARS = 300000
MAX_CONTEXT_CHARS = 300000
MAX_NODES_DEFAULT = 4
MAX_NODES_HARD_LIMIT = 10
MAX_ROWS_PER_NODE = 80
MAX_CAUSES_PER_MATRIX_ITEM = 4
ROUND_TWO_REVIEW_BATCH_SIZE = 10
ROUND_TWO_REFINEMENT_BATCH_SIZE = 6
ROUND_TWO_BATCH_RETRY_ATTEMPTS = 3
ROUND_THREE_BATCH_SIZE = 4

# Round 2 separates scenario validity from safeguard identification.  The
# first four checks determine whether the deviation-cause-consequence scenario
# is valid and distinct.  Safeguard/action quality is audited independently and
# must never turn a credible scenario into Agree, Partially agree, or Disagree.
# This prevents an absent safeguard from being mistaken for an invalid hazard.
ROUND_TWO_CORE_CRITERIA = (
    "node_guideword_relevance",
    "physical_causal_credibility",
    "consequence_validity",
    "scenario_distinctness",
)
ROUND_TWO_NON_DECISION_CRITERIA = (
    "safeguard_action_validity",
)
ROUND_TWO_REVIEW_CRITERIA = (
    ROUND_TWO_CORE_CRITERIA + ROUND_TWO_NON_DECISION_CRITERIA
)
ROUND_TWO_DECISION_METHOD = {
    "name": "Literature-grounded HAZOP technical review and consensus protocol",
    "version": "2026-11-06-three-outcome-v2",
    "hazop_basis": [
        "IEC 61882 guideword-based examination",
        "DOE-HDBK-1224-2024 deviation-cause-consequence-safeguard-action chain",
    ],
    "consensus_basis": [
        "RAND/UCLA independent rating, controlled discussion, and re-rating",
        "ChatEval role-diverse multi-agent evaluation",
        "ReConCILE multi-agent discussion with explicit explanations",
    ],
    "literature_references": [
        {
            "citation": "IEC 61882:2016, Hazard and operability studies (HAZOP studies) - Application guide",
            "url": "https://webstore.iec.ch/en/publication/24321",
        },
        {
            "citation": "DOE-HDBK-1224-2024, Hazard and Accident Analysis",
            "url": "https://www.energy.gov/documents/doe-hdbk-1224-2024pdf",
        },
        {
            "citation": "Fitch et al., The RAND/UCLA Appropriateness Method User's Manual",
            "url": "https://www.rand.org/pubs/monograph_reports/MR1269.html",
        },
        {
            "citation": "Chan et al. (2024), ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate",
            "url": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/25cc3adf8c85f7c70989cb8a97a691a7-Abstract-Conference.html",
        },
        {
            "citation": "Chen et al. (2024), ReConCILE: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs",
            "url": "https://aclanthology.org/2024.acl-long.381/",
        },
    ],
    "criterion_ratings": ["pass", "revision_required", "fail"],
    "decision_rule": {
        "disagree": "one or more scenario-validity checks fail, including duplicate or non-credible scenarios",
        "partially_agree": "no scenario-validity check fails, but at least one scenario field requires a material correction",
        "agree": "all scenario-validity checks pass without material correction",
        "safeguard_independence": "safeguard/action findings are reported separately and do not determine the scenario decision",
    },
    "aggregation_rule": (
        "Agreement is reported as consistency, not technical truth; the Chair "
        "adjudicates contested evidence rather than accepting a simple majority."
    ),
    "review_unit": (
        "One unique guideword-deviation-cause scenario after Chair consolidation and "
        "deterministic preservation of unmatched expert candidates, with all matching "
        "Round 1 source-agent identifiers retained as lineage."
    ),
    "round_2a_batch_size": ROUND_TWO_REVIEW_BATCH_SIZE,
    "missing_review_rule": (
        "After three attempts an omitted assignment remains blank and is reported as "
        "incomplete coverage; it is never converted to agreement or disagreement."
    ),
    "round_2b_entry_rule": (
        "Round 2B is triggered for every scenario that is not unanimous Agree, "
        "including unanimous Partially agree, unanimous Disagree, differing decisions, "
        "or conflicting scenario-validity criterion ratings. Missing reviews alone do not create a dispute."
    ),
    "round_2b_conclusion_rule": (
        "After discussion, one panel conclusion is always reported as Agree, Partially agree, "
        "or Disagree when at least one completed review exists. Individual decisions and dissent "
        "remain stored; no mixed outcome category is used."
    ),
    "round_2b_panel_aggregation": (
        "A unanimous category is reported when all completed reviews match; otherwise the unique "
        "plurality category is reported. An exact tie is conservatively reported as Partially agree. "
        "Missing reviews are excluded from the vote and reported separately as incomplete coverage. "
        "This categorical panel summary is an auditable discussion result, not a substitute for the "
        "Chair's evidence-grounded Round 3 technical adjudication."
    ),
}
ROUND_THREE_DECISION_METHOD = {
    "name": "Chair final worksheet adjudication using four scenario-validity checks and a separate safeguard audit",
    "version": "2026-11-06-three-outcome-v2",
    "criteria": list(ROUND_TWO_REVIEW_CRITERIA),
    "criterion_ratings": ["pass", "revision_required", "fail"],
    "decision_rule": dict(ROUND_TWO_DECISION_METHOD["decision_rule"]),
    "worksheet_treatment": {
        "chair_agree": "Retained",
        "chair_partially_agree": "Retained after the required correction or open action is recorded",
        "chair_disagree": "Rejected",
        "chair_proposal": "New only after targeted review by every panel member",
        "duplicate": "Rejected—duplicate, with the retained scenario ID recorded",
        "unlinked": "Rejected because no Round 2 panel basis exists",
        "panel_consensus_override": "allowed only with explicit case evidence and Chair resolution",
    },
    "authority_rule": (
        "Round 2 provides categorical panel advice. The Chair is the final technical adjudicator and must "
        "apply the same four scenario-validity checks to the P&ID, node design intent, process description, "
        "and procedures. Safeguards are audited independently from scenario validity. Any departure from "
        "the panel conclusion requires explicit case evidence and a stored Chair resolution. The backend "
        "derives the official Chair decision from the four scenario-validity checks."
    ),
    "round_two_panel_classification": {
        "agree": "the post-discussion panel conclusion is Agree",
        "partially_agree": "the post-discussion panel conclusion is Partially agree",
        "disagree": "the post-discussion panel conclusion is Disagree",
        "coverage": "review completeness and individual dissent are reported separately from the three-category conclusion",
    },
    "literature_references": list(ROUND_TWO_DECISION_METHOD["literature_references"]),
}

HAZOP_GUIDEWORDS = [
    ("No", "Complete negation of the design intent"),
    ("Less", "Quantitative decrease from design intent"),
    ("More", "Quantitative increase from design intent"),
    ("Part of", "Qualitative decrease - only part of the intention is achieved"),
    ("As well as", "Qualitative increase - something additional occurs"),
    ("Reverse", "Logical opposite of the design intent"),
    ("Other than", "Complete substitution - different material, route, or condition"),
]
HAZOP_PARAMETERS = ["flow", "pressure", "temperature", "level", "composition"]

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=12)


class JobStopped(Exception):
    pass

PROVIDERS = {"auto", "anthropic", "openai", "gemini", "ollama"}
AGENTS = [
    ("panel_chair", "Panel Chair", "HAZOP facilitator and workflow controller"),
    ("pid_image_parser", "P&ID Image Parser", "Vision parsing of P&ID symbols, tags, lines, and instruments"),
    ("process_engineer", "Process Engineer", "Process design, physical credibility, causes"),
    ("instrumentation_control", "Instrumentation & Control Engineer", "Control loops, alarms, interlocks, SIS"),
    ("safety_consequence", "Safety Engineer", "Fire, explosion, toxic, environmental consequences"),
    ("maintenance_reliability", "Maintenance Engineer", "Failure modes, fouling, corrosion, inspection"),
    ("mechanical_engineer", "Mechanical Engineer", "Equipment, piping, vessels, relief devices, mechanical design"),
    (
        "scribe_report",
        "Scribe / Report Agent",
        "Narrative report summarization; worksheet and action-register formatting are deterministic",
    ),
]
AGENT_IDS = [agent_id for agent_id, _, _ in AGENTS]
PANEL_AGENT_IDS = [
    "panel_chair",
    "process_engineer",
    "instrumentation_control",
    "safety_consequence",
    "maintenance_reliability",
    "mechanical_engineer",
]

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5.5",
    "gemini": "gemini-3.7-flash",
    "openai": "gpt-6.1-sol",
    "ollama": "gemma4",
}
AVAILABLE_MODELS = {
    "anthropic": [
        "claude-sonnet-5.5",
        "claude-opus-5.5",
    ],
    "gemini": [
        "gemini-3.7-flash",
        "gemini-3.1-pro-preview",
    ],
    "openai": [
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "gpt-6-luna",
    ],
    "ollama": [
        "gemma4",
        "qwen3.5",
        "llama4",
    ],
}

DEFAULT_TEMPERATURE = 1.0
DEFAULT_VISION_TEMPERATURE = 0.0
THINKING_LEVELS = {"auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"}
GEMINI_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
ANTHROPIC_THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 6144,
    "xhigh": 7168,
}
AUTO_PROVIDER_ORDER = {
    "panel_chair": ["openai", "anthropic", "gemini", "ollama"],
    "pid_image_parser": ["openai", "gemini", "anthropic", "ollama"],
    "process_engineer": ["openai", "anthropic", "gemini", "ollama"],
    "instrumentation_control": ["anthropic", "openai", "gemini", "ollama"],
    "maintenance_reliability": ["anthropic", "openai", "gemini", "ollama"],
    "safety_consequence": ["anthropic", "openai", "gemini", "ollama"],
    "mechanical_engineer": ["anthropic", "openai", "gemini", "ollama"],
    "scribe_report": ["openai", "anthropic", "gemini", "ollama"],
}
PROVIDER_MIN_INTERVAL_SECONDS = {"gemini": 6.0}
PROVIDER_RATE_LIMIT_LOCKS = {"gemini": threading.Lock()}
PROVIDER_NEXT_ALLOWED_AT = {"gemini": 0.0}
PROVIDER_ACTIVE_REQUEST_LIMITS = {"gemini": threading.Semaphore(1)}
GEMINI_TEXT_OUTPUT_TOKENS = 8192
GEMINI_VISION_OUTPUT_TOKENS = 16384
GEMINI_UNAVAILABLE_FOR_THIS_APP = {"gemini-2.5-pro", "models/gemini-2.5-pro"}

TEXT_MODEL_FALLBACKS: Dict[str, List[str]] = {}

VISION_MODEL_FALLBACKS: Dict[str, List[str]] = {}
# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dumps(data: Any) -> bytes:
    return json.dumps(json_safe(data), ensure_ascii=True).encode("utf-8")


def short_hash(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def estimated_tokens(text: Any) -> int:
    return int(math.ceil(len(str(text or "")) / 4.0))


def backend_source_hash() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()



def qualitative_agent_outputs(outputs: Any) -> List[Dict[str, Any]]:
    removed_role = ("ri" + "sk", "ana" + "lyst")
    filtered: List[Dict[str, Any]] = []
    for agent in outputs or []:
        if not isinstance(agent, dict):
            continue
        name = normalize_whitespace(agent.get("agent_name") or agent.get("name") or agent.get("id")).lower()
        if all(part in name for part in removed_role):
            continue
        filtered.append(agent)
    return filtered


def clamp_temperature(value: Any, default: float = DEFAULT_TEMPERATURE) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        temperature = default
    return max(0.0, min(2.0, temperature))


def default_agent_temperature(agent_id: str) -> float:
    return DEFAULT_VISION_TEMPERATURE if agent_id == "pid_image_parser" else DEFAULT_TEMPERATURE


def default_agent_thinking_level(agent_id: str) -> str:
    # Apply the same deliberate reasoning setting to every role so the
    # generated run metadata and every provider call use a consistent default.
    return "high"


def normalize_thinking_level(value: Any) -> str:
    level = normalize_whitespace(value).lower()
    return level if level in THINKING_LEVELS else "auto"


def agent_thinking_level(agent_id: str, value: Any) -> str:
    level = normalize_thinking_level(value)
    if level == "auto":
        return default_agent_thinking_level(agent_id)
    return level


def openai_supports_reasoning_effort(model: str) -> bool:
    model_lower = normalize_whitespace(model).lower()
    return model_lower.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def openai_supports_temperature(model: str) -> bool:
    model_lower = normalize_whitespace(model).lower()
    # GPT-5/GPT-6 reasoning models and the o-series reject the sampling
    # temperature parameter in the Responses API.
    # Reasoning effort is the supported control surface for these calls.
    if model_lower.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        return False
    return True


def openai_reasoning_effort(level: str, model: str = "") -> Optional[str]:
    level = normalize_thinking_level(level)
    model_lower = normalize_whitespace(model).lower()
    if level == "auto":
        return None
    if model_lower.startswith(("gpt-6-astra", "gpt-6.1-sol")):
        if level in {"none", "minimal"}:
            return "low"
        return level if level in {"low", "medium", "high", "xhigh", "max"} else None
    if model_lower.startswith(("gpt-5.6", "gpt-6.1-terra", "gpt-6-luna")):
        if level == "minimal":
            return "low"
        return level if level in {"none", "low", "medium", "high", "xhigh", "max"} else None
    if level == "none":
        return None
    if level == "xhigh":
        return "high"
    if level == "max":
        return "high"
    return level if level in {"minimal", "low", "medium", "high"} else None


def openai_responses_only(model: str) -> bool:
    """Keep current GPT-6.1 Sol calls on the documented Responses route."""
    return normalize_whitespace(model).lower().startswith("gpt-6.1-sol")


def gemini_thinking_level(level: str) -> Optional[str]:
    level = normalize_thinking_level(level)
    if level == "xhigh":
        return "high"
    if level in GEMINI_THINKING_LEVELS:
        return level
    return None


def gemini_generate_content_thinking_config(level: str, model: str) -> Optional[Dict[str, Any]]:
    level = normalize_thinking_level(level)
    if level == "auto":
        return {"thinkingBudget": -1} if "2.5" in normalize_whitespace(model).lower() else None
    if level == "none":
        level = "minimal"
    model_lower = normalize_whitespace(model).lower()
    if model_lower.startswith("gemini-3"):
        thinking = "high" if level == "xhigh" else level
        if thinking in {"minimal", "low", "medium", "high"}:
            return {"thinkingLevel": thinking}
        return None
    if "2.5" in model_lower:
        if "pro" in model_lower and level == "minimal":
            return {"thinkingBudget": 128}
        budgets = {
            "minimal": 0,
            "low": 1024,
            "medium": 8192,
            "high": 16384,
            "xhigh": 24576,
        }
        budget = budgets.get(level)
        return {"thinkingBudget": budget} if budget is not None else None
    return None


def anthropic_thinking_payload(level: str, model: str) -> Optional[Dict[str, Any]]:
    level = normalize_thinking_level(level)
    if level in {"auto", "none"}:
        return None
    model_lower = normalize_whitespace(model).lower()
    if anthropic_uses_adaptive_thinking(model_lower):
        return {"type": "adaptive"}
    budget = ANTHROPIC_THINKING_BUDGETS.get(level)
    if not budget:
        return None
    return {"type": "enabled", "budget_tokens": budget, "display": "omitted"}


def anthropic_uses_adaptive_thinking(model: str) -> bool:
    model_lower = normalize_whitespace(model).lower()
    return any(
        marker in model_lower
        for marker in (
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-opus-4.8",
            "claude-opus-4-8",
            "claude-opus-4.7",
            "claude-opus-4-7",
            "claude-sonnet-4.6",
            "claude-sonnet-4-6",
            "claude-opus-4.6",
            "claude-opus-4-6",
        )
    )


def anthropic_supports_temperature(model: str) -> bool:
    model_lower = normalize_whitespace(model).lower()
    # Sampling params (temperature/top_p/top_k) were removed (HTTP 400) on the
    # Claude 5 family and Opus 4.7/4.8; Opus/Sonnet 4.6 and older still accept them.
    return not any(
        marker in model_lower
        for marker in (
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-opus-4.8",
            "claude-opus-4-8",
            "claude-opus-4.7",
            "claude-opus-4-7",
        )
    )


def anthropic_output_effort(level: str) -> Optional[str]:
    level = normalize_thinking_level(level)
    if level in {"auto", "none"}:
        return None
    if level == "xhigh":
        return "high"
    if level == "minimal":
        return "low"
    return level if level in {"low", "medium", "high"} else None


def numeric_value(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_whitespace(value).replace(",", "")
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return default
    return default


def clamp_text(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.86)].rstrip()
    tail = text[-int(limit * 0.1) :].lstrip()
    return f"{head}\n\n[... content trimmed for length ...]\n\n{tail}"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return value


def compact_json(data: Any, limit: int = 90000) -> str:
    return clamp_text(json.dumps(json_safe(data), ensure_ascii=True, indent=2), limit)


# ─────────────────────────────────────────────────────────────────────────────
# Robust JSON extraction
# Handles: plain JSON, ```json fenced blocks, prose with embedded {...},
#          single-quoted JSON (some Ollama models), and truncated JSON repair.
# ─────────────────────────────────────────────────────────────────────────────

def _json_repair_candidates(candidate: str) -> List[str]:
    """Return strict-JSON repair variants for common LLM formatting mistakes."""
    text = str(candidate or "").strip()
    variants = [text]
    if "\\'" in text:
        variants.append(text.replace("\\'", "'"))
    if text.startswith("{{"):
        repaired = "{" + text[2:].lstrip()
        if repaired.rstrip().endswith("}}"):
            repaired = repaired.rstrip()[:-1].rstrip()
        variants.append(repaired)
        if "\\'" in repaired:
            variants.append(repaired.replace("\\'", "'"))
    # Some models emit {{...}} with a single valid object inside. Remove only
    # one duplicated outer brace pair so nested objects remain intact.
    if text.startswith("{{") and text.endswith("}}"):
        repaired = text[1:-1].strip()
        variants.append(repaired)
        if "\\'" in repaired:
            variants.append(repaired.replace("\\'", "'"))
    unique: List[str] = []
    seen: Set[str] = set()
    for item in variants:
        if item and item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def _try_parse_json(candidate: str) -> Optional[Dict[str, Any]]:
    """Try to parse candidate as JSON dict. Returns None on failure."""
    for item in _json_repair_candidates(candidate):
        try:
            parsed = json.loads(item)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        except TypeError:
            pass
    return None


def _repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to repair a truncated JSON object by closing open brackets/braces.
    Handles truncation mid-string, mid-array, and mid-object.
    """
    brace_pos = text.find("{")
    if brace_pos == -1:
        return None

    candidate = text[brace_pos:]

    # Track parser state
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape = False
    # Track whether we're on a key or value (to know how to close mid-string)
    # key positions: after { or , at even depth inside an object
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)

    if depth_brace == 0 and depth_bracket == 0 and not in_string:
        return None  # nothing to repair

    suffix = ""
    if in_string:
        # Close the open string value, then close the containing object
        suffix += '"'
        # We are inside an object at depth_brace level, close it
        # (the string was a value, so we just need to close the object levels)
        suffix += "}" * max(0, depth_brace - 1)
        # Close any open arrays between the current object and the root
        suffix += "]" * max(0, depth_bracket)
        # Close root object
        suffix += "}"
    else:
        # Not mid-string — just close arrays then objects
        suffix += "]" * max(0, depth_bracket)
        suffix += "}" * max(0, depth_brace)

    if not suffix:
        return None

    # Try repair as-is
    result = _try_parse_json(candidate + suffix)
    if result is not None:
        return result

    # Strip trailing comma and retry
    stripped = re.sub(r",\s*$", "", candidate.rstrip())
    result = _try_parse_json(stripped + suffix)
    if result is not None:
        return result

    # Last resort: if there's a trailing incomplete key-value, remove it
    # e.g. {"a": 1, "b": "partia → remove , "b": "partia
    last_comma = stripped.rfind(",")
    if last_comma != -1:
        truncated_at_comma = stripped[:last_comma]
        result = _try_parse_json(truncated_at_comma + "}" * max(1, depth_brace) )
        if result is not None:
            return result

    return None


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from model output.
    Handles:
    - Plain JSON responses
    - Markdown ```json ... ``` fences
    - Inline JSON embedded in prose
    - Single-quoted JSON (some Ollama models)
    - Truncated JSON (token-limit cutoff)
    - JSON with Python-style True/False/None
    """
    if not text or not text.strip():
        raise ValueError("Model returned an empty response. Check your API key, model name, and prompt.")

    candidate = text.strip()

    # 1. Try the whole thing as-is
    result = _try_parse_json(candidate)
    if result is not None:
        return result

    # 2. Strip markdown fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence_match:
        result = _try_parse_json(fence_match.group(1).strip())
        if result is not None:
            return result

    # 3. Extract the largest {...} block from prose
    #    Walk character by character to find balanced braces
    best: Optional[Dict[str, Any]] = None
    in_string = False
    escape = False
    depth = 0
    start = None
    for i, ch in enumerate(candidate):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = candidate[start: i + 1]
                result = _try_parse_json(block)
                if result is not None:
                    # prefer the largest valid object found
                    if best is None or len(block) > len(json.dumps(best)):
                        best = result
                # reset to look for more blocks
                start = None
                in_string = False
                escape = False

    if best is not None:
        return best

    # 4. Handle Python-style True/False/None (some Ollama models)
    py_fixed = (
        candidate
        .replace(": True", ": true")
        .replace(": False", ": false")
        .replace(": None", ": null")
        .replace(",True", ",true")
        .replace(",False", ",false")
        .replace(",None", ",null")
    )
    result = _try_parse_json(py_fixed)
    if result is not None:
        return result

    # 5. Try again after stripping fence from python-fixed version
    fence_match2 = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", py_fixed, re.DOTALL)
    if fence_match2:
        result = _try_parse_json(fence_match2.group(1).strip())
        if result is not None:
            return result

    # 6. Attempt to repair truncated JSON (token-limit cutoff)
    result = _repair_truncated_json(candidate)
    if result is not None:
        return result

    # 7. Nothing worked — provide a diagnostic error
    preview = candidate[:300].replace("\n", " ")
    raise ValueError(
        f"Could not extract a JSON object from model response. "
        f"The model may have returned prose instead of JSON, hit a token limit, "
        f"or the API key / model may be misconfigured. "
        f"Response preview: {preview!r}"
    )


def extract_explicit_role_rows(text: str, tags: List[str], limit: int = 40) -> List[Dict[str, Any]]:
    """Build equipment/control role rows only from supplied sentences that explicitly mention the tag."""
    statements = split_complete_statements(text, limit=90, max_chars=520)
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for tag in tags:
        clean_tag = normalize_whitespace(tag)
        if not clean_tag:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(clean_tag)}(?![A-Za-z0-9])", flags=re.I)
        for statement in statements:
            if not pattern.search(statement):
                continue
            key = f"{clean_tag.upper()}::{statement.lower()}"
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "tag_or_name": clean_tag,
                "role": statement,
                "evidence": [f"Supplied sentence containing {clean_tag}"],
            })
            break
        if len(rows) >= limit:
            break
    return rows


def extract_explicit_flow_rows(context: Dict[str, str], limit: int = 10) -> List[Dict[str, Any]]:
    """Extract only clear supplied statements about material, energy, or procedural flow."""
    source_text = " ".join([
        context.get("background_and_process_description", ""),
        context.get("detailed_proposed_sequence", ""),
    ])
    statements = split_complete_statements(source_text, limit=80, max_chars=520)
    flow_terms = re.compile(
        r"\b(feed|fed|flow|flows|transfer|transfers|pumped|circulated|condensed|"
        r"boiled|heated|reflux|blowback|displace|depressurize|vent|drain|"
        r"open|close|receives)\b",
        flags=re.I,
    )
    rows: List[Dict[str, Any]] = []
    for statement in statements:
        if not flow_terms.search(statement):
            continue
        rows.append({
            "path_name": "Supplied process or procedure flow",
            "material_or_service": "",
            "normal_direction": statement,
            "controlling_items": [],
            "evidence": ["Supplied study text"],
        })
        if len(rows) >= limit:
            break
    return rows


def manual_process_understanding(manual_setup: Dict[str, Any], documents: Dict[str, Any]) -> Dict[str, Any]:
    context = manual_setup.get("manual_context") or {}
    procedure = parse_numbered_procedure_steps(context.get("full_manual_text", ""))
    guideword_matrix = manual_setup.get("guideword_matrix") or []
    known_safeguards = sanitize_known_safeguards(
        manual_setup.get("known_safeguards")
        or extract_known_safeguards_from_text(
            supplied_document_text(documents),
            "process_description_procedure_and_drawing_text",
        ),
        30,
    )
    worksheet_columns = manual_setup.get("worksheet_columns") or [
        "Guideword", "Deviation / Effect", "Potential cause", "Consequence",
        "Safeguard", "Necessary Action",
    ]

    overview = split_complete_statements(
        context.get("background_and_process_description", ""),
        limit=12,
        max_chars=420,
    )
    if context.get("node_definition_section"):
        overview.append("Use the supplied node definitions and design intentions as the governing node basis for the HAZOP study.")
    if context.get("guideword_section"):
        overview.append("Use the supplied guideword section as the governing guideword matrix for each applicable node.")
    if not overview:
        overview = ["Manual HAZOP setup was supplied by the user and is the controlling study basis."]

    supplied_nodes = []
    for node in manual_setup.get("nodes", []):
        if not isinstance(node, dict):
            continue
        supplied_nodes.append({
            "node_id": node.get("node_id", ""),
            "node_name": node.get("node_name", ""),
            "boundary": node.get("boundaries", ""),
            "design_intent": node.get("design_intent", ""),
            "parameters": node.get("parameters", []),
            "study_focus": node.get("study_focus", []),
            "steps": node.get("steps", ""),
        })

    material_flow = extract_explicit_flow_rows(context)

    role_tags: List[str] = []
    seen_tags: Set[str] = set()
    for node in manual_setup.get("nodes", []):
        for tag in node.get("equipment", []) or []:
            key = str(tag).upper()
            if key in seen_tags:
                continue
            seen_tags.add(key)
            role_tags.append(str(tag))
    text_tags = sorted(set(re.findall(
        r"\b(?:V\d{1,3}|P\d{1,3}|H\d{1,3}|C\d{1,3}|T\d{1,3}|J\d{1,3}|IST|"
        r"LIC|TIC|FIC|PV|PRV|NRV|N2|C6)\b",
        context.get("full_manual_text", ""),
        flags=re.I,
    )))
    for tag in text_tags:
        key = tag.upper()
        if key not in seen_tags:
            seen_tags.add(key)
            role_tags.append(tag)
    role_basis_text = context.get("background_and_process_description", "")
    equipment_roles = extract_explicit_role_rows(role_basis_text, role_tags)

    return {
        "process_overview": overview,
        "material_and_energy_flow": material_flow,
        "equipment_and_control_roles": equipment_roles,
        "procedure_basis": {
            "initial_state": sanitize_list(procedure.get("initial_setup", ""), 6),
            "ordered_steps": [
                {
                    "step_no": str(item.get("step_number", "")),
                    "action": item.get("instruction", ""),
                    "purpose": "",
                    "affected_node_or_boundary": "",
                }
                for item in procedure.get("steps", [])
            ],
            "operator_roles": [],
        },
        "hazop_basis": {
            "study_scope": study_scope_summary(documents),
            "worksheet_columns": worksheet_columns,
            "supplied_nodes": supplied_nodes,
            "guideword_matrix": guideword_matrix,
            "guideword_section": context.get("guideword_section", ""),
        },
        "known_safeguards": known_safeguards,
        "uncertainties": manual_setup.get("warnings", []),
        "do_not_assume": [
            "Do not add guidewords, nodes, safeguards, valve states, or operating steps that are not supplied in the study text or visible in the P&ID.",
            "Do not move communication-related causes into non-communication guideword rows.",
            "Use any HAZOP Worksheets section supplied inside the uploaded study text only for matching node and guideword rows; do not copy unrelated rows across nodes.",
        ],
        "basis_source": "manual_hazop_setup",
    }


def build_process_understanding(payload: Dict[str, Any]) -> Dict[str, Any]:
    manual_setup = parse_manual_hazop_setup(payload["documents"])
    if manual_setup.get("nodes") or manual_setup.get("guideword_matrix"):
        return manual_process_understanding(manual_setup, payload["documents"])

    fallback: Dict[str, Any] = {
        "process_overview": [
            "Shared process understanding could not be generated by the model; downstream agents must rely directly on the supplied P&ID parse and text documents.",
            "No HAZOP rows were generated in this fallback basis."
        ],
        "material_and_energy_flow": [],
        "equipment_and_control_roles": [],
        "procedure_basis": {
            "initial_state": [],
            "ordered_steps": [],
            "operator_roles": [],
        },
        "hazop_basis": {
            "study_scope": study_scope_summary(payload["documents"]),
            "worksheet_columns": manual_setup.get("worksheet_columns") or [
                "Guideword", "Deviation / Effect", "Potential cause", "Consequence",
                "Safeguard", "Necessary Action",
            ],
            "supplied_nodes": manual_setup.get("nodes", []),
        },
        "known_safeguards": [],
        "uncertainties": ["Process-understanding model call failed or returned invalid JSON."],
        "do_not_assume": [
            "Do not invent equipment, valve states, safeguards, guidewords, or operating conditions that are not supplied or visible."
        ],
    }
    if manual_setup.get("manual_context"):
        context = manual_setup["manual_context"]
        fallback["process_overview"] = [
            clamp_text(context.get("background_and_process_description", ""), 1200)
            or "Manual HAZOP setup was supplied by the user.",
            "Use the supplied node definitions, design intentions, detailed proposed sequence, and guideword section as the governing study basis.",
        ]
        fallback["procedure_basis"]["initial_state"] = sanitize_list(
            manual_setup["nodes"][0].get("procedure_initial_setup") if manual_setup.get("nodes") else "",
            4,
        )
        fallback["procedure_basis"]["ordered_steps"] = [
            {"step_no": str(item.get("step_number", "")), "action": item.get("instruction", ""), "purpose": "", "affected_node_or_boundary": ""}
            for item in parse_numbered_procedure_steps(context.get("full_manual_text", "")).get("steps", [])
        ]

    try:
        raw = run_model(
            "panel_chair",
            payload["agents"]["panel_chair"],
            payload["api_keys"],
            system_prompt_for("panel_chair"),
            process_understanding_prompt(payload["documents"]),
            trace={
                "job_id": payload.get("_job_id", ""),
                "stage": "process_understanding",
                "round": "pre-study",
            },
        )
        parsed = extract_json_object(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Process-understanding response was not a JSON object.")
        parsed["raw_text"] = raw
        return parsed
    except Exception as err:
        fallback["error"] = str(err)
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Sanitizers
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_list(value: Any, limit: int = 20) -> List[str]:
    items: List[str] = []
    seen = set()
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str) and normalize_whitespace(value):
        raw_items = [value]
    else:
        raw_items = []
    for item in raw_items:
        text = normalize_whitespace(item)
        if not text:
            continue
        key = text.lower().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
        if len(items) >= limit:
            break
    return items


def sanitize_assumptions(value: Any, limit: int = 20) -> List[str]:
    """Keep technical assumptions and remove internal merge/debug notes."""
    blocked_patterns = (
        "preserved from ",
        "because chair/scribe output was incomplete",
        "this recommendation consolidates",
        "this combines",
        "draft rows",
        "expert inputs",
        "expert scenario fallback",
        "requires verification:",
        "documented operating procedure",
        "documented operating procedures",
        "routine procedure",
        "operator rounds",
        "valve-position checks",
        "field observation",
        "no automated ",
        "no local ",
        "lack of ",
        "needs verification",
        "requires verification",
        "must be confirmed",
        "must be verified",
        "needs to be confirmed",
        "needs to be verified",
        "is an assumption",
        "supplied",
        "uploaded",
        "user-provided",
        "user provided",
        "provided by user",
        "p&id",
        "pid",
        "process description",
        "worksheet",
        "manual hazop",
        "manual setup",
        "procedure step",
        "source reference",
        "source references",
        "not shown",
        "not provided",
        "unknown",
        "assuming",
    )
    cleaned: List[str] = []
    seen = set()
    for item in sanitize_list(value, limit * 2):
        lower = item.lower()
        if any(pattern in lower for pattern in blocked_patterns):
            continue
        key = lower.rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def sanitize_source_references(value: Any, limit: int = 20) -> List[str]:
    """Keep only external web references or DOI citations."""
    blocked_terms = (
        "supplied",
        "uploaded",
        "user-provided",
        "user provided",
        "provided by user",
    )
    cleaned: List[str] = []
    seen = set()
    for item in sanitize_list(value, limit * 2):
        lower = item.lower().strip().rstrip(".")
        if any(term in lower for term in blocked_terms):
            continue
        if not re.search(r"(https?://|www\.|\bdoi\s*:|\b10\.\d{4,9}/\S+)", item, re.IGNORECASE):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def safeguard_evidence_sentence(text: str, start: int, end: int, limit: int = 520) -> str:
    """Return the complete supplied sentence surrounding a safeguard mention."""
    source = str(text or "")
    prior_sentence_ends = list(re.finditer(r"\.(?=\s+[A-Z0-9]|\s*$)", source[:start]))
    prior_period_end = prior_sentence_ends[-1].end() if prior_sentence_ends else 0
    left = max(prior_period_end, source.rfind("\n", 0, start) + 1)
    sentence_end = re.search(r"\.(?=\s+[A-Z0-9]|\s*$)", source[end:])
    period_position = end + sentence_end.start() if sentence_end else -1
    right_candidates = [position for position in (
        period_position, source.find("\n", end)
    ) if position >= 0]
    right = min(right_candidates) + 1 if right_candidates else min(len(source), end + 260)
    return clamp_text(normalize_whitespace(source[left:right]), limit)


def safeguard_evidence_status(evidence: str) -> str:
    """Distinguish installed/documented controls from recommendations or proposals."""
    lower = normalize_whitespace(evidence).lower()
    if re.search(
        r"\b(?:recommend(?:ed|ation)?|propos(?:e|ed|al)|consider(?:ation)?|future)\b|"
        r"\bshould\s+be\s+(?:installed|provided|added|fitted|implemented|clearly\s+label(?:ed|led))\b|"
        r"\bneed\s+for\s+(?:a|an|the)\b",
        lower,
    ):
        return "recommended_unconfirmed"
    return "documented_existing"


def safeguard_evidence_is_negative(evidence: str) -> bool:
    """Reject mentions that explicitly say a control is absent or not visible."""
    lower = normalize_whitespace(evidence).lower()
    return bool(re.search(
        r"\b(?:not|isn't|isnt)\s+(?:visibl(?:e|y)|shown|identified|installed|provided|available|documented|present)\b|"
        r"\b(?:no|without)\b.{0,60}\b(?:safeguard|alarm|trip|interlock|relief|indicator|controller|"
        r"control|nrv|non[-\s]return|bund|containment|flame\s+arrester|seal\s+pot)\b|"
        r"\b(?:safeguard|alarm|trip|interlock|relief|indicator|controller|control|nrv|non[-\s]return|"
        r"bund|containment|flame\s+arrester|seal\s+pot)\b.{0,40}\b(?:not\s+(?:documented|visible|"
        r"installed|provided|available|present)|is\s+absent)\b",
        lower,
    ))


def extract_known_safeguards_from_text(
    text: str,
    source: str = "supplied_text",
) -> List[Dict[str, str]]:
    """Extract controls from supplied text while retaining evidence and status."""
    evidence_text = str(text or "")
    patterns = [
        (
            r"\bvalves?\s+(?:should\s+be\s+)?(?:clearly\s+)?label(?:ed|led)\b|\bvalve\s+label(?:ing|ling)\b",
            "Valve labeling and practice",
            "administrative",
        ),
        (
            r"\bindependent\s+(?:line[-\s]?up|valve|field)\s+verification\b|\bline[-\s]?up\s+check\b|\bpre[-\s]?start\s+check(?:list)?\b",
            "Independent line-up verification",
            "administrative",
        ),
        (
            r"\bpermit[-\s]?to[-\s]?work\b|\bPTW\b|\block[-\s]?out\s*[/\-]?\s*tag[-\s]?out\b|\bLOTO\b",
            "Permit-to-work / LOTO",
            "administrative",
        ),
        (
            r"\bradio\s+(?:communication|protocol|procedure|message)\b|\bcommunicat(?:e|ion)\s+by\s+radio\b",
            "Radio communication protocol",
            "administrative",
        ),
        (
            r"\blead\s+operator\b.{0,120}\b(?:control(?:s|led)?|instruction(?:s)?|procedure)\b|"
            r"\bprocedure\b.{0,120}\blead\s+operator\b",
            "Lead-operator procedural control",
            "administrative",
        ),
        (
            r"\bdummy\s+(?:run|practice)\b|\bpractice\s+run\b",
            "Dummy-run procedural practice",
            "administrative",
        ),
        (
            r"\b(?:nitrogen|n2)\s+is\s+heard\b|\bheard\s+passing\b|\baudible\s+(?:nitrogen|n2)\s+breakthrough\b",
            "Audible nitrogen-breakthrough indication",
            "administrative",
        ),
        (
            r"\blocked[-\s]?(?:open|closed)\b|\bcar[-\s]?seal(?:ed)?[-\s]?(?:open|closed)?\b|\bchain(?:ed)?[-\s]?(?:open|closed)\b",
            "Locked or car-sealed valve position",
            "administrative",
        ),
        (
            r"\bblind\b|\bspade\b|\bspectacle\s+blind\b|\bpositive\s+isolation\b",
            "Positive isolation blind",
            "mechanical",
        ),
        (
            r"\bNRV\b|\bnon[-\s]return\s+valve\b|\bcheck\s+valve\b",
            "Non-return valve",
            "mechanical",
        ),
        (
            r"\brestriction\s+orifice\b|\bflow\s+restrict(?:or|ion)\b|\bflow\s+limiter\b",
            "Flow restriction device",
            "mechanical",
        ),
        (
            r"\bpressure\s+regulat(?:or|ion)\b|\bregulat(?:ed|ing)\s+pressure\b",
            "Pressure regulator",
            "mechanical",
        ),
        (
            r"\bflame\s+arrester\b|\bflame\s+arrestor\b",
            "Flame arrester",
            "mechanical",
        ),
        (
            r"\bPRV\b|\bPSV\b|\bpressure\s+relief\s+valve\b|\brelief\s+valve\b|\bpressure\s*\(vacuum\)\s+valve\b",
            "Pressure relief valve",
            "mechanical",
        ),
        (
            r"\brupture\s+disc\b|\brupture\s+disk\b",
            "Rupture disk",
            "mechanical",
        ),
        (
            r"\bvent\s+(?:to\s+)?(?:safe|atmosphere|flare|blowdown)\b|\bsafe\s+vent\b|\bvent\s+system\b",
            "Vent to safe location",
            "mitigation",
        ),
        (
            r"\bflare\b|\bblowdown\s+system\b|\bblowdown\s+drum\b",
            "Flare or blowdown system",
            "mitigation",
        ),
        (
            r"\bscrubber\b|\babsorber\b",
            "Scrubber or absorber",
            "mitigation",
        ),
        (
            r"\bhigh[-\s]level\s+trip\b|\bhigh\s+level\s+trip\b",
            "High-level trip",
            "instrumented",
        ),
        (
            r"\b(?:high[-\s]high|low[-\s]low|HH|LL)\s+(?:trip|interlock)\b|\btrip\s+function\b",
            "High-high / low-low trip",
            "instrumented",
        ),
        (
            r"\binterlock\b|\bemergency\s+shutdown\b|\bESD\b|\bshutdown\s+valve\b|\bSIS\b|\bSIF\b|\bsafety\s+instrumented\b",
            "Automatic shutdown/interlock",
            "instrumented",
        ),
        (
            r"\bFIC\b|\bflow\s+(?:indicator\s*/\s*)?controller\b|\bflow\s+control\s+loop\b",
            "Flow control loop",
            "instrumented",
        ),
        (
            r"\bPIC\b|\bpressure\s+(?:indicator\s*/\s*)?controller\b|\bpressure\s+control\s+loop\b|"
            r"\bsplit[-\s]?range\s+pressure\s+control\b",
            "Pressure control loop",
            "instrumented",
        ),
        (
            r"\bTIC\b|\btemperature\s+(?:indicator\s*/\s*)?controller\b|\btemperature\s+control\s+loop\b",
            "Temperature control loop",
            "instrumented",
        ),
        (
            r"\bLIC\b|\blevel\s+(?:indicator\s*/\s*)?controller\b|\blevel\s+control\s+loop\b",
            "Level control loop",
            "instrumented",
        ),
        (
            r"\bhigh[-\s]level\s+alarm\b|\blow[-\s]level\s+alarm\b|\blevel\s+alarm\b",
            "Level alarm",
            "instrumented",
        ),
        (
            r"\bpressure\s+alarm\b|\bhigh[-\s]pressure\s+alarm\b|\blow[-\s]pressure\s+alarm\b",
            "Pressure alarm",
            "instrumented",
        ),
        (
            r"\btemperature\s+alarm\b|\bhigh[-\s]temperature\s+alarm\b|\blow[-\s]temperature\s+alarm\b",
            "Temperature alarm",
            "instrumented",
        ),
        (
            r"\bflow\s+alarm\b|\bhigh[-\s]flow\s+alarm\b|\blow[-\s]flow\s+alarm\b",
            "Flow alarm",
            "instrumented",
        ),
        (
            r"\bpressure\s+gauge\b|\bflow\s+meter\b|\bsight\s*glass\b|\blevel\s+indicator\b|\btemperature\s+indicator\b|\bpressure\s+indicator\b",
            "Local indication",
            "instrumented",
        ),
        (
            r"\bbunded\s+enclosure\b|\bbund\b",
            "Bunded enclosure",
            "passive",
        ),
        (
            r"\boverflow\b.{0,100}\bsealed\s+with\s+glycol\b|\bglycol[-\s]seal(?:ed)?\s+(?:overflow|seal\s+pot)\b",
            "Glycol-sealed overflow to bund",
            "passive",
        ),
        (
            r"\bdike\b|\bsecondary\s+containment\b|\bcontainment\s+curb\b",
            "Secondary containment",
            "passive",
        ),
        (
            r"\bnitrogen\s+blanket(?:ing)?\b|\binert(?:ed|ing|ion)\b",
            "Nitrogen blanketing/inerting",
            "passive",
        ),
        (
            r"\bfire\s+(?:and\s+)?gas\s+detection\b|\bgas\s+detector\b|\bfire\s+detector\b|\bflammable\s+gas\s+detector\b",
            "Fire and gas detection",
            "mitigation",
        ),
        (
            r"\bventilation\b|\bextraction\s+fan\b",
            "Ventilation",
            "mitigation",
        ),
        (
            r"\bfirewater\b|\bdeluge\b|\bsprinkler\b",
            "Firewater / deluge",
            "mitigation",
        ),
    ]
    safeguards: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for pattern, name, category in patterns:
        match = re.search(pattern, evidence_text, flags=re.I)
        if not match:
            continue
        evidence = safeguard_evidence_sentence(evidence_text, match.start(), match.end())
        if safeguard_evidence_is_negative(evidence):
            continue
        resolved_name = name
        if name == "Non-return valve" and re.search(r"\b(?:temporary|flexible)?\s*(?:nitrogen|n2)?\s*"
                                                     r"hose\b|\bhose\s+connection\b", evidence, flags=re.I):
            resolved_name = "Temporary nitrogen-hose non-return valve"
        key = resolved_name.lower()
        if key in seen:
            continue
        seen.add(key)
        safeguards.append({
            "name": resolved_name,
            "category": category,
            "evidence": evidence or normalize_whitespace(match.group(0)),
            "status": safeguard_evidence_status(evidence),
            "source": source,
        })
    return safeguards


def safeguard_names(value: Any, confirmed_only: bool = False) -> List[str]:
    names: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                if confirmed_only and normalize_whitespace(item.get("status")).lower() in {
                    "recommended_unconfirmed", "proposed", "unconfirmed",
                }:
                    continue
                text = normalize_whitespace(item.get("name") or item.get("safeguard") or item.get("label"))
            else:
                text = normalize_whitespace(item)
            if text:
                names.append(text)
    elif isinstance(value, str):
        names.extend(row_text_items(value, 20))
    return names


def documented_safeguard_phrase(item: str, documents: Dict[str, Any]) -> bool:
    """Return True when the named safeguard is explicitly supported by the uploaded text."""
    lower = normalize_whitespace(item).lower()
    if not lower:
        return False
    extracted = [
        s["name"].lower()
        for s in extract_known_safeguards_from_text(supplied_document_text(documents))
        if s.get("status") != "recommended_unconfirmed"
    ]
    if lower in extracted:
        return True
    if "valve labeling" in lower and "valve labeling and practice".lower() in extracted:
        return True
    if ("line-up" in lower or "lineup" in lower or "verification" in lower) and "independent line-up verification" in extracted:
        return True
    if ("permit" in lower or "loto" in lower or "lockout" in lower) and "permit-to-work / loto" in extracted:
        return True
    if ("locked" in lower or "car-seal" in lower or "car sealed" in lower or "chained" in lower) and "locked or car-sealed valve position" in extracted:
        return True
    if ("blind" in lower or "spade" in lower or "positive isolation" in lower) and "positive isolation blind" in extracted:
        return True
    if ("non-return" in lower or "nrv" in lower) and any("non-return valve" in name for name in extracted):
        return True
    if ("restriction" in lower or "limiter" in lower) and "flow restriction device" in extracted:
        return True
    if "pressure regulator" in lower and "pressure regulator" in extracted:
        return True
    if "flame arrester" in lower and "flame arrester" in extracted:
        return True
    if ("pressure relief" in lower or "prv" in lower or "psv" in lower) and "pressure relief valve" in extracted:
        return True
    if "rupture" in lower and "rupture disk" in extracted:
        return True
    if "vent" in lower and "vent to safe location" in extracted:
        return True
    if ("flare" in lower or "blowdown" in lower) and "flare or blowdown system" in extracted:
        return True
    if ("scrubber" in lower or "absorber" in lower) and "scrubber or absorber" in extracted:
        return True
    if "radio" in lower and "radio communication protocol" in extracted:
        return True
    if ("lead-operator" in lower or "lead operator" in lower) and "lead-operator procedural control" in extracted:
        return True
    if ("trip" in lower or "interlock" in lower or "shutdown" in lower or "sis" in lower or "sif" in lower) and (
        "high-level trip" in extracted
        or "high-high / low-low trip" in extracted
        or "automatic shutdown/interlock" in extracted
    ):
        return True
    if "flow control" in lower and "flow control loop" in extracted:
        return True
    if "pressure control" in lower and "pressure control loop" in extracted:
        return True
    if "temperature control" in lower and "temperature control loop" in extracted:
        return True
    if "level control" in lower and "level control loop" in extracted:
        return True
    if "alarm" in lower and any(name in extracted for name in ["level alarm", "pressure alarm", "temperature alarm", "flow alarm"]):
        return True
    if ("indicator" in lower or "indication" in lower or "gauge" in lower or "meter" in lower or "sight" in lower) and "local indication" in extracted:
        return True
    if "secondary containment" in lower and "secondary containment" in extracted:
        return True
    if ("nitrogen" in lower or "inert" in lower) and "nitrogen blanketing/inerting" in extracted:
        return True
    if ("fire" in lower or "gas detection" in lower) and "fire and gas detection" in extracted:
        return True
    if "ventilation" in lower and "ventilation" in extracted:
        return True
    if ("firewater" in lower or "deluge" in lower or "sprinkler" in lower) and "firewater / deluge" in extracted:
        return True
    return False


def safeguard_applies_to_row(safeguard: str, row: Dict[str, Any]) -> bool:
    """Match a documented safeguard to the row mechanism without using a scoring threshold."""
    sg = normalize_whitespace(safeguard).lower()
    row_text = normalize_whitespace(" ".join([
        str(row.get("guideword", "")),
        str(row.get("parameter", "")),
        str(row.get("deviation", "") or row.get("effect", "")),
        str(row.get("potential_cause", "") or row.get("cause", "")),
        str(row.get("consequence", "")),
    ])).lower()
    if not sg or not row_text or row_is_not_applicable(row):
        return False
    if "dummy run" in sg or "practice run" in sg or "supervision" in sg:
        return False
    mechanical_or_integrity_failure = bool(re.search(
        r"\b(?:regulator|hose\s+material|hose\s+defect|material\s+defect|external\s+impact|abrasion|rupture|"
        r"pressure\s+relief|mechanical\s+seal|bearing|corrosion|erosion|fouling|blockage|blocked|stuck|"
        r"fails?\s+(?:open|closed|high|low|to\s+open|to\s+close)|calibration)\b",
        row_text,
    ))
    human_or_lineup_error = bool(re.search(
        r"\b(?:operator|wrong|mistake|misread|left\s+open|left\s+closed|opened|closed|sequence|line[-\s]?up|"
        r"valve\s+position|incorrectly\s+set|v\d+)\b",
        row_text,
    ))
    if "valve labeling" in sg:
        return human_or_lineup_error and not (
            mechanical_or_integrity_failure
            and not re.search(r"\b(?:operator|wrong|mistake|misread|left\s+open|left\s+closed|incorrectly\s+set)\b", row_text)
        )
    if "radio" in sg:
        return bool(re.search(r"\bcommunication|radio|instruction|message|misunderstanding\b", row_text))
    if "lead-operator" in sg or "lead operator" in sg:
        return bool(re.search(
            r"\boperator|communication|instruction|message|misunderstanding|sequence|"
            r"too\s+early|too\s+late|opened|closed\b",
            row_text,
        ))
    if "audible nitrogen-breakthrough" in sg or "audible n2 breakthrough" in sg:
        if re.search(r"\b(?:fails?\s+to\s+hear|cannot\s+hear|miss(?:es|ed)?|mishear|misinterpret)\b", row_text):
            return False
        return bool(re.search(r"\b(?:breakthrough|clearance|incomplete|less|magnitude|nitrogen|n2)\b", row_text))
    if "line-up" in sg or "lineup" in sg or "verification" in sg:
        return human_or_lineup_error
    if "permit" in sg or "loto" in sg or "lockout" in sg:
        return bool(re.search(r"\bmaintenance|isolation|disconnect|connect|temporary|hose|blind|spade|drain|line\s+break\b", row_text))
    if "locked" in sg or "car-sealed" in sg:
        return human_or_lineup_error or bool(re.search(r"\bbypass|manual|block\s+valve|isolation\b", row_text))
    if "positive isolation" in sg or "blind" in sg:
        return bool(re.search(r"\bisolation|wrong\s+route|backflow|reverse|maintenance|contamination|release\b", row_text))
    if "non-return" in sg or "nrv" in sg:
        return bool(re.search(r"\breverse|backflow|drain|upstream|downstream|pump|return\b", row_text))
    if "flow restriction" in sg or "restrictor" in sg or "orifice" in sg or "limiter" in sg:
        return bool(re.search(r"\bmore\s+flow|high\s+flow|too\s+fast|excessive\s+flow|high\s+rate|jet|velocity\b", row_text))
    if "pressure regulator" in sg:
        return bool(re.search(r"\boverpressure|high\s+pressure|utility|nitrogen|air|gas|regulator\b", row_text))
    if "flame arrester" in sg:
        return bool(re.search(r"\bflame|ignition|fire|flammable|vent\b", row_text))
    if "pressure relief" in sg or "prv" in sg or "psv" in sg:
        return bool(re.search(
            r"\boverpressure|pressure\s*\(\s*more\s*\)|high(?:\s+[a-z0-9-]+){0,3}\s+pressure|"
            r"pressure\s+build|raises?\s+pressure\s+above|rupture|relief|"
            r"exceed(?:s|ed)?\s+(?:design|allowable)\b",
            row_text,
        ))
    if "rupture disk" in sg:
        return bool(re.search(r"\boverpressure|high\s+pressure|pressure\s+build|rupture|relief|exceed(?:s|ed)?\s+design\b", row_text))
    if "vent to safe" in sg or sg == "ventilation":
        return bool(re.search(r"\brelease|vent|gas|vapou?r|asphyxiation|toxic|flammable\b", row_text))
    if "flare" in sg or "blowdown" in sg or "scrubber" in sg or "absorber" in sg:
        return bool(re.search(r"\brelease|relief|vent|toxic|flammable|vapou?r|gas|overpressure\b", row_text))
    if "trip" in sg or "interlock" in sg or "shutdown" in sg:
        return bool(re.search(
            r"\b(?:high|low|high[-\s]high|low[-\s]low|no\s+flow|loss\s+of\s+flow|overpressure|"
            r"high\s+pressure|high\s+temperature|low\s+level|high\s+level|shutdown|trip)\b",
            row_text,
        ))
    if "flow control" in sg:
        if re.search(r"\breverse|backflow\b", row_text):
            return False
        return bool(re.search(r"\b(?:flow|no\s+flow|more\s+flow|less\s+flow|high\s+flow|low\s+flow|too\s+fast|too\s+slow)\b", row_text))
    if "pressure control" in sg:
        if re.search(r"\breverse|backflow\b", row_text):
            return False
        return bool(re.search(r"\b(?:pressure|overpressure|high\s+pressure|low\s+pressure|vacuum)\b", row_text))
    if "temperature control" in sg:
        return bool(re.search(r"\b(?:temperature|overheat|heating|cooling|hot|cold|high\s+temperature|low\s+temperature)\b", row_text))
    if "level control" in sg:
        return bool(re.search(r"\b(?:level|overflow|overfill|underfill|high\s+level|low\s+level)\b", row_text))
    if "level alarm" in sg or "level trip" in sg:
        return bool(re.search(r"\blevel|overflow|underfill|overfill\b", row_text))
    if "pressure alarm" in sg:
        return bool(re.search(r"\bpressure|overpressure|vacuum\b", row_text))
    if "temperature alarm" in sg:
        return bool(re.search(r"\btemperature|overheat|cooling|heating|hot|cold\b", row_text))
    if "flow alarm" in sg:
        return bool(re.search(r"\bflow|no\s+flow|less\s+flow|more\s+flow|reverse\s+flow\b", row_text))
    if "local indication" in sg:
        return bool(re.search(r"\bflow|pressure|temperature|level|visible|operator|indicat|gauge|meter|sight\b", row_text))
    if "bund" in sg:
        return bool(re.search(r"\brelease|spill|leak|overflow|environmental\b", row_text))
    if "secondary containment" in sg:
        return bool(re.search(r"\brelease|spill|leak|overflow|environmental|liquid\b", row_text))
    if "seal pot" in sg or "glycol-sealed overflow" in sg:
        return bool(re.search(r"\brelease|spill|overflow|vent|vapou?r|liquid|environmental\b", row_text))
    if "nitrogen blanketing" in sg or "inerting" in sg:
        if re.search(r"\b(?:n2|nitrogen)\b", row_text) and re.search(r"\b(?:prove|proving|supply|hose|purge)\b", row_text):
            return False
        return bool(re.search(r"\boxygen|air|flammable|fire|explosion|blanket|inert\b", row_text))
    if "fire and gas detection" in sg or "firewater" in sg or "deluge" in sg:
        return bool(re.search(r"\bfire|gas|flammable|release|leak|explosion\b", row_text))
    return False


def sanitize_known_safeguards(value: Any, limit: int = 30) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    seen: Set[str] = set()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                name = normalize_whitespace(item.get("name") or item.get("safeguard") or item.get("label"))
                category = normalize_whitespace(item.get("category"))
                evidence = normalize_whitespace(item.get("evidence"))
                status = normalize_whitespace(item.get("status")) or "documented_existing"
                source = normalize_whitespace(item.get("source"))
                applies_to = normalize_whitespace(item.get("applies_to"))
            else:
                name = normalize_whitespace(item)
                category = ""
                evidence = ""
                status = "documented_existing"
                source = ""
                applies_to = ""
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({
                "name": name,
                "category": category,
                "evidence": evidence,
                "status": status,
                "source": source,
                "applies_to": applies_to,
            })
            if len(cleaned) >= limit:
                break
    return cleaned


def extract_structured_pid_safeguards(pid_parse: Any) -> List[Dict[str, str]]:
    """Build installed-control evidence from the structured P&ID parse.

    Location is retained in the safeguard name so, for example, the J1
    discharge NRV cannot be mistaken for the temporary nitrogen-hose NRV.
    """
    if not isinstance(pid_parse, dict):
        return []
    controls: List[Dict[str, str]] = []

    def add(name: str, category: str, evidence: str, applies_to: str) -> None:
        if not name or not evidence:
            return
        controls.append({
            "name": name,
            "category": category,
            "evidence": normalize_whitespace(evidence),
            "status": "documented_existing",
            "source": "parsed_pid",
            "applies_to": normalize_whitespace(applies_to),
        })

    for device in pid_parse.get("valves_and_relief_devices") or []:
        if not isinstance(device, dict):
            continue
        tag = normalize_whitespace(device.get("tag"))
        device_type = normalize_whitespace(device.get("type"))
        location = normalize_whitespace(device.get("location"))
        evidence = normalize_whitespace(device.get("image_evidence"))
        lower = f"{tag} {device_type} {location}".lower()
        applies_to = normalize_whitespace(f"{tag} {location}")
        if "pressure/vacuum" in lower or re.search(r"\bpv\b", lower):
            add("IST pressure/vacuum valve (PV)", "mechanical", evidence, applies_to or "IST")
        elif "non-return" in lower or re.search(r"\bnrv\b", lower):
            if "j1" in lower or "discharge" in lower:
                add("J1 discharge non-return valve (NRV)", "mechanical", evidence, applies_to)
            elif "nitrogen" in lower:
                add("Nitrogen-blanket non-return valve", "mechanical", evidence, applies_to)
            else:
                add("P&ID non-return valve", "mechanical", evidence, applies_to)
        elif "restriction orifice" in lower or re.search(r"\brop\b", lower):
            add("Nitrogen-blanket restriction orifice (ROP)", "mechanical", evidence, applies_to)

    for instrument in pid_parse.get("instruments_and_controls") or []:
        if not isinstance(instrument, dict):
            continue
        tag = normalize_whitespace(instrument.get("tag"))
        instrument_type = normalize_whitespace(instrument.get("type"))
        associated = normalize_whitespace(instrument.get("associated_equipment_or_line"))
        evidence = normalize_whitespace(instrument.get("image_evidence"))
        lower = f"{tag} {instrument_type}".lower()
        applies_to = normalize_whitespace(f"{tag} {associated}")
        if re.search(r"\bpic\b", lower) or "pressure indicator/controller" in lower:
            add("IST pressure control loop (PT/PIC and split-range valves)", "instrumented", evidence, applies_to)
        elif re.search(r"\blic\b", lower) or "level indicator/controller" in lower:
            add("IST level control loop (LT/LIC/LCV)", "instrumented", evidence, applies_to)
        elif "level alarm" in lower:
            add(f"IST {instrument_type}", "instrumented", evidence, applies_to)
        elif instrument_type in {"level indicator", "pressure indicator", "temperature indicator"}:
            add(f"IST {instrument_type}", "instrumented", evidence, applies_to)

    for equipment in pid_parse.get("equipment") or []:
        if not isinstance(equipment, dict):
            continue
        equipment_type = normalize_whitespace(equipment.get("type"))
        evidence = normalize_whitespace(equipment.get("image_evidence"))
        if "flame arrest" in equipment_type.lower():
            add("IST vent flame arrester", "mechanical", evidence, "IST vent")
        elif "seal pot" in equipment_type.lower():
            add("IST seal pot", "passive", evidence, "IST overflow/vent connection")
    return sanitize_known_safeguards(controls, 60)


def safeguard_is_relevant_to_node(safeguard: Dict[str, Any], node: Dict[str, Any]) -> bool:
    """Conservatively map a documented control to its actual node boundary."""
    name = normalize_whitespace(safeguard.get("name")).lower()
    evidence = normalize_whitespace(safeguard.get("evidence")).lower()
    applies_to = normalize_whitespace(safeguard.get("applies_to")).lower()
    node_text = normalize_whitespace(" ".join([
        str(node.get("node_id", "")), str(node.get("node_name", "")),
        str(node.get("node_description", "")), str(node.get("boundaries", "")),
        str(node.get("design_intent", "")), " ".join(sanitize_list(node.get("equipment"), 40)),
        " ".join(sanitize_list(node.get("lines"), 40)),
        " ".join(sanitize_list(node.get("procedure_steps"), 40)),
    ])).lower()
    has_procedure = bool(sanitize_list(node.get("procedure_steps"), 40))

    if "temporary nitrogen-hose non-return" in name:
        return bool(re.search(r"\b(?:n2|nitrogen|hose|v4|v5|v6)\b", node_text))
    if "j1 discharge non-return" in name:
        return bool(re.search(r"\b(?:lcv|petrol blending|j1 discharge|transfer pump discharge)\b", node_text))
    if "nitrogen-blanket" in name:
        return "blanket" in node_text or "split range" in node_text
    if "lead-operator" in name or "radio communication" in name:
        return has_procedure
    if "valve labeling" in name:
        return has_procedure
    if "dummy-run" in name or "practice run" in name:
        return has_procedure
    if "audible nitrogen-breakthrough" in name:
        return "ist" in node_text and bool(re.search(r"\b(?:n2|nitrogen|blowback|clear)\b", node_text))
    if "flow restriction device" in name:
        return bool(re.search(r"\b(?:n2|nitrogen)\b", node_text))
    if any(term in name for term in (
        "pressure/vacuum", "pressure control", "level control", "level alarm",
        "level indicator", "temperature indicator", "vent flame arrester",
        "bunded enclosure", "secondary containment", "pressure relief",
        "level trip", "level alarm", "pressure alarm", "local indication",
        "blanketing", "inerting", "seal pot", "glycol-sealed overflow",
    )):
        return "ist" in node_text
    if "non-return" in name and "hose" in f"{evidence} {applies_to}":
        return bool(re.search(r"\b(?:n2|nitrogen|hose|v4|v5|v6)\b", node_text))
    # For controls not covered above, require an explicit equipment/location
    # overlap rather than propagating a facility-wide control to every node.
    tokens = set(re.findall(r"\b(?:[a-z]{1,4}\d+|ist|j1|lcv|pv|nrv|rop)\b", applies_to))
    return bool(tokens and any(token in node_text for token in tokens))


def safeguard_identity(safeguard: Dict[str, Any]) -> str:
    """Return a location-aware identity for merging duplicate evidence sources."""
    name = normalize_whitespace(safeguard.get("name")).lower()
    if "temporary nitrogen-hose" in name:
        return "temporary_nitrogen_hose_nrv"
    if "j1 discharge" in name and ("non-return" in name or "nrv" in name):
        return "j1_discharge_nrv"
    if "nitrogen-blanket" in name and "non-return" in name:
        return "nitrogen_blanket_nrv"
    if "pressure/vacuum" in name or "pressure relief" in name:
        return "ist_pressure_vacuum_relief"
    if "pressure control" in name:
        return "ist_pressure_control"
    if "level control" in name:
        return "ist_level_control"
    return name


def merge_safeguard_evidence(items: List[Dict[str, Any]], limit: int = 80) -> List[Dict[str, str]]:
    """Merge the same control from process text and P&ID without losing provenance."""
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for raw in sanitize_known_safeguards(items, limit * 2):
        key = safeguard_identity(raw)
        if key not in merged:
            merged[key] = dict(raw)
            order.append(key)
            continue
        current = merged[key]
        # Prefer the location/tag-rich P&ID label while retaining both source
        # statements. This avoids two apparent safeguards for one control.
        if raw.get("source") == "parsed_pid":
            current["name"] = raw.get("name") or current.get("name")
        for field in ("evidence", "source", "applies_to"):
            values = [normalize_whitespace(current.get(field)), normalize_whitespace(raw.get(field))]
            current[field] = " | ".join(value for index, value in enumerate(values) if value and value not in values[:index])
        if raw.get("status") == "documented_existing":
            current["status"] = "documented_existing"
    return sanitize_known_safeguards([merged[key] for key in order], limit)


def attach_evidence_safeguards_to_nodes(
    nodes: List[Dict[str, Any]],
    process_understanding: Dict[str, Any],
    pid_parse: Any,
) -> List[Dict[str, Any]]:
    """Merge process/procedure and parsed-P&ID controls into each relevant node."""
    text_controls = sanitize_known_safeguards(process_understanding.get("known_safeguards"), 80)
    pid_controls = extract_structured_pid_safeguards(pid_parse)
    candidates = sanitize_known_safeguards(text_controls + pid_controls, 120)
    enriched: List[Dict[str, Any]] = []
    for raw_node in nodes:
        node = dict(raw_node)
        confirmed = sanitize_known_safeguards(node.get("known_safeguards"), 80)
        recommended: List[Dict[str, str]] = []
        for safeguard in candidates:
            if not safeguard_is_relevant_to_node(safeguard, node):
                continue
            if safeguard.get("status") == "recommended_unconfirmed":
                recommended.append(safeguard)
            else:
                confirmed.append(safeguard)

        # A documented pre-step verification can be credited only to this
        # procedural node; it is not spread to other nodes or treated as an
        # independent hardware barrier.
        for step in sanitize_list(node.get("procedure_steps"), 40):
            if re.search(r"\bverify\b.*\bclosed\b", step, flags=re.I):
                confirmed.append({
                    "name": "Pre-depressurization valve-closure verification",
                    "category": "administrative",
                    "evidence": normalize_whitespace(step),
                    "status": "documented_existing",
                    "source": "process_procedure",
                    "applies_to": node.get("node_id", ""),
                })
        node["known_safeguards"] = merge_safeguard_evidence(confirmed, 80)
        node["recommended_controls_not_credited_as_safeguards"] = sanitize_known_safeguards(recommended, 40)
        enriched.append(node)
    return enriched


def sanitize_multiline_field(value: Any, limit: int = 12, field: str = "", compact: bool = False) -> str:
    items = row_text_items(value, limit * 2 if compact else limit)
    if compact:
        items = compact_hazop_items(items, field, limit)
    return "\n".join(items) if items else ""


def sanitize_node(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    node_id = normalize_whitespace(raw.get("node_id")) or f"N-{index:03d}"
    node_name = normalize_whitespace(raw.get("node_name")) or normalize_whitespace(raw.get("name")) or f"Node {index}"
    node: Dict[str, Any] = {
        "node_id": node_id,
        "node_name": node_name,
        "node_description": normalize_whitespace(raw.get("node_description") or raw.get("description")) or node_name,
        "boundaries": normalize_whitespace(raw.get("boundaries")),
        "design_intent": normalize_whitespace(raw.get("design_intent")) or f"Normal intended operation for {node_name}.",
        "equipment": sanitize_list(raw.get("equipment"), 30),
        "lines": sanitize_list(raw.get("lines"), 30),
        "materials": sanitize_list(raw.get("materials"), 20),
        "operating_conditions": normalize_whitespace(raw.get("operating_conditions")),
        "parameters": sanitize_list(raw.get("parameters"), 12) or list(HAZOP_PARAMETERS),
        "known_safeguards": sanitize_known_safeguards(raw.get("known_safeguards"), 30),
    }
    for key in (
        "steps",
        "procedure_initial_setup",
        "procedure_steps",
        "guideword_matrix",
        "guidewords_by_parameter",
        "manual_source",
        "manual_context",
        "study_focus",
        "worksheet_columns",
    ):
        if raw.get(key) is not None:
            node[key] = raw.get(key)
    return node


def sanitize_row(raw: Dict[str, Any], node: Dict[str, Any], index: int) -> Dict[str, Any]:
    raw_deviation = raw.get("deviation")
    # Every other worksheet column is run through compact_hazop_items to
    # enforce the short-bullet cell-length contract; deviation/effect was the
    # one field left as a raw, uncompacted string, so a verbose model response
    # here passed straight into the worksheet as a full sentence.
    deviation = sanitize_multiline_field(raw_deviation, 1, "deviation", True) if raw_deviation is not None else ""
    raw_causes = raw.get("potential_causes")
    joined_causes = "\n".join(compact_hazop_items(row_text_items(raw_causes, 12), "potential_cause", 4)) if raw_causes else ""
    raw_potential_cause = raw.get("potential_cause") if raw.get("potential_cause") is not None else raw.get("cause")
    potential_cause = sanitize_multiline_field(raw_potential_cause, 4, "potential_cause", True) or joined_causes
    raw_safeguards = raw.get("safeguards") if raw.get("safeguards") is not None else raw.get("existing_safeguards")
    joined_safeguards = "\n".join(compact_hazop_items(row_text_items(raw_safeguards, 12), "safeguard", 4)) if raw_safeguards else ""
    raw_safeguard = raw.get("safeguard")
    safeguard = sanitize_multiline_field(raw_safeguard, 4, "safeguard", True) or joined_safeguards
    raw_action = raw.get("necessary_action") if raw.get("necessary_action") is not None else raw.get("action_required")
    necessary_action = sanitize_multiline_field(raw_action, 4, "necessary_action", True)
    raw_consequence = raw.get("consequence")
    consequence = sanitize_multiline_field(raw_consequence, 4, "consequence", True)
    if not consequence and isinstance(raw.get("consequences"), list):
        consequence = "\n".join(compact_hazop_items(row_text_items(raw.get("consequences"), 12), "consequence", 4))
    if not potential_cause and not consequence:
        potential_cause = "Not applicable"
        consequence = "Not applicable"
        safeguard = safeguard or "Not applicable"
        necessary_action = necessary_action or "Not applicable"
    if not safeguard:
        safeguard = "Not identified"
    if not necessary_action:
        necessary_action = "Not identified"
    assumptions = sanitize_assumptions(raw.get("assumptions") or raw.get("basis"), 20)
    source_references = sanitize_source_references(
        raw.get("source_references")
        or raw.get("data_sources")
        or raw.get("references"),
        20,
    )
    row_origin = normalize_whitespace(raw.get("row_origin") or raw.get("origin")) or "llm_or_pipeline"
    automatically_added_row = bool(raw.get("automatically_added_row"))
    fallback_generated_row = bool(raw.get("fallback_generated_row") or raw.get("fallback_used"))
    llm_generated_row = bool(raw.get("llm_generated_row")) if raw.get("llm_generated_row") is not None else not automatically_added_row
    parent_row_id = normalize_whitespace(raw.get("parent_row_id"))
    cause_id = normalize_whitespace(raw.get("cause_id"))
    row_id = normalize_whitespace(raw.get("row_id")) or cause_id or f"{node['node_id']}-R{index:03d}"
    if cause_id and row_id != cause_id:
        parent_row_id = parent_row_id or row_id
        row_id = cause_id
    row = {
        "row_id": row_id,
        "parent_row_id": parent_row_id,
        "cause_id": cause_id or row_id,
        "cause_index": max(1, int(numeric_value(raw.get("cause_index"), 1) or 1)),
        "group_size": max(1, int(numeric_value(raw.get("group_size"), 1) or 1)),
        "node_id": node["node_id"],
        "node_name": node["node_name"],
        "node_description": normalize_whitespace(raw.get("node_description")) or node["node_description"],
        "design_intent": normalize_whitespace(raw.get("design_intent")) or node["design_intent"],
        "guideword": normalized_guideword(raw.get("guideword")),
        "parameter": row_parameter(raw, node_parameters(node)),
        "matrix_key": normalize_whitespace(raw.get("matrix_key")),
        "expected_deviation": normalize_whitespace(
            raw.get("chair_deviation")
            or raw.get("expected_deviation")
            or raw.get("effect")
            or raw.get("deviation_hint")
        ),
        "deviation": deviation or f"{normalized_guideword(raw.get('guideword'))} {row_parameter(raw, node_parameters(node))}",
        "potential_cause": potential_cause,
        "cause": potential_cause,
        "consequence": consequence,
        "safeguard": safeguard,
        "safeguards": safeguard,
        "necessary_action": necessary_action,
        "assumptions": assumptions,
        "source_references": source_references,
        "action_required": necessary_action,
        "source_round": normalize_whitespace(raw.get("source_round")) or "Round 3",
        "row_origin": row_origin,
        "llm_generated_row": llm_generated_row,
        "automatically_added_row": automatically_added_row,
        "fallback_generated_row": fallback_generated_row,
        "review_id": normalize_whitespace(raw.get("review_id")),
        "source_row_id": normalize_whitespace(raw.get("source_row_id")),
        "source_agent": normalize_whitespace(raw.get("source_agent")),
        "source_agents": sanitize_list(raw.get("source_agents"), 20),
        "source_review_ids": sanitize_list(raw.get("source_review_ids"), 80),
        "source_proposal_ids": sanitize_list(raw.get("source_proposal_ids"), 40),
        "proposal_id": normalize_whitespace(raw.get("proposal_id")),
        "chair_resolution": normalize_whitespace(raw.get("chair_resolution")),
        "chair_adjudication_id": normalize_whitespace(raw.get("chair_adjudication_id")),
        "chair_criterion_assessment": (
            raw.get("chair_criterion_assessment")
            if isinstance(raw.get("chair_criterion_assessment"), dict)
            else {}
        ),
        "round_2_panel_basis": (
            raw.get("round_2_panel_basis")
            if isinstance(raw.get("round_2_panel_basis"), dict)
            else {}
        ),
        "chair_suggested_decision": normalized_round_two_decision(
            raw.get("chair_suggested_decision")
        ),
        "chair_decision": normalized_round_two_decision(raw.get("chair_decision")),
        "chair_decision_overridden_by_rubric": bool(
            raw.get("chair_decision_overridden_by_rubric", False)
        ),
        "chair_decision_method": normalize_whitespace(raw.get("chair_decision_method")),
        "chair_final_treatment": normalize_whitespace(raw.get("chair_final_treatment")),
        "chair_evidence_basis": sanitize_list(raw.get("chair_evidence_basis"), 20),
        "chair_revision_summary": normalize_whitespace(raw.get("chair_revision_summary")),
        "chair_review_status": normalize_whitespace(raw.get("chair_review_status")),
        "chair_review_attempt": int(numeric_value(raw.get("chair_review_attempt"), 0) or 0),
        "evidence_basis": sanitize_list(raw.get("evidence_basis"), 20),
        "safeguard_evidence": sanitize_list(raw.get("safeguard_evidence"), 20),
        "safeguard_sources": [
            dict(item)
            for item in (raw.get("safeguard_sources") or [])
            if isinstance(item, dict)
        ],
        "safeguard_identification_status": normalize_whitespace(
            raw.get("safeguard_identification_status")
        ),
        "final_status": normalize_whitespace(raw.get("final_status")),
        "rejection_reason": normalize_whitespace(raw.get("rejection_reason")),
        "duplicate_of_row_id": normalize_whitespace(raw.get("duplicate_of_row_id")),
        "round_3_candidate_origin": normalize_whitespace(
            raw.get("round_3_candidate_origin")
        ),
        "chair_authorized": bool(raw.get("chair_authorized", True)),
    }
    return complete_worksheet_row(row, node)


def guideword_meaning(guideword: str, parameter: str = "") -> str:
    canonical = normalized_guideword_no_lookup(guideword)
    parameter = normalize_whitespace(parameter)
    meanings = {
        "No": "Complete absence of the intended condition.",
        "Zero": "Zero or no intended quantity for the parameter.",
        "Less": "Quantitative decrease from the design intent.",
        "Low": "Parameter lower than the intended operating condition.",
        "More": "Quantitative increase from the design intent.",
        "High": "Parameter higher than the intended operating condition.",
        "Part of": "Only part of the design intent is achieved.",
        "As well as": "Additional material, phase, path, or condition occurs.",
        "Reverse": "Logical opposite of the design intent.",
        "Other than": "Different material, route, condition, or operating state.",
        "Contamination": "Unwanted contaminant or wrong material is present.",
    }
    base = meanings.get(canonical, canonical or "Required guideword")
    return f"{base} Parameter: {parameter}." if parameter else base


def normalized_guideword_no_lookup(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    aliases = {
        "zero": "Zero",
        "nil": "Zero",
        "none": "Zero",
        "high": "High",
        "higher": "High",
        "too high": "High",
        "low": "Low",
        "lower": "Low",
        "too low": "Low",
        "no": "No",
        "less": "Less",
        "more": "More",
        "part of": "Part of",
        "partial": "Part of",
        "as well as": "As well as",
        "reverse": "Reverse",
        "other than": "Other than",
        "contamination": "Contamination",
    }
    return aliases.get(text, normalize_whitespace(value))


def guideword_base_label(value: Any) -> str:
    text = normalize_whitespace(value)
    previous = None
    while previous != text:
        previous = text
        text = normalize_whitespace(re.sub(r"\s*\([^)]*\)\s*$", "", text))
    return text


def normalized_guideword(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    aliases = {
        "zero": "Zero",
        "nil": "Zero",
        "none": "Zero",
        "high": "High",
        "higher": "High",
        "too high": "High",
        "low": "Low",
        "lower": "Low",
        "too low": "Low",
        "no": "No",
        "less": "Less",
        "more": "More",
        "part of": "Part of",
        "partial": "Part of",
        "as well as": "As well as",
        "reverse": "Reverse",
        "other than": "Other than",
        "contamination": "Contamination",
    }
    if text in aliases:
        return aliases[text]
    for guideword, _ in HAZOP_GUIDEWORDS:
        if text == guideword.lower():
            return guideword
    return normalize_whitespace(value)


def node_parameters(node: Dict[str, Any]) -> List[str]:
    return [
        normalize_whitespace(parameter).lower()
        for parameter in (node.get("parameters") or HAZOP_PARAMETERS)
        if normalize_whitespace(parameter)
    ] or list(HAZOP_PARAMETERS)


def guideword_parameter_matrix(node: Dict[str, Any]) -> List[Dict[str, str]]:
    passthrough_keys = (
        "cause",
        "causes",
        "potential_cause",
        "potential_causes",
        "consequence",
        "consequences",
        "safeguard",
        "safeguards",
        "existing_safeguards",
        "protection",
        "necessary_action",
        "action",
        "actions",
        "recommendation",
        "recommended_action",
        "basis",
        "reference",
        "source",
    )

    def with_matrix_evidence(base: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        for key in passthrough_keys:
            if key in source:
                base[key] = source.get(key)
        return base

    explicit_matrix = node.get("guideword_matrix")
    if isinstance(explicit_matrix, list) and explicit_matrix:
        matrix = []
        for item in explicit_matrix:
            if not isinstance(item, dict):
                continue
            guideword = normalized_guideword(item.get("guideword"))
            parameter = normalize_whitespace(item.get("parameter")).lower()
            if not guideword or not parameter:
                continue
            chair_deviation = normalize_whitespace(
                item.get("chair_deviation")
                or item.get("chair_effect")
                or item.get("chair_defined_effect")
                or item.get("chair_defined_deviation")
            )
            deviation_hint = normalize_whitespace(
                chair_deviation
                or item.get("deviation_hint")
                or item.get("expected_deviation")
                or item.get("effect")
            )
            if node.get("manual_source") and parameter == "procedure" and deviation_hint:
                guideword = procedural_guideword_label(guideword, deviation_hint)
            matrix_key = normalize_whitespace(item.get("matrix_key")) or f"{parameter}:{guideword}:{deviation_hint or guideword}"
            matrix.append(with_matrix_evidence({
                "guideword": guideword,
                "meaning": normalize_whitespace(item.get("meaning")) or guideword_meaning(guideword, parameter),
                "chair_deviation": chair_deviation,
                "chair_effect_basis": normalize_whitespace(item.get("chair_effect_basis")),
                "parameter": parameter,
                "deviation_hint": deviation_hint,
                "expected_deviation": deviation_hint,
                "effect": deviation_hint,
                "matrix_key": matrix_key,
                "required": "yes",
            }, item))
        if matrix:
            return matrix

    by_parameter = node.get("guidewords_by_parameter")
    if isinstance(by_parameter, dict) and by_parameter:
        matrix = []
        for parameter in node_parameters(node):
            guidewords = by_parameter.get(parameter) or by_parameter.get(parameter.lower()) or []
            if not isinstance(guidewords, list):
                guidewords = []
            for entry in guidewords:
                if isinstance(entry, dict):
                    guideword = normalized_guideword(entry.get("guideword"))
                    meaning = normalize_whitespace(entry.get("meaning")) or guideword_meaning(guideword, parameter)
                else:
                    guideword = normalized_guideword(entry)
                    meaning = guideword_meaning(guideword, parameter)
                if not guideword:
                    continue
                deviation_hint = normalize_whitespace(
                    entry.get("chair_deviation")
                    or entry.get("deviation_hint")
                    or entry.get("expected_deviation")
                    or entry.get("effect")
                ) if isinstance(entry, dict) else ""
                if node.get("manual_source") and parameter == "procedure" and deviation_hint:
                    guideword = procedural_guideword_label(guideword, deviation_hint)
                matrix_key = f"{parameter}:{guideword}:{deviation_hint or guideword}"
                item_source = entry if isinstance(entry, dict) else {}
                matrix.append(with_matrix_evidence({
                    "guideword": guideword,
                    "meaning": meaning,
                    "parameter": parameter,
                    "deviation_hint": deviation_hint,
                    "expected_deviation": deviation_hint,
                    "effect": deviation_hint,
                    "matrix_key": matrix_key,
                    "required": "yes",
                }, item_source))
        if matrix:
            return matrix

    return [
        {
            "guideword": guideword,
            "meaning": meaning,
            "parameter": parameter,
            "deviation_hint": "",
            "expected_deviation": "",
            "effect": "",
            "matrix_key": f"{parameter}:{guideword}",
            "required": "yes",
        }
        for parameter in node_parameters(node)
        for guideword, meaning in HAZOP_GUIDEWORDS
    ]


def guideword_matrix_text(node: Dict[str, Any]) -> str:
    lines = []
    for item in guideword_parameter_matrix(node):
        qualifier = normalize_whitespace(item.get("qualifier") or item.get("deviation_hint"))
        expected = normalize_whitespace(item.get("chair_deviation") or item.get("expected_deviation") or item.get("effect"))
        chair_effect = normalize_whitespace(item.get("chair_deviation"))
        chair_basis = normalize_whitespace(item.get("chair_effect_basis"))
        parts = [f"- matrix_key={item.get('matrix_key', '')} | {item['parameter']} / {item['guideword']}"]
        if qualifier and qualifier != expected:
            parts.append(f" | qualifier/context={qualifier} (classification only; do not copy into Deviation/Effect)")
        if chair_effect:
            parts.append(f" | Chair-defined Deviation/Effect={chair_effect}")
        if chair_basis:
            parts.append(f" | Chair basis={chair_basis}")
        if expected:
            parts.append(f" | supplied expected deviation/effect={expected}")
        parts.append(f": {item['meaning']}")
        lines.append("".join(parts))
    return "\n".join(lines)


def fallback_chair_effect_for_matrix_item(item: Dict[str, Any], node: Dict[str, Any]) -> str:
    guideword = normalize_whitespace(item.get("guideword"))
    parameter = normalize_whitespace(item.get("parameter")).lower()
    existing = normalize_whitespace(
        item.get("chair_deviation")
        or item.get("deviation_hint")
        or item.get("expected_deviation")
        or item.get("effect")
    )
    sequence_deviation = procedural_sequence_deviation(
        guideword,
        parameter,
        item.get("qualifier") or existing,
        node,
    ) if node.get("manual_source") else ""
    if sequence_deviation:
        return sequence_deviation
    if existing and existing.lower() not in {"too early", "too late", "too fast", "too slow", "more", "less", "communication", "reverse", "incomplete", "others"}:
        return existing
    boundary = normalize_whitespace(node.get("boundaries")) or normalize_whitespace(node.get("node_name")) or "the node"
    material = (sanitize_list(node.get("materials"), 1) or ["material"])[0]
    service = normalize_whitespace(material) if material else "material"
    lower = guideword.lower()
    if "too fast" in lower or (lower == "rate" and "too fast" in existing.lower()):
        return f"High {service} flow or operation rate through {boundary}"
    if "too slow" in lower or (lower == "rate" and "too slow" in existing.lower()):
        return f"Low {service} flow or delayed operation through {boundary}"
    if (lower.startswith("magnitude") and "less" in lower) or lower == "less":
        return f"Less {service} than required through {boundary}"
    if (lower.startswith("magnitude") and "more" in lower) or lower == "more":
        return f"More {service} than required through {boundary}"
    if lower.startswith("pressure") or parameter == "pressure":
        return f"High pressure in {boundary}"
    if lower.startswith("communication"):
        return f"Incorrect operator instruction affecting {boundary}"
    if lower.startswith("reverse"):
        return f"Reverse flow through {boundary}"
    if lower.startswith("incomplete"):
        return f"Incomplete execution of the intended operation at {boundary}"
    if lower.startswith("out of sequence") and "too late" in lower:
        return f"Required operation at {boundary} occurs too late for the design intent"
    if lower.startswith("out of sequence") and "too early" in lower:
        return f"Required operation at {boundary} occurs too early for the design intent"
    if lower.startswith("out of sequence"):
        return f"Operation at {boundary} occurs out of the intended sequence"
    if lower.startswith("others"):
        return f"Other credible deviation affecting {boundary}"
    if parameter:
        return fallback_deviation_phrase(guideword, parameter)
    return normalize_whitespace(existing) or f"{guideword} deviation in {boundary}"


def fallback_deviation_phrase(guideword: Any, parameter: Any) -> str:
    guideword_text = normalize_whitespace(guideword)
    parameter_text = normalize_whitespace(parameter)
    guideword_lower = guideword_text.lower()
    parameter_lower = parameter_text.lower()
    if guideword_lower in {"no", "zero"}:
        return f"No {parameter_lower}".strip()
    if guideword_lower in {"less", "low"}:
        return f"Less {parameter_lower}".strip()
    if guideword_lower in {"more", "high"}:
        return f"More {parameter_lower}".strip()
    if guideword_lower == "reverse":
        return f"Reverse {parameter_lower}".strip()
    if guideword_lower in {"part of", "partial"}:
        return f"Partial {parameter_lower}".strip()
    if guideword_lower == "as well as":
        return f"Additional {parameter_lower}".strip()
    if guideword_lower == "other than":
        return f"Other {parameter_lower}".strip()
    return f"{guideword_text} {parameter_text}".strip()


def weak_deviation_text(value: Any) -> bool:
    text = normalize_whitespace(value).lower().strip(" .;:")
    if not text:
        return True
    return (
        field_is_missing(text)
        or text in {
            "not applicable",
            "too early",
            "too late",
            "too fast",
            "too slow",
            "more",
            "less",
            "communication",
            "reverse",
            "incomplete",
            "others",
            "other",
            "out of sequence sequence",
            "effect not identified",
            "deviation not identified",
        }
        or text.startswith("effect not ")
        or text.startswith("deviation not ")
    )


def matrix_item_has_supplied_evidence(item: Dict[str, Any]) -> bool:
    """Return True when a matrix item carries same-row evidence from uploaded study material."""
    row_evidence_keys = (
        "cause",
        "causes",
        "potential_cause",
        "potential_causes",
        "consequence",
        "consequences",
        "safeguard",
        "safeguards",
        "existing_safeguards",
        "necessary_action",
        "action",
        "actions",
        "action_required",
        "basis",
        "reference",
        "source",
        "worksheet_row",
        "worksheet_evidence",
        "manual_evidence",
    )
    for key in row_evidence_keys:
        value = item.get(key)
        if isinstance(value, list):
            if any(normalize_whitespace(v) for v in value):
                return True
        elif isinstance(value, dict):
            if any(normalize_whitespace(v) for v in value.values()):
                return True
        elif normalize_whitespace(value):
            return True
    for key in ("deviation", "effect", "expected_deviation", "deviation_hint", "chair_deviation"):
        text = normalize_whitespace(item.get(key))
        if text and not weak_deviation_text(text):
            return True
    return False


def procedural_scope_context(node: Dict[str, Any]) -> Dict[str, Any]:
    scope_text = normalize_whitespace(" ".join([
        str(node.get("node_name") or ""),
        str(node.get("node_description") or ""),
        str(node.get("boundaries") or ""),
        str(node.get("design_intent") or ""),
        str(node.get("operating_conditions") or ""),
        " ".join(sanitize_list(node.get("materials"), 40)),
        " ".join(sanitize_list(node.get("equipment"), 80)),
        " ".join(sanitize_list(node.get("lines"), 40)),
        " ".join(sanitize_list(node.get("procedure_steps"), 80)),
        " ".join(sanitize_list(node.get("procedure_initial_setup"), 40)),
    ]))
    lower = scope_text.lower()
    boundary = (
        normalize_whitespace(node.get("boundaries"))
        or normalize_whitespace(node.get("node_name"))
        or "the node"
    )
    tags = []
    for tag in re.findall(
        r"\b(?:V|P|L|H|C|T|R|E|J|LIC|TIC|FIC|PIC|FC|LC|TC|PT|LT|FT|PV|PRV|PSV|NRV)[- ]?\d+[A-Z]?\b|\bIST\b",
        scope_text,
        flags=re.IGNORECASE,
    ):
        normalized = tag.upper().replace(" ", "-")
        if normalized not in tags:
            tags.append(normalized)
    valve_tags = [tag for tag in tags if re.match(r"^V[- ]?\d+[A-Z]?$", tag, flags=re.IGNORECASE)]
    action_tags = valve_tags or tags
    utility_service = "N2" if re.search(r"\b(?:n2|nitrogen)\b", lower) else ""
    process_service = (
        "petroleum feedstock"
        if re.search(r"\b(?:c6|petroleum|feedstock)\b", lower)
        else ""
    )
    material_service = (sanitize_list(node.get("materials"), 1) or ["service"])[0]
    return {
        "scope_text": scope_text,
        "lower": lower,
        "boundary": boundary,
        "tags": tags,
        "valve_tags": valve_tags,
        "tag_pair": "/".join(action_tags[:2]) if action_tags else boundary,
        "primary_tag": action_tags[0] if action_tags else "the relevant valve",
        "secondary_tag": action_tags[1] if len(action_tags) > 1 else "the downstream valve",
        "service": utility_service or process_service or material_service,
        "utility_service": utility_service,
        "process_service": process_service or material_service,
        "operation": (
            "blowback"
            if "blowback" in lower
            else (
                "proving"
                if re.search(r"\bprove|proving\b", lower)
                else (
                    "purging"
                    if re.search(r"\bpurge|purging|displace|displacement\b", lower)
                    else "operation"
                )
            )
        ),
        "nitrogen_service": bool(re.search(r"\b(?:n2|nitrogen)\b", lower)),
        "hose_air_displacement": "hose" in lower and ("air" in lower or "displace" in lower),
        "blowback_to_ist": bool("blowback" in lower and "ist" in lower),
        "feedstock_service": bool(re.search(r"\b(?:c6|petroleum|feedstock)\b", lower)),
    }


def procedural_guideword_category(guideword: Any, parameter: Any = "", qualifier: Any = "") -> str:
    text = normalize_whitespace(" ".join([str(guideword or ""), str(parameter or ""), str(qualifier or "")])).lower()
    if "communication" in text:
        return "communication"
    if "reverse" in text:
        return "reverse"
    if "pressure" in text:
        return "pressure_more" if "more" in text or "high" in text else "pressure"
    if "rate" in text or "too fast" in text or "too slow" in text:
        return "rate_fast" if "fast" in text else "rate_slow" if "slow" in text else "rate"
    if "magnitude" in text or re.search(r"\b(?:more|less)\b", text):
        return "magnitude_more" if "more" in text else "magnitude_less" if "less" in text else "magnitude"
    if "incomplete" in text or "completion" in text:
        return "incomplete"
    if "other" in text or "others" in text:
        return "other"
    if "too early" in text:
        return "sequence_early"
    if "too late" in text:
        return "sequence_late"
    if "out of sequence" in text or "sequence" in text:
        return "sequence"
    return ""


def procedural_sequence_deviation(guideword: Any, parameter: Any, qualifier: Any, node: Dict[str, Any]) -> str:
    """Derive manual-HAZOP style effects for procedural guideword rows."""
    category = procedural_guideword_category(guideword, parameter, qualifier)
    if not category:
        return ""
    context = procedural_scope_context(node)
    boundary = context["boundary"]
    service = context["service"]
    operation = context["operation"]
    if category == "sequence_early":
        return f"{operation.capitalize()} starts before {boundary} is ready"
    if category == "sequence_late":
        return f"{operation.capitalize()} delayed, leaving {service} in {boundary}"
    if category == "sequence":
        return f"{operation.capitalize()} occurs outside the required sequence for {boundary}"
    if category == "rate_fast":
        return f"High {service} flow rate through {boundary}"
    if category == "rate_slow":
        return f"Low {service} flow rate through {boundary}"
    if category == "rate":
        return f"Abnormal {service} flow rate through {boundary}"
    if category == "pressure_more":
        return f"High pressure in {boundary}"
    if category == "pressure":
        return f"Abnormal pressure in {boundary}"
    if category == "magnitude_more":
        return f"Excessive {service} quantity or duration through {boundary}"
    if category == "magnitude_less":
        return f"Insufficient {service} quantity or duration through {boundary}"
    if category == "magnitude":
        return f"Incorrect {service} quantity or duration through {boundary}"
    if category == "communication":
        return f"Incorrect operator instruction for {operation} at {boundary}"
    if category == "reverse":
        return f"Reverse flow through {boundary}"
    if category == "incomplete":
        return f"Incomplete {operation} at {boundary}"
    if category == "other":
        return f"Wrong or unavailable service for {operation} at {boundary}"
    return f"Credible abnormal {operation} state at {boundary}"


def procedural_sequence_recovery_cells(guideword: Any, node: Dict[str, Any]) -> Dict[str, str]:
    """Conservative default cells when a required procedural row was wrongly marked N/A."""
    category = procedural_guideword_category(guideword)
    if not category:
        return {}
    context = procedural_scope_context(node)
    boundary = context["boundary"]
    service = context["service"]
    process_service = context["process_service"]
    residual_service = process_service if context["blowback_to_ist"] or "clear" in context["lower"] else service
    operation = context["operation"]
    tag_pair = context["tag_pair"]
    primary_tag = context["primary_tag"]
    secondary_tag = context["secondary_tag"]
    if category == "sequence_early":
        return {
            "potential_cause": f"{primary_tag} opened before required {tag_pair} lineup is confirmed",
            "consequence": f"{operation.capitalize()} design intent not achieved at {boundary}",
            "necessary_action": f"Add hold point to confirm {tag_pair} lineup before {operation}",
        }
    if category == "sequence_late":
        return {
            "potential_cause": f"{operation.capitalize()} delayed or {tag_pair} closure delayed",
            "consequence": f"{service.capitalize()} remains in {boundary} longer than intended",
            "necessary_action": f"Define completion timing and closure criteria for {tag_pair}",
        }
    if category == "rate_fast":
        return {
            "potential_cause": f"{primary_tag} opened too quickly or too far",
            "consequence": f"Mechanical stress or excessive discharge during {operation}",
            "necessary_action": f"Specify controlled opening rate for {primary_tag} during {operation}",
        }
    if category == "rate_slow":
        return {
            "potential_cause": f"{primary_tag} insufficiently opened or {service} supply restricted",
            "consequence": f"Incomplete {operation} of {boundary}",
            "necessary_action": f"Define {service} flow confirmation for {operation}",
        }
    if category in {"pressure", "pressure_more"}:
        return {
            "potential_cause": f"{primary_tag} opened against closed or restricted {secondary_tag}",
            "consequence": f"Overpressure of {boundary}",
            "necessary_action": f"Verify pressure protection and indication for {boundary}",
        }
    if category == "magnitude_more":
        return {
            "potential_cause": f"{primary_tag} left open longer than required",
            "consequence": f"Excessive {service} transfer or discharge during {operation}",
            "necessary_action": f"Define maximum {operation} duration or quantity criterion",
        }
    if category == "magnitude_less":
        return {
            "potential_cause": f"{primary_tag} closed before sufficient {operation} is completed",
            "consequence": f"Residual {residual_service} remains in {boundary}",
            "necessary_action": f"Define minimum {operation} duration or completion criterion",
        }
    if category == "communication":
        return {
            "potential_cause": f"Operator instruction misunderstood during {operation}",
            "consequence": f"Incorrect {tag_pair} sequence or incomplete {operation}",
            "necessary_action": f"Define communication hold point for {operation} at {boundary}",
        }
    if category == "reverse":
        return {
            "potential_cause": f"{secondary_tag} left open or {primary_tag} passing after {operation}",
            "consequence": f"Reverse flow or contamination through {boundary}",
            "necessary_action": f"Verify closure sequence and reverse-flow protection for {boundary}",
        }
    if category == "incomplete":
        return {
            "potential_cause": f"{tag_pair} not fully aligned or {service} supply insufficient",
            "consequence": f"{operation.capitalize()} remains incomplete at {boundary}",
            "necessary_action": f"Add positive completion verification for {operation}",
        }
    if category == "other":
        return {
            "potential_cause": f"Wrong utility connected or required {service} supply unavailable",
            "consequence": f"{operation.capitalize()} design intent not achieved at {boundary}",
            "necessary_action": f"Verify utility identity and availability before {operation}",
        }
    return {}


def apply_chair_effect_definitions(node: Dict[str, Any], definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
    updated = dict(node)
    matrix = [dict(item) for item in guideword_parameter_matrix(node)]
    definition_by_key: Dict[str, Dict[str, Any]] = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        key = normalize_whitespace(definition.get("matrix_key"))
        if key:
            definition_by_key[key] = definition
    for item in matrix:
        definition = definition_by_key.get(normalize_whitespace(item.get("matrix_key")), {})
        deviation = normalize_whitespace(
            definition.get("deviation")
            or definition.get("effect")
            or definition.get("chair_deviation")
        )
        model_deviation = deviation
        applicability = normalize_whitespace(definition.get("applicability")).lower()
        item_has_evidence = matrix_item_has_supplied_evidence(item)
        sequence_deviation = procedural_sequence_deviation(
            item.get("guideword"),
            item.get("parameter"),
            item.get("qualifier") or item.get("deviation_hint") or item.get("expected_deviation"),
            updated,
        ) if updated.get("manual_source") else ""
        if not deviation or weak_deviation_text(deviation):
            deviation = sequence_deviation or fallback_chair_effect_for_matrix_item(item, updated)
        if applicability == "not_applicable" and sequence_deviation:
            deviation = sequence_deviation
            applicability = "credible"
        if applicability == "not_applicable" and item_has_evidence and (
            weak_deviation_text(model_deviation) or weak_deviation_text(deviation)
        ):
            deviation = fallback_chair_effect_for_matrix_item(item, updated)
            applicability = "credible"
        if applicability == "not_applicable" and weak_deviation_text(deviation):
            deviation = "Not applicable"
        item["chair_deviation"] = deviation
        item["chair_effect_applicability"] = applicability or ("credible" if not weak_deviation_text(deviation) else "not_applicable")
        item["chair_effect_basis"] = normalize_whitespace(definition.get("basis"))
        item["deviation_hint"] = deviation
        item["expected_deviation"] = deviation
        item["effect"] = deviation
    updated["guideword_matrix"] = matrix
    return updated


def node_scope_gate_text(node: Dict[str, Any]) -> str:
    """Compact node-scope instructions injected into every HAZOP round."""
    boundary = normalize_whitespace(node.get("boundaries"))
    equipment = ", ".join(sanitize_list(node.get("equipment"), 30)) or "None listed"
    lines = ", ".join(sanitize_list(node.get("lines"), 30)) or "None listed"
    materials = ", ".join(sanitize_list(node.get("materials"), 20)) or "None listed"
    steps = sanitize_list(node.get("procedure_steps"), 12)
    step_text = "\n".join(f"- {step}" for step in steps) if steps else "- None listed"
    return f"""
NODE-SCOPE GATE:
- Current node: {normalize_whitespace(node.get("node_id"))} - {normalize_whitespace(node.get("node_name"))}.
- Boundary: {boundary or "Not provided"}.
- Design intention: {normalize_whitespace(node.get("design_intent")) or "Not provided"}.
- Node equipment/tags: {equipment}.
- Node lines: {lines}.
- Node materials: {materials}.
- Node-specific procedure steps:
{step_text}
- Before accepting any cause, effect, safeguard, or action, verify that it belongs to this node boundary and design intention.
- If a tag appears only elsewhere in the procedure or P&ID and is not part of this node boundary, do not use it as a cause/effect/action for this node.
- If a row needs adjacent-node context only to explain a consequence path, mention the current-node effect first and keep the adjacent tag out of the final worksheet unless the consequence cannot be understood without it.
""".strip()


def row_parameter(row: Dict[str, Any], fallback_parameters: List[str]) -> str:
    parameter = normalize_whitespace(row.get("parameter")).lower()
    normalized_fallbacks = [normalize_whitespace(candidate).lower() for candidate in fallback_parameters if normalize_whitespace(candidate)]
    if parameter:
        for candidate in normalized_fallbacks:
            if parameter == candidate or re.search(rf"\b{re.escape(candidate)}\b", parameter):
                return candidate
            if re.search(rf"\b{re.escape(parameter)}\b", candidate):
                return candidate
        for canonical in HAZOP_PARAMETERS:
            canonical_lower = canonical.lower()
            if parameter == canonical_lower or re.search(rf"\b{re.escape(canonical_lower)}\b", parameter):
                return canonical_lower
        return parameter
    deviation = normalize_whitespace(row.get("deviation")).lower()
    for candidate in normalized_fallbacks:
        if re.search(rf"\b{re.escape(candidate)}\b", deviation):
            return candidate
    for canonical in HAZOP_PARAMETERS:
        canonical_lower = canonical.lower()
        if re.search(rf"\b{re.escape(canonical_lower)}\b", deviation):
            return canonical_lower
    return normalized_fallbacks[0] if normalized_fallbacks else ""


def is_placeholder_cell_item(value: Any) -> bool:
    text = normalize_whitespace(value).lower().strip(" .;:")
    return text in {
        "n/a",
        "na",
        "none",
        "none listed",
        "not applicable",
        "not identified",
        "not provided",
        "not available",
        "deviation not identified",
        "effect not identified",
        "cause not identified",
        "consequence not identified",
        "safeguard not identified",
        "action not identified",
        "no cause identified",
        "no consequence identified",
        "no safeguard identified",
        "no safeguards identified",
        "no action identified",
    }


def row_text_items(value: Any, limit: int = 20) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value or "").replace("\r", "\n").strip()
        text = re.sub(r"[ \t\f\v]+", " ", text)
        raw_items = None
        if re.match(r"^\s*\[\s*['\"]", text):
            try:
                parsed_literal = ast.literal_eval(text)
                if isinstance(parsed_literal, list):
                    raw_items = parsed_literal
            except (SyntaxError, ValueError):
                raw_items = None
        if raw_items is None:
            raw_items = re.split(r"\n+|(?:^|\s+)\d+\.\s+", text)
    if raw_items is None:
        raw_items = []
    items: List[str] = []
    seen = set()
    for item in raw_items:
        text = normalize_whitespace(item)
        text = text.strip("[]")
        if not text:
            continue
        key = text.lower().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
        if len(items) >= limit * 2:
            break
    if any(not is_placeholder_cell_item(item) for item in items):
        items = [item for item in items if not is_placeholder_cell_item(item)]
    return items[:limit]


def compact_hazop_phrase(text: Any, field: str = "") -> str:
    phrase = normalize_whitespace(text)
    if not phrase:
        return ""
    phrase = phrase.strip("[]")
    normalized = phrase.lower().strip(" .")
    constants = {
        "n/a": "Not applicable",
        "na": "Not applicable",
        "none": "Not identified" if field in {"safeguard", "safeguards"} else "Not applicable",
        "not applicable": "Not applicable",
        "not identified": "Not identified",
        "requires verification": "Not identified" if field in {"safeguard", "safeguards"} else "Requires verification",
    }
    if normalized in constants:
        return constants[normalized]
    phrase = re.sub(r"^\s*[-*•]\s*", "", phrase)
    phrase = re.sub(r"^\s*\d+[.)]\s*", "", phrase)
    phrase = re.sub(r"\b([A-Za-z][A-Za-z0-9_-]*)\s+\1\b", r"\1", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bout\s+of\s+sequence\s+sequence\b", "Out of sequence", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\brate\s+rate\b", "Rate", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bmagnitude\s+magnitude\b", "Magnitude", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\bpressure\s+pressure\b", "Pressure", phrase, flags=re.IGNORECASE)
    if field in {"safeguard", "safeguards"}:
        phrase = re.sub(
            r"\s*\((?:administrative|independent technical safeguard|independent safeguard|technical safeguard|not independent|requires verification)\)\s*",
            "",
            phrase,
            flags=re.IGNORECASE,
        )
        if re.match(r"^not identified\s*\([^)]*\)\s*$", phrase, flags=re.IGNORECASE):
            return "Not identified"
    phrase = re.sub(r"\s*\bbefore safeguards act\b\s*", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\s*\bfor this exact node\b\s*", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\s*\bfor this node\b\s*", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"^(?:complete|specific|documented|existing|compact)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"^(?:the\s+)?(?:potential\s+)?cause\s+(?:is|may be|could be)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"^(?:the\s+)?consequence\s+(?:is|may be|could be)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"^(?:the\s+)?safeguard\s+(?:is|may be|could be)\s+", "", phrase, flags=re.IGNORECASE)
    if field in {"safeguard", "safeguards"} and phrase.lower().startswith("requires verification"):
        return "Not identified"
    if field in {"necessary_action", "action_required"}:
        phrase = re.sub(r"^(?:the\s+)?(?:team|panel)\s+should\s+", "", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"^consider\s+whether\s+", "Consider ", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"^review\s+whether\s+", "Review ", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"^verify\s+whether\s+", "Verify ", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"^requires verification:\s*", "Verify ", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\bin order to\b", "to", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        referral = re.match(r"^(refer to .+? engineering team for further study)\b", phrase, flags=re.IGNORECASE)
        if referral:
            return referral.group(1).strip(" ,.;:")
        first_action = re.split(r"\s*;\s+", phrase, maxsplit=1)[0].strip()
        if 3 <= len(first_action.split()) <= 10:
            phrase = first_action
    # Target bullet lengths are a prompt-side contract (see OUTPUT_FORMAT_RULES:
    # "never truncate mechanically and never split a claim at a comma just to
    # hit a word count"). This function used to enforce that target here by
    # cutting the phrase at the first clause/sentence boundary and discarding
    # the remainder -- which silently deleted a second independent clause
    # rather than shortening filler, the exact mechanical truncation the
    # prompt promises never to do. Never discard content here; only clean
    # formatting. A too-long phrase is left intact rather than cut.
    return phrase.strip(" ,.;:")


def is_incomplete_hazop_fragment(text: Any) -> bool:
    phrase = normalize_whitespace(text).strip(" ,.;:")
    if not phrase:
        return True
    lower = phrase.lower()
    if phrase.count("(") > phrase.count(")") or phrase.count("[") > phrase.count("]"):
        return True
    dangling_terms = {
        "and", "or", "of", "to", "for", "with", "without", "due", "because",
        "if", "when", "where", "against", "between", "from", "by", "via", "into",
        "as", "than", "then", "while", "until", "after", "before",
    }
    words = lower.split()
    if words and words[-1].strip(" ,.;:") in dangling_terms:
        return True
    if re.search(r"\b(?:if|when|where|because|against|between|from|to|by|with|without|via)\s+(?:a|an|the|v\d+|l\d+|p\d+|c\d+|h\d+|t\d+)$", lower):
        return True
    if re.search(r"\b(?:could|would|should|may|might|must|can|cannot|will)\s*$", lower):
        return True
    if re.match(
        r"^(?:and|or|but|because|which|that|where|when|while|causing|leading\s+to|"
        r"resulting\s+in|thereby|therefore|due\s+to)\b",
        lower,
    ):
        return True
    return False


def hazop_fragment_continues_previous(text: Any) -> bool:
    """Return True when a model-created list item is a clause continuation."""
    lower = normalize_whitespace(text).lower().lstrip(" ,.;:")
    return bool(re.match(
        r"^(?:and|or|but|because|which|that|where|when|while|causing|leading\s+to|"
        r"resulting\s+in|thereby|therefore|due\s+to)\b",
        lower,
    ))


def hazop_fragment_expects_continuation(text: Any) -> bool:
    """Return True when the preceding clause ends with a dangling connector."""
    lower = normalize_whitespace(text).lower().rstrip(" ,.;:")
    return bool(re.search(
        r"\b(?:and|or|but|because|which|that|where|when|while|to|for|with|without|"
        r"from|into|due\s+to|leading\s+to|resulting\s+in)$",
        lower,
    ))


def compact_hazop_items(items: List[str], field: str = "", limit: int = 8) -> List[str]:
    # Repair accidental model list fragmentation before validating bullets.
    # A continuation clause is part of the previous technical statement, not
    # a new cause/consequence/safeguard/action bullet.
    repaired: List[str] = []
    for item in items:
        phrase = compact_hazop_phrase(item, field)
        if not phrase:
            continue
        if repaired and (
            hazop_fragment_continues_previous(phrase)
            or hazop_fragment_expects_continuation(repaired[-1])
        ):
            separator = ", " if hazop_fragment_continues_previous(phrase) else " "
            repaired[-1] = f"{repaired[-1].rstrip(' ,.;:')}{separator}{phrase.lstrip(' ,.;:')}"
        else:
            repaired.append(phrase)

    output: List[str] = []
    seen = set()
    # No punctuation-based splitting here. Separating a genuine second idea
    # from punctuation inside one technical claim needs language judgment, not
    # regex. The prompt contract requires one idea per JSON array element; this
    # sanitizer preserves model meaning and only removes incomplete/duplicate
    # fragments instead of inventing a new structure.
    for phrase in repaired:
        if is_incomplete_hazop_fragment(phrase):
            continue
        key = phrase.lower().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        output.append(phrase)
        if len(output) >= limit:
            break
    if any(not is_placeholder_cell_item(item) for item in output):
        output = [item for item in output if not is_placeholder_cell_item(item)]
    return output


TECHNICAL_DEDUPE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "during",
    "for", "from", "if", "in", "into", "is", "it", "of", "on", "or",
    "that", "the", "their", "then", "through", "to", "when", "while",
    "with", "without", "after", "within", "around", "toward", "towards",
    "too",
    "required", "specified", "actual", "potential", "possible", "may",
    "might", "can", "could", "would", "should", "remains", "remain",
    "require", "lead", "its", "attempted",
}

TECHNICAL_TOKEN_NORMALIZATION = {
    "nitrogen": "n2",
    "proving": "prove", "proved": "prove", "proof": "prove",
    "purging": "purge", "purged": "purge",
    "establish": "start", "establishes": "start", "established": "start",
    "initiate": "start", "initiates": "start", "initiated": "start",
    "displacing": "displace", "displaced": "displace", "displacement": "displace",
    "closing": "close", "closed": "close", "closes": "close", "closure": "close", "shutoff": "close",
    "opening": "open", "opened": "open",
    "delayed": "delay", "delays": "delay",
    "continued": "continue", "continues": "continue", "continuing": "continue",
    "excessive": "excess", "excessively": "excess",
    "quickly": "rapid", "rapidly": "rapid", "far": "excess",
    "insufficient": "low", "inadequate": "low", "inadequately": "low",
    "blocked": "block", "blockage": "block", "obstructed": "block",
    "obstruction": "block", "obstructs": "block",
    "leaking": "leak", "leakage": "leak", "leaks": "leak",
    "failed": "fail", "fails": "fail", "failure": "fail",
    "seizes": "seize", "seizure": "seize", "seized": "seize",
    "prevents": "prevent", "preventing": "prevent", "prevented": "prevent",
    "shut": "close",
    "damaged": "damage", "damages": "damage",
    "released": "release", "releases": "release",
    "discharged": "discharge", "discharges": "discharge",
    "injuries": "injury", "injured": "injury",
    "operators": "operator", "personnel": "person",
    "accumulates": "accumulate", "accumulation": "accumulate",
    "locations": "location", "destination": "location", "destinations": "location",
    "dispersion": "disperse", "dispersal": "disperse", "disperses": "disperse",
    "asphyxiation": "asphyxia", "asphyxiating": "asphyxia",
    "connections": "connection", "couplings": "coupling",
    "quantities": "quantity", "volumes": "volume",
    "completed": "complete", "completion": "complete",
    "promptly": "prompt",
    "verified": "verify", "verification": "verify", "confirms": "confirm",
    "confirmed": "confirm", "confirmation": "confirm",
    "criteria": "criterion",
}

TECHNICAL_MECHANISM_CLASSES = {
    "debris_obstruction": {"debris"},
    "hose_kink": {"kink"},
    "generic_obstruction": {"block", "restriction", "restrict"},
    "leakage": {"leak", "packing", "seal", "tightness"},
    "equipment_failure": {"seize", "stuck", "bind", "fail", "damage", "rupture", "break"},
    "communication": {"radio", "message", "readback", "misunderstand", "communication"},
    "operator_omission": {"omit", "forgot", "skip", "bypass"},
    "operator_timing": {"delay", "premature", "early", "late", "repeat", "prolong"},
    "pressure_mismatch": {"underrated", "rating", "allowable", "capability", "pressure"},
}


def technical_statement_signature(value: Any) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return normalized content tokens, equipment tags, and mechanism classes.

    This is deliberately deterministic and local. It is used only to suppress
    obvious paraphrases in the human-facing worksheet; atomic scenario records
    and their exact wording remain unchanged in the JSON audit trail.
    """
    text = normalize_whitespace(value).lower()
    text = re.sub(r"\boxygen[-\s]+deficien(?:cy|t)\b", "asphyxia", text)
    tokens: Set[str] = set()
    tags: Set[str] = set()
    for raw in re.findall(r"[a-z]+\d+|\d+(?:\.\d+)?|[a-z]+", text):
        token = TECHNICAL_TOKEN_NORMALIZATION.get(raw, raw)
        if re.fullmatch(r"[a-z]+\d+", token):
            tags.add(token)
        if token in TECHNICAL_DEDUPE_STOPWORDS or len(token) < 2:
            continue
        tokens.add(token)
    mechanisms = {
        label for label, vocabulary in TECHNICAL_MECHANISM_CLASSES.items()
        if tokens & vocabulary
    }
    return tokens, tags, mechanisms


def technical_statements_are_paraphrases(left: Any, right: Any, field: str = "") -> bool:
    """Identify obvious same-meaning worksheet bullets without merging hazards.

    Different equipment tags or different initiating-mechanism classes remain
    separate. The thresholds intentionally favor false negatives: uncertain
    pairs stay visible instead of being silently combined.
    """
    left_text = normalize_whitespace(left).lower().rstrip(" .")
    right_text = normalize_whitespace(right).lower().rstrip(" .")
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    left_tokens, left_tags, left_mechanisms = technical_statement_signature(left_text)
    right_tokens, right_tags, right_mechanisms = technical_statement_signature(right_text)
    if not left_tokens or not right_tokens:
        return False
    if left_tags and right_tags and left_tags.isdisjoint(right_tags):
        return False
    left_specific = left_mechanisms - {"generic_obstruction"}
    right_specific = right_mechanisms - {"generic_obstruction"}
    if left_specific and right_specific and left_specific.isdisjoint(right_specific):
        return False
    if (
        left_mechanisms and right_mechanisms
        and left_mechanisms.isdisjoint(right_mechanisms)
    ):
        return False
    common = left_tokens & right_tokens
    if len(common) < 3:
        return False
    containment = len(common) / min(len(left_tokens), len(right_tokens))
    union = left_tokens | right_tokens
    jaccard = len(common) / len(union) if union else 0.0
    if field in {"deviation", "effect", "consequence", "necessary_action", "action_required"}:
        return containment >= 0.58 and jaccard >= 0.30
    return containment >= 0.42 and jaccard >= 0.25


def semantic_dedupe_worksheet_items(
    values: List[str], field: str = "", limit: int = 30
) -> List[str]:
    """Remove exact and clear paraphrase duplicates for worksheet presentation."""
    output: List[str] = []
    for value in values:
        text = normalize_whitespace(value)
        if not text:
            continue
        duplicate_index = next(
            (
                index for index, prior in enumerate(output)
                if technical_statements_are_paraphrases(text, prior, field)
            ),
            None,
        )
        if duplicate_index is not None:
            # Prefer the more technically complete qualified statement while
            # retaining one bullet for the concept.
            new_tokens = technical_statement_signature(text)[0]
            old_tokens = technical_statement_signature(output[duplicate_index])[0]
            if len(new_tokens) > len(old_tokens):
                output[duplicate_index] = text
            continue
        output.append(text)
        if len(output) >= limit:
            break
    return output


def dedupe_group_causes(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Remove exact and near-paraphrase causes within one matrix item."""

    def cause_signature(value: Any) -> Tuple[str, Set[str]]:
        text = compact_hazop_phrase(value, "potential_cause").lower().rstrip(".")
        replacements = {
            "fails": "fail", "failed": "fail", "failure": "fail",
            "closed": "close", "closing": "close", "opened": "open",
            "opening": "open", "blocked": "block", "blockage": "block",
            "incorrect": "wrong", "incorrectly": "wrong",
        }
        tokens = []
        for token in re.findall(r"[a-z0-9]+", text):
            token = replacements.get(token, token)
            if token in {"the", "a", "an", "is", "of", "to", "for", "and", "or", "may", "can"}:
                continue
            tokens.append(token)
        return text, set(tokens)

    output: List[Dict[str, Any]] = []
    seen: List[Tuple[str, Set[str]]] = []
    for row in rows:
        cause_text = row.get("potential_cause") or row.get("cause") or ""
        key, tokens = cause_signature(cause_text)
        duplicate = False
        for prior_key, prior_tokens in seen:
            if key and key == prior_key:
                duplicate = True
                break
            union = tokens | prior_tokens
            overlap = len(tokens & prior_tokens) / len(union) if union else 0.0
            if len(tokens & prior_tokens) >= 2 and overlap >= 0.75:
                duplicate = True
                break
        if duplicate:
            continue
        if key:
            seen.append((key, tokens))
        output.append(row)
        if len(output) >= limit:
            break
    return output


def safeguard_evidence_text(documents: Dict[str, Any]) -> str:
    if not isinstance(documents, dict):
        return ""
    return normalize_whitespace(supplied_document_text(documents)).lower()


MISSING_SAFEGUARD_MARKERS = {
    "not identified",
    "not applicable",
    "none",
    "n/a",
    "na",
    "no safeguard identified",
    "no safeguards identified",
}


GENERIC_ACTION_MARKERS = {
    "assign row-specific review",
    "correct valve operating sequence",
    "define protection",
    "define row-specific action",
    "define row-specific protection",
    "identify documented safeguard",
    "identify row-specific safeguard",
    "refer to engineering team for further study",
    "refer to engineering team",
    "refer to operations team for further study",
    "refer to process engineering",
    "refer to operations/process engineering for row-specific review",
    "review procedure",
    "review procedure for clarity",
    "verify operator training",
}


def is_missing_safeguard_marker(text: Any) -> bool:
    lower = normalize_whitespace(text).lower().strip(" .;:")
    return lower in MISSING_SAFEGUARD_MARKERS


# Matches a step-number reference used as the leading subject of a sentence,
# with or without a qualifier word in front -- "Step 1 ...", "Steps 1-2 ...",
# "Procedure steps 1-2 ...", "The operating steps 1 ...", "Documented
# procedure step 1 ...". A single source of truth for this pattern because it
# is checked in three independent cleaning paths (safeguard, deviation/effect
# via strip_procedure_step_leakage, and is_weak_procedural_effect) and each
# had drifted to a narrower, inconsistent version of the same check.
STEP_LABEL_LEAK_PATTERN = re.compile(
    r"^(?:procedure\s+|operating\s+|operational\s+|documented\s+|the\s+)*steps?\s+\d+(?:\s*[-–]\s*\d+)?\b",
    re.IGNORECASE,
)


def is_procedure_step_safeguard(item: str) -> bool:
    lower = normalize_whitespace(item).lower()
    if not lower:
        return False
    # Named, documented administrative controls are safeguards.  Do not drop
    # them merely because their mechanism is procedural or administrative.
    documented_administrative_controls = (
        "radio communication protocol",
        "lead-operator procedural control",
        "lead operator procedural control",
        "pre-depressurization valve-closure verification",
        "valve-closure verification",
        "line-up verification",
        "lineup verification",
        "permit-to-work",
        "permit to work",
        "lockout/tagout",
        "lockout tagout",
        "loto",
    )
    if any(control in lower for control in documented_administrative_controls):
        return False
    if lower.startswith(("procedure step", "operating procedure", "operational procedure", "documented procedure")):
        return True
    if STEP_LABEL_LEAK_PATTERN.match(lower):
        return True
    if re.search(r"\bprocedure\s+steps?\s+\d+\b", lower):
        return True
    if re.search(r"\bsteps?\s+\d+(?:\s*[-–]\s*\d+)?\s*:", lower):
        return True
    if lower.startswith("sequence ") and any(term in lower for term in ("requires", "places", "specifies")):
        return True
    if lower.startswith("lead operator controls the documented"):
        return True
    return False


def is_unsupported_generic_safeguard(item: str, documents: Dict[str, Any]) -> bool:
    text = normalize_whitespace(item)
    lower = text.lower().rstrip(".")
    if not lower or lower in {"not identified", "not applicable", "none", "n/a"}:
        return False
    if documented_safeguard_phrase(text, documents):
        return False
    generic_admin = any(pattern in lower for pattern in (
        "requires verification",
        "dummy run",
        "practice run",
        "one-on-one supervision",
        "one on one supervision",
        "documented operating procedure",
        "documented operating procedures",
        "routine operator round",
        "routine operator rounds",
        "operator rounds",
        "operator visual inspection",
        "field observation",
        "observation of downstream",
        "valve position check",
        "valve-position check",
        "position checks",
        "routine procedure",
        "operator vigilance",
    ))
    if not generic_admin:
        return False
    evidence = safeguard_evidence_text(documents)
    if not evidence:
        return True
    required_phrases = [
        "operator rounds",
        "routine rounds",
        "valve position",
        "position check",
        "visual inspection",
        "operating procedure",
        "field observation",
    ]
    return not any(phrase in evidence for phrase in required_phrases)


def node_scope_plain_text(node: Optional[Dict[str, Any]]) -> str:
    if not isinstance(node, dict):
        return ""
    parts: List[str] = []
    for key in (
        "node_id",
        "node_name",
        "node_description",
        "boundaries",
        "design_intent",
        "operating_conditions",
    ):
        parts.append(normalize_whitespace(node.get(key)))
    for key in ("equipment", "lines", "materials", "procedure_steps", "procedure_initial_setup"):
        parts.extend(sanitize_list(node.get(key), 80))
    for safeguard in sanitize_known_safeguards(node.get("known_safeguards"), 80):
        parts.extend([safeguard.get("name", ""), safeguard.get("evidence", "")])
    return normalize_whitespace(" ".join(part for part in parts if part)).lower()


def matrix_safeguards_for_row(row: Dict[str, Any], node: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(node, dict):
        return []
    matrix_item = matrix_item_for_completed_row(row, node)
    return row_text_items(
        matrix_text_value(matrix_item, "safeguard", "safeguards", "existing_safeguards", "protection"),
        12,
    )


def safeguard_supported_for_current_node(safeguard: str, row: Dict[str, Any], node: Optional[Dict[str, Any]]) -> bool:
    """Return True only for a safeguard documented in this node or matrix item.

    Facility-wide text is deliberately insufficient. A safeguard may be used in
    a final row only when the same matrix item or the node's explicit safeguard
    inventory names it. Mechanism matching is checked separately by
    :func:`safeguard_applies_to_row`.
    """
    text = normalize_whitespace(safeguard)
    lower = text.lower()
    if not lower:
        return False
    documented = matrix_safeguards_for_row(row, node)
    if isinstance(node, dict):
        documented.extend(safeguard_names(node.get("known_safeguards"), confirmed_only=True))

    def concepts(value: str) -> Set[str]:
        value = normalize_whitespace(value).lower()
        groups = (
            ({
                "pressure relief valve", "pressure relief", "pressure-vacuum valve",
                "pressure vacuum valve", "pressure/vacuum valve", "p/v valve",
                "pv valve", "prv", "psv",
            }, "pressure_relief"),
            ({"non-return valve", "non return valve", "nrv", "check valve"}, "non_return"),
            ({"local indication", "pressure gauge", "flow meter", "level indicator", "temperature indicator"}, "local_indication"),
            ({"pressure control loop", "pressure controller", "pic"}, "pressure_control"),
            ({"flow control loop", "flow controller", "fic"}, "flow_control"),
            ({"temperature control loop", "temperature controller", "tic"}, "temperature_control"),
            ({"level control loop", "level controller", "lic"}, "level_control"),
            ({"nitrogen blanketing/inerting", "nitrogen blanketing", "inerting"}, "inerting"),
            ({"vent to safe location", "safe vent", "vent system"}, "safe_vent"),
            ({"flame arrester", "flame arrestor"}, "flame_arrester"),
            ({"lead-operator procedural control", "lead operator control", "lead operator controls the procedure"}, "lead_operator_control"),
            ({"radio communication protocol", "radio communication", "operator radio communication"}, "radio_communication"),
            ({"pre-depressurization valve-closure verification", "valve closure verification", "line-up verification"}, "lineup_verification"),
            ({"bunded enclosure", "bund", "secondary containment"}, "secondary_containment"),
            ({"audible nitrogen-breakthrough indication", "audible nitrogen breakthrough", "audible n2 breakthrough"}, "audible_breakthrough"),
            ({"ist seal pot", "seal pot", "glycol-sealed overflow to bund", "glycol sealed overflow"}, "seal_pot_overflow"),
            ({"level alarm", "high level alarm", "low level alarm"}, "level_alarm"),
            ({"high-level trip", "high level trip", "level trip"}, "level_trip"),
        )
        result = {value}
        for aliases, concept in groups:
            if any(alias in value for alias in aliases):
                result.add(concept)
        return result

    requested = concepts(text)
    return any(requested & concepts(value) for value in documented if normalize_whitespace(value))


def contextualize_safeguard_phrase(safeguard: str, row: Dict[str, Any], node: Optional[Dict[str, Any]]) -> str:
    # Formatting must not turn a generic facility control into a new,
    # scenario-specific technical claim. Preserve documented wording.
    return normalize_whitespace(safeguard)


def safeguard_validation_label(safeguard: str) -> str:
    """Map tag-rich wording to a control class solely for mechanism checks."""
    text = normalize_whitespace(safeguard)
    lower = text.lower()
    if re.search(r"\bpic\b|pressure\s+(?:indicator\s*/\s*)?controller", lower):
        return "Pressure control loop"
    if re.search(
        r"pressure\s*[/\-\s]\s*vacuum\s+valve|\bp\s*/?\s*v\s+valve\b|\bpv\s+valve\b",
        lower,
    ):
        return "Pressure relief valve"
    if re.search(r"\bfic\b|flow\s+(?:indicator\s*/\s*)?controller", lower):
        return "Flow control loop"
    if re.search(r"\btic\b|temperature\s+(?:indicator\s*/\s*)?controller", lower):
        return "Temperature control loop"
    if re.search(r"\blic\b|level\s+(?:indicator\s*/\s*)?controller", lower):
        return "Level control loop"
    return text


def documented_safeguard_candidates_for_row(
    row: Dict[str, Any], node: Optional[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Return only current-matrix and current-node safeguard evidence.

    This deliberately does not search facility-wide document text.  The
    evidence must already have been associated with the current node by the
    P&ID/process parser, or documented on the matching matrix item.
    """
    if not isinstance(node, dict):
        return []
    candidates: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def add(name: Any, evidence: Any, source: str, status: str = "documented_existing") -> None:
        text = normalize_whitespace(name)
        if not text or is_missing_safeguard_marker(text):
            return
        normalized_status = normalize_whitespace(status).lower()
        if normalized_status in {"recommended_unconfirmed", "proposed", "unconfirmed"}:
            return
        key = text.lower()
        if key in seen:
            return
        probe = dict(row)
        if not safeguard_supported_for_current_node(text, probe, node):
            return
        if not safeguard_applies_to_row(safeguard_validation_label(text), probe):
            return
        if is_procedure_step_safeguard(text):
            return
        seen.add(key)
        candidates.append({
            "name": text,
            "evidence": normalize_whitespace(evidence),
            "source": source,
            "status": normalize_whitespace(status) or "documented_existing",
        })

    matrix_item = matrix_item_for_completed_row(row, node)
    if isinstance(matrix_item, dict):
        matrix_evidence = normalize_whitespace(
            matrix_text_value(matrix_item, "safeguard_evidence", "evidence", "source_evidence")
        )
        for name in row_text_items(
            matrix_text_value(matrix_item, "safeguard", "safeguards", "existing_safeguards", "protection"),
            20,
        ):
            add(name, matrix_evidence, "current_matrix_item")

    for safeguard in sanitize_known_safeguards(node.get("known_safeguards"), 80):
        add(
            safeguard.get("name"),
            safeguard.get("evidence"),
            safeguard.get("source") or "current_node_confirmed_safeguards",
            safeguard.get("status") or "documented_existing",
        )
    return candidates


def clean_node_safeguards(rows: List[Dict[str, Any]], documents: Dict[str, Any], node: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Validate and, when needed, backfill only row-relevant documented controls."""
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        safeguards = row_text_items(item.get("safeguard") or item.get("safeguards"), 12)
        kept: List[str] = []
        saw_not_applicable = False
        saw_not_identified = False
        evidence_records: List[Dict[str, str]] = []
        candidates = documented_safeguard_candidates_for_row(item, node)
        candidate_by_name = {candidate["name"].lower(): candidate for candidate in candidates}

        def matching_candidate(value: str) -> Optional[Dict[str, str]]:
            key = normalize_whitespace(value).lower()
            exact = candidate_by_name.get(key)
            if exact:
                return exact
            requested = set(re.findall(r"[a-z0-9]+", key))
            best: Optional[Dict[str, str]] = None
            best_score = 0.0
            for candidate in candidates:
                name = normalize_whitespace(candidate.get("name")).lower()
                if not name:
                    continue
                if key in name or name in key:
                    return candidate
                candidate_tokens = set(re.findall(r"[a-z0-9]+", name))
                union = requested | candidate_tokens
                score = len(requested & candidate_tokens) / len(union) if union else 0.0
                if score > best_score:
                    best, best_score = candidate, score
            return best if best_score >= 0.35 else None
        for safeguard in safeguards:
            if is_missing_safeguard_marker(safeguard):
                marker = normalize_whitespace(safeguard).lower().rstrip(".")
                saw_not_applicable = saw_not_applicable or marker == "not applicable"
                saw_not_identified = saw_not_identified or marker in {
                    "not identified", "none", "n/a", "na",
                    "no safeguard identified", "no safeguards identified",
                }
                continue
            if is_procedure_step_safeguard(safeguard):
                continue
            if is_unsupported_generic_safeguard(safeguard, documents):
                continue
            if not safeguard_supported_for_current_node(safeguard, item, node):
                continue
            if not safeguard_applies_to_row(safeguard_validation_label(safeguard), item):
                continue
            compact = compact_hazop_phrase(contextualize_safeguard_phrase(safeguard, item, node), "safeguard")
            if compact and not is_missing_safeguard_marker(compact):
                kept.append(compact)
                candidate = matching_candidate(safeguard)
                evidence_records.append(candidate or {
                    "name": compact,
                    "evidence": "",
                    "source": "current_node_or_matrix_documented_control",
                    "status": "documented_existing",
                })
        kept = compact_hazop_items(kept, "safeguard", 4)
        if not kept:
            if saw_not_applicable or row_is_not_applicable(item):
                kept = ["Not applicable"]
            else:
                applicable = candidates[:4]
                kept = compact_hazop_items(
                    [candidate["name"] for candidate in applicable], "safeguard", 4
                )
                evidence_records = applicable
                if not kept:
                    kept = ["Not identified"]
                if not saw_not_identified and not applicable:
                    item["safeguard_validation_note"] = (
                        "Unsupported or undocumented safeguard removed during final validation."
                    )
        item["safeguard"] = "\n".join(kept)
        item["safeguards"] = item["safeguard"]
        item["safeguard_sources"] = evidence_records
        item["safeguard_evidence"] = unique_strings([
            record.get("evidence", "") for record in evidence_records
        ])
        item["safeguard_identification_status"] = (
            "documented_applicable" if evidence_records else "not_identified"
        )
        cleaned.append(item)
    return cleaned


def action_is_generic(action: Any) -> bool:
    lower = normalize_whitespace(action).lower().strip(" .;:")
    if not lower:
        return True
    if lower in GENERIC_ACTION_MARKERS:
        return True
    if lower.startswith("refer to engineering team") and "specific" not in lower:
        return True
    if lower.startswith(("assign row-specific", "define row-specific", "identify row-specific")):
        return True
    if lower.startswith("identify documented safeguard"):
        return True
    if lower in {"specific action if obvious from the evidence", "specific obvious action"}:
        return True
    return False


def row_action_keywords(row: Dict[str, Any], node: Optional[Dict[str, Any]] = None) -> set:
    text = normalize_whitespace(" ".join([
        str(row.get("guideword") or ""),
        str(row.get("parameter") or ""),
        str(row.get("deviation") or row.get("effect") or ""),
        str(row.get("potential_cause") or row.get("cause") or ""),
        str(row.get("consequence") or ""),
        str(row.get("safeguard") or row.get("safeguards") or ""),
        node_scope_plain_text(node),
    ])).lower()
    keywords = {
        word
        for word in re.findall(r"\b[a-z][a-z0-9_-]{3,}\b", text)
        if word not in {
            "this", "that", "with", "from", "into", "through", "between",
            "node", "guideword", "effect", "cause", "consequence", "safeguard",
            "action", "operator", "process", "procedure", "operation", "system",
            "equipment", "material", "control", "valve",
        }
    }
    tags = {
        tag.upper().replace(" ", "-")
        for tag in re.findall(
            r"\b(?:V|P|L|H|C|T|R|E|J|LIC|TIC|FIC|PIC|FC|LC|TC|PT|LT|FT|PV|PRV|PSV|NRV)[- ]?\d+[A-Z]?\b",
            text,
            flags=re.IGNORECASE,
        )
    }
    return keywords | {tag.lower() for tag in tags}


def action_has_row_specificity(action: Any, row: Dict[str, Any], node: Optional[Dict[str, Any]] = None) -> bool:
    lower = normalize_whitespace(action).lower()
    if not lower:
        return False
    if any(tag in lower for tag in row_action_keywords(row, node)):
        return True
    shared_concepts = {
        "air", "asphyxiation", "blockage", "communication", "contamination",
        "debris", "flow", "hose", "inert", "level", "nitrogen", "overpressure",
        "pressure", "purge", "release", "relief", "reverse", "rupture",
        "temperature", "vent",
    }
    row_text = normalize_whitespace(" ".join([
        str(row.get("deviation") or row.get("effect") or ""),
        str(row.get("potential_cause") or row.get("cause") or ""),
        str(row.get("consequence") or ""),
    ])).lower()
    return any(concept in lower and concept in row_text for concept in shared_concepts)


def action_is_generic_for_row(action: Any, row: Dict[str, Any], node: Optional[Dict[str, Any]] = None) -> bool:
    if action_is_generic(action):
        return True
    lower = normalize_whitespace(action).lower().strip(" .;:")
    weak_prefixes = (
        "review procedure",
        "verify operator training",
        "ensure operator training",
        "verify operator competency",
        "correct valve operating sequence",
        "define protection",
        "identify documented safeguard",
        "confirm safeguard",
    )
    return lower.startswith(weak_prefixes) and not action_has_row_specificity(action, row, node)


def clean_node_actions(rows: List[Dict[str, Any]], node: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        actions = row_text_items(item.get("necessary_action") or item.get("action_required"), 8)
        kept: List[str] = []
        for action in actions:
            if action_is_generic_for_row(action, item, node):
                continue
            compact = compact_hazop_phrase(action, "necessary_action")
            if compact and not action_is_generic_for_row(compact, item, node):
                kept.append(compact)
        kept = compact_hazop_items([action for action in kept if action], "necessary_action", 4)
        if not kept:
            kept = ["Not applicable" if row_is_not_applicable(item) else "Not identified"]
            item["action_validation_note"] = (
                "Missing, generic, or unsupported action was not replaced by deterministic inference."
            )
        item["necessary_action"] = "\n".join(kept)
        item["action_required"] = item["necessary_action"]
        cleaned.append(item)
    return cleaned


UNSUPPORTED_PROCEDURAL_SAFEGUARD_PATTERNS = (
    "operational procedure",
    "operating procedure",
    "documented procedure",
    "procedure compliance",
    "routine operator round",
    "routine operator rounds",
    "operator rounds",
    "operator training",
    "operator competence",
    "operator vigilance",
    "lead operator control of the procedure",
    "valve position checks",
    "position checks",
    "requires verification",
)


def strip_procedure_step_leakage(text: Any) -> str:
    item = normalize_whitespace(text)
    if not item:
        return ""
    preserve_step_phrase = bool(re.search(r"\bstep\s+\d+\b.*\bdepressur", item, flags=re.IGNORECASE))
    if not preserve_step_phrase:
        item = re.sub(r"\s*\(\s*steps?\s+\d+(?:\s*[-–]\s*\d+)?\s*\)", "", item, flags=re.IGNORECASE)
        item = re.sub(r"\b(?:in|during|from|for|under)\s+steps?\s+\d+(?:\s*[-–]\s*\d+)?\b", "", item, flags=re.IGNORECASE)
        item = re.sub(r"\b(?:in|during|from|for|under|after|before)\s+steps?\s*$", "", item, flags=re.IGNORECASE)
        item = re.sub(r"^\s*steps?\s+\d+(?:\s*[-–]\s*\d+)?\s*[:;,-]\s*", "", item, flags=re.IGNORECASE)
        item = re.sub(r"\bsteps?\s+\d+(?:\s*[-–]\s*\d+)?\s+(?:procedure|sequence)\b", "procedure sequence", item, flags=re.IGNORECASE)
        # The substitutions above only catch step labels followed by punctuation
        # or introduced by a preposition. When the step label is the grammatical
        # subject, the raw procedural label can leak into the worksheet cell.
        # Reject the item so later validation uses a real technical statement
        # instead of a step-label sentence.
        if STEP_LABEL_LEAK_PATTERN.match(item):
            return ""
    item = re.sub(r"\s{2,}", " ", item)
    return item.strip(" ,.;:")


PROCEDURAL_EFFECT_QUALIFIERS = {
    "too early",
    "too late",
    "too fast",
    "too slow",
    "more",
    "less",
    "high",
    "low",
    "zero",
    "reverse",
    "incomplete",
    "communication",
    "ineffective communication",
    "others",
    "other",
}


def is_weak_procedural_effect(effect: Any, guideword: Any = "", expected: Any = "") -> bool:
    text = normalize_whitespace(effect).lower().strip(" .;:")
    if not text:
        return True
    # A leading step label is a leaked procedure reference, not a technical
    # effect. Reject it so the row falls back to a proper deviation instead of
    # a step-label sentence. Preserve the same depressurization exception used
    # in strip_procedure_step_leakage().
    if STEP_LABEL_LEAK_PATTERN.match(text) and not re.search(r"\bstep\s+\d+\b.*\bdepressur", text):
        return True
    # A bare equipment/line tag ("V5", "L1") with no descriptive content is
    # not a technical effect statement. A real deviation needs at least a
    # subject and a state/action word.
    # Only one-word text is rejected here; the weak_values check below still
    # handles multi-word text that merely restates the guideword/qualifier.
    if len(text.split()) < 2:
        return True
    if re.search(r"\bv\d+\s+(?:is\s+)?(?:opened|closed)\s+(?:before|after)\s+v\d+\b", text):
        return True
    if re.search(r"\b(?:operator\s+)?(?:opens?|closes?)\s+v\d+\s+(?:before|after|too\s+early|too\s+late)\b", text):
        return True
    if re.search(r"\bv\d+\s+(?:opened|closed)\s+too\s+(?:early|late|quickly|slowly)\b", text):
        return True
    expected_text = normalize_whitespace(expected).lower().strip(" .;:")
    guideword_text = normalize_whitespace(guideword).lower().strip(" .;:")
    base_guideword = guideword_base_label(guideword_text).lower().strip(" .;:")
    text_without_parentheses = normalize_whitespace(re.sub(r"[()]", " ", text)).lower().strip(" .;:")
    qualifier_text = expected_text or guideword_text
    weak_values = {
        expected_text,
        guideword_text,
        base_guideword,
        f"{base_guideword} {expected_text}".strip(),
        f"{base_guideword} {base_guideword}".strip(),
        "out of sequence sequence",
        "out of sequence too early",
        "out of sequence too late",
        "rate too fast",
        "rate too slow",
        "magnitude more",
        "magnitude less",
        "pressure more",
        "high pressure",
        "low flow",
        "high flow",
        "zero flow",
        "reverse flow",
    } | PROCEDURAL_EFFECT_QUALIFIERS
    if text in {value for value in weak_values if value} or text_without_parentheses in {value for value in weak_values if value}:
        return True
    if base_guideword and re.fullmatch(rf"{re.escape(base_guideword)}(?:\s+(?:too\s+early|too\s+late|too\s+fast|too\s+slow|more|less|high|low|zero|reverse|incomplete|other|others|sequence))*", text):
        return True
    if text.startswith(f"{expected_text} or ") or text.endswith(f" or {expected_text}"):
        return True
    if qualifier_text and re.fullmatch(rf"(?:{re.escape(qualifier_text)}|or|\s)+", text):
        return True
    return False


def clean_manual_worksheet_field(value: Any, field: str) -> str:
    cleaned: List[str] = []
    # Deviation/Effect is the row's single defining abnormal event, never
    # multiple bullets (unlike cause/consequence/safeguard/action, which can
    # legitimately have several). Stop at the first credible candidate.
    is_single_valued = field in {"deviation", "effect"}
    for raw_item in row_text_items(value, 20):
        item = strip_procedure_step_leakage(raw_item)
        if not item:
            continue
        if is_single_valued:
            if not is_incomplete_hazop_fragment(item):
                cleaned.append(item)
                break
            continue
        lower = item.lower().rstrip(".")
        if field in {"safeguard", "safeguards"}:
            if any(pattern in lower for pattern in UNSUPPORTED_PROCEDURAL_SAFEGUARD_PATTERNS):
                continue
        if field in {"potential_cause", "cause"}:
            item = re.sub(r"\boperator error\s+in\s+following\s+the\s+procedure\b", "Human factors", item, flags=re.IGNORECASE)
        cleaned.append(item)
    limit = 1 if is_single_valued else 8
    return "\n".join(compact_hazop_items(cleaned, field, limit))


def clean_procedural_effect(value: Any, guideword: Any = "", expected: Any = "") -> str:
    items = row_text_items(value, 12)
    cleaned: List[str] = []
    # Deviation/Effect is single-valued -- stop at the first credible
    # candidate instead of collecting every "X or Y" alternative as its own
    # bullet.
    for item in items:
        item = strip_procedure_step_leakage(item)
        if not item:
            continue
        item = normalize_whitespace(item)
        if is_weak_procedural_effect(item, guideword, expected):
            continue
        if is_incomplete_hazop_fragment(item):
            continue
        cleaned.append(item)
        if cleaned:
            break
    return "\n".join(compact_hazop_items(cleaned, "deviation", 1))


def clean_manual_worksheet_row(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    for field in ("deviation", "effect", "potential_cause", "cause", "consequence", "necessary_action", "action_required"):
        if item.get(field):
            item[field] = clean_manual_worksheet_field(item.get(field), field)
    safeguard = clean_manual_worksheet_field(item.get("safeguard") or item.get("safeguards"), "safeguard")
    item["safeguard"] = safeguard or ("Not applicable" if row_is_not_applicable(item) else "Not identified")
    item["safeguards"] = item["safeguard"]
    return item


COMMUNICATION_CAUSE_PATTERNS = (
    "miscommunication",
    "communication failure",
    "ineffective communication",
    "radio failure",
    "radio communication",
    "misunderstanding between operators",
    "operator-message misunderstanding",
    "message misunderstanding",
)


def has_communication_matrix_item(node: Dict[str, Any]) -> bool:
    return any(
        normalize_whitespace(item.get("guideword")).lower() == "communication"
        for item in guideword_parameter_matrix(node)
    )


def is_communication_row(row: Dict[str, Any]) -> bool:
    return normalized_guideword(row.get("guideword")).lower() == "communication"


def remove_patterned_items(value: Any, patterns: tuple[str, ...]) -> Any:
    items = row_text_items(value, 20)
    kept = [
        item for item in items
        if not any(pattern in item.lower() for pattern in patterns)
    ]
    if isinstance(value, list):
        return kept
    return "\n".join(kept)


def row_is_invalid_meta_row(row: Dict[str, Any]) -> bool:
    text = normalize_whitespace(
        " ".join([
            str(row.get("guideword", "")),
            str(row.get("deviation", "")),
            str(row.get("effect", "")),
            str(row.get("parameter", "")),
        ])
    ).lower()
    return any(pattern in text for pattern in (
        "all relevant guideword",
        "all relevant procedural",
        "multiple guideword",
        "all guideword",
        "various deviation",
        "various deviations",
    ))


def remove_cross_guideword_communication_items(
    rows: List[Dict[str, Any]],
    node: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not has_communication_matrix_item(node):
        return rows
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if is_communication_row(item):
            cleaned.append(item)
            continue
        for field in ("potential_cause", "cause", "assumptions", "source_references"):
            item[field] = remove_patterned_items(item.get(field), COMMUNICATION_CAUSE_PATTERNS)
        if not normalize_whitespace(item.get("potential_cause") or item.get("cause")):
            item["potential_cause"] = "Not identified"
            item["cause"] = "Not identified"
        cleaned.append(item)
    return cleaned


def procedural_row_category(row: Dict[str, Any]) -> str:
    text = normalize_whitespace(" ".join([
        str(row.get("guideword") or ""),
        str(row.get("parameter") or ""),
    ])).lower()
    if "communication" in text:
        return "communication"
    if "reverse" in text:
        return "reverse"
    if "pressure" in text:
        return "pressure"
    if "rate" in text or "too fast" in text or "too slow" in text:
        return "rate"
    if "magnitude" in text or re.search(r"\b(?:more|less)\b", text):
        return "magnitude"
    if "incomplete" in text or "completion" in text:
        return "incomplete"
    if "other" in text or "others" in text:
        return "other"
    if "sequence" in text or "too early" in text or "too late" in text:
        return "sequence"
    return ""


def cross_guideword_signal_categories(text: Any) -> Set[str]:
    lower = normalize_whitespace(text).lower()
    categories: Set[str] = set()
    if not lower:
        return categories
    if re.search(r"\b(?:communication|radio|instruction misunderstood|message misunderstanding|misunderstanding)\b", lower):
        categories.add("communication")
    if re.search(r"\b(?:reverse flow|backflow|back flow|non-return|nrv|upstream process)\b", lower):
        categories.add("reverse")
    if re.search(r"\b(?:overpressure|high pressure|pressure surge|pressure relief|prv|psv|rupture due to pressure|pressure protection)\b", lower):
        categories.add("pressure")
    if re.search(r"\b(?:high flow|low flow|flow rate|too fast|too slow|opened too far|opened too quickly|flow restriction|velocity)\b", lower):
        categories.add("rate")
    if re.search(r"\b(?:more n2|less n2|excessive n2|excessive duration|left open longer|closed before sufficient|too much|too little)\b", lower):
        categories.add("magnitude")
    if re.search(r"\b(?:incomplete execution|not fully opened|not fully closed|full open|full close|full travel)\b", lower):
        categories.add("incomplete")
    if re.search(r"\b(?:wrong utility|wrong material|unavailable supply|supply unavailable|wrong n2|wrong nitrogen)\b", lower):
        categories.add("other")
    if re.search(r"\b(?:out of sequence|wrong sequence|sequence verification|before .*confirmed|before .*ready|after .*required)\b", lower):
        categories.add("sequence")
    return categories


def has_cross_guideword_content(text: Any, row: Dict[str, Any]) -> bool:
    category = procedural_row_category(row)
    if not category:
        return False
    signals = cross_guideword_signal_categories(text)
    if not signals:
        return False
    allowed = {category}
    # Out-of-sequence rows often name the valve state that creates no-flow or
    # residual-air effects; do not treat that as rate/magnitude contamination.
    if category == "sequence":
        allowed.add("incomplete")
    # Less magnitude, incomplete, and sequence-too-late can share residual-air
    # language. The strong incomplete signals above are limited to execution
    # wording, so no extra allowance is needed here.
    return bool(signals - allowed)


def clean_cross_guideword_content(rows: List[Dict[str, Any]], node: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not node.get("manual_source"):
        return rows
    cleaned_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        category = procedural_row_category(item)
        if not category:
            cleaned_rows.append(item)
            continue
        expected_deviation = procedural_sequence_deviation(
            item.get("guideword"),
            item.get("parameter"),
            item.get("expected_deviation") or item.get("deviation_hint"),
            node,
        )
        if expected_deviation and has_cross_guideword_content(item.get("deviation") or item.get("effect"), item):
            item["deviation"] = expected_deviation
            item["effect"] = expected_deviation
            item["expected_deviation"] = expected_deviation
        # Guideword discipline applies to the deviation being examined, not to
        # every downstream field. A sequence deviation may legitimately cause
        # high pressure, be mitigated by a pressure-relief device, and require
        # a pressure action. Removing those statements as "cross-guideword"
        # content destroys the HAZOP cause-consequence-safeguard chain.
        # Round 2/3 already adjudicate relevance of causes and field content;
        # this deterministic pass therefore corrects only the deviation label.
        recovery = procedural_sequence_recovery_cells(item.get("guideword"), node)
        if field_is_missing(item.get("potential_cause") or item.get("cause")) and recovery.get("potential_cause"):
            item["potential_cause"] = recovery["potential_cause"]
            item["cause"] = recovery["potential_cause"]
        if field_is_missing(item.get("consequence")) and recovery.get("consequence"):
            item["consequence"] = recovery["consequence"]
        if field_is_missing(item.get("necessary_action") or item.get("action_required")) and recovery.get("necessary_action"):
            item["necessary_action"] = recovery["necessary_action"]
            item["action_required"] = recovery["necessary_action"]
        cleaned_rows.append(item)
    return cleaned_rows


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_whitespace(value)
        if text:
            return text
    return ""


def merge_rows_by_guideword(rows: List[Dict[str, Any]], node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the atomic, auditable worksheet representation.

    Each distinct credible cause remains an internal sub-row so its matched
    consequence, safeguard, action, review decisions, and lineage are not
    lost. The final human-facing worksheet is consolidated separately into
    one row per node/matrix item by ``consolidate_worksheet_rows``.
    """
    parameters = node_parameters(node)
    matrix = guideword_parameter_matrix(node)
    strict_manual_matrix = bool(node.get("manual_source") and matrix)

    def matrix_item_for_row(row: Dict[str, Any], guideword: str, parameter: str) -> Optional[Dict[str, Any]]:
        explicit = normalize_whitespace(row.get("matrix_key")).lower()
        if explicit:
            for item in matrix:
                if normalize_whitespace(item.get("matrix_key")).lower() == explicit:
                    return item
        row_text = normalize_whitespace(
            " ".join([
                str(row.get("guideword", "")),
                str(row.get("deviation", "")),
                str(row.get("effect", "")),
                str(row.get("expected_deviation", "")),
                str(row.get("deviation_hint", "")),
            ])
        ).lower()
        exact_candidates = [
            item for item in matrix
            if (
                item.get("guideword", "").lower() == guideword.lower()
                or guideword_base_label(item.get("guideword")).lower() == guideword_base_label(guideword).lower()
            )
            and item.get("parameter", "").lower() == parameter.lower()
        ]
        for item in exact_candidates:
            hint = normalize_whitespace(
                item.get("chair_deviation")
                or item.get("deviation_hint")
                or item.get("expected_deviation")
                or item.get("effect")
            ).lower()
            if hint and hint in row_text:
                return item
        if len(exact_candidates) == 1:
            return exact_candidates[0]
        hinted_candidates = []
        for item in matrix:
            hint = normalize_whitespace(
                item.get("chair_deviation")
                or item.get("deviation_hint")
                or item.get("expected_deviation")
                or item.get("effect")
            ).lower()
            item_guideword = normalize_whitespace(item.get("guideword")).lower()
            if not hint:
                continue
            guideword_matches_hint = (
                guideword.lower() == hint
                or guideword.lower().endswith(f" {hint}")
                or hint in guideword.lower()
            )
            item_guideword_base = guideword_base_label(item_guideword).lower()
            guideword_base = guideword_base_label(guideword).lower()
            guideword_contains_category = item_guideword_base and (
                item_guideword_base == guideword_base or item_guideword_base in guideword.lower()
            )
            text_contains_hint = hint in row_text
            text_contains_category = (
                (item_guideword and item_guideword in row_text)
                or (item_guideword_base and item_guideword_base in row_text)
            )
            if guideword_matches_hint or (text_contains_hint and (text_contains_category or guideword_contains_category)):
                hinted_candidates.append(item)
        if len(hinted_candidates) == 1:
            return hinted_candidates[0]
        same_parameter_hints = [
            item for item in hinted_candidates
            if item.get("parameter", "").lower() == parameter.lower()
        ]
        if len(same_parameter_hints) == 1:
            return same_parameter_hints[0]
        if strict_manual_matrix:
            return None
        return None

    def row_matrix_key(row: Dict[str, Any], guideword: str, parameter: str) -> str:
        matched_item = matrix_item_for_row(row, guideword, parameter)
        if matched_item:
            return normalize_whitespace(matched_item.get("matrix_key")).lower()
        explicit = normalize_whitespace(row.get("matrix_key"))
        if explicit:
            explicit_key = explicit.lower()
            if not matrix or any(normalize_whitespace(item.get("matrix_key")).lower() == explicit_key for item in matrix):
                return explicit_key
        deviation = normalize_whitespace(
            row.get("deviation")
            or row.get("effect")
            or row.get("chair_deviation")
            or row.get("expected_deviation")
        ).lower()
        matches = [
            item for item in matrix
            if (
                item.get("guideword", "").lower() == guideword.lower()
                or guideword_base_label(item.get("guideword")).lower() == guideword_base_label(guideword).lower()
            )
            and item.get("parameter", "").lower() == parameter.lower()
        ]
        for item in matches:
            hint = normalize_whitespace(
                item.get("chair_deviation")
                or item.get("deviation_hint")
                or item.get("expected_deviation")
                or item.get("effect")
            ).lower()
            if hint and re.search(rf"\b{re.escape(hint)}\b", deviation):
                return normalize_whitespace(item.get("matrix_key")).lower()
        if len(matches) == 1:
            return normalize_whitespace(matches[0].get("matrix_key")).lower()
        return f"{parameter}:{guideword}:{deviation or guideword}".lower()

    order = {
        normalize_whitespace(item.get("matrix_key")).lower(): index
        for index, item in enumerate(matrix)
    }
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row_is_invalid_meta_row(row):
            continue
        guideword = normalized_guideword(row.get("guideword"))
        parameter = row_parameter(row, parameters)
        if not guideword or not parameter:
            continue
        matched_matrix_item = matrix_item_for_row(row, guideword, parameter)
        if strict_manual_matrix and not matched_matrix_item:
            continue
        if matched_matrix_item:
            guideword = matched_matrix_item["guideword"]
            parameter = matched_matrix_item["parameter"]
            expected_deviation = normalize_whitespace(
                matched_matrix_item.get("chair_deviation")
                or matched_matrix_item.get("expected_deviation")
                or matched_matrix_item.get("effect")
                or matched_matrix_item.get("deviation_hint")
            )
        else:
            expected_deviation = normalize_whitespace(
                row.get("chair_deviation")
                or row.get("expected_deviation")
                or row.get("effect")
                or row.get("deviation_hint")
            )
        candidate = dict(row)
        candidate["assumptions"] = sanitize_assumptions(candidate.get("assumptions"), 20)
        candidate["source_references"] = sanitize_source_references(candidate.get("source_references"), 20)
        if node.get("manual_source"):
            candidate["assumptions"] = []
            candidate = clean_manual_worksheet_row(candidate)
        candidate["guideword"] = guideword
        candidate["parameter"] = parameter
        key = row_matrix_key(row, guideword, parameter)
        candidate["matrix_key"] = key
        if matched_matrix_item:
            key = normalize_whitespace(matched_matrix_item.get("matrix_key")).lower()
            candidate["matrix_key"] = key
            candidate["expected_deviation"] = expected_deviation
        actual_deviation = normalize_whitespace(candidate.get("deviation") or candidate.get("effect"))
        if node.get("manual_source"):
            cleaned_deviation = clean_procedural_effect(
                candidate.get("deviation") or candidate.get("effect"),
                guideword,
                expected_deviation,
            )
            expected_is_usable = bool(
                expected_deviation
                and expected_deviation.lower().rstrip(".") != "not applicable"
                and not weak_deviation_text(expected_deviation)
                and not is_weak_procedural_effect(expected_deviation, guideword, "")
            )
            weak_deviation = (
                not cleaned_deviation
                or is_weak_procedural_effect(actual_deviation, guideword, expected_deviation)
                or actual_deviation.lower() in {
                    f"{guideword.lower()} {parameter.lower()}",
                    f"{guideword_base_label(guideword).lower()} {parameter.lower()}",
                }
            )
            if row_is_not_applicable(candidate):
                candidate["deviation"] = (
                    cleaned_deviation
                    if cleaned_deviation and not weak_deviation
                    else (expected_deviation if expected_is_usable else "Not applicable")
                )
            elif cleaned_deviation and not weak_deviation:
                candidate["deviation"] = cleaned_deviation
            elif expected_is_usable:
                candidate["deviation"] = expected_deviation
            else:
                candidate["deviation"] = "Deviation not identified"
            if is_weak_procedural_effect(candidate.get("deviation"), guideword, expected_deviation):
                candidate["deviation"] = (
                    expected_deviation
                    if expected_is_usable
                    else ("Not applicable" if row_is_not_applicable(candidate) else "Deviation not identified")
                )
        else:
            candidate["deviation"] = expected_deviation or actual_deviation or f"{guideword} {parameter}"
        group = grouped.setdefault(key, [])
        if not group:
            group.append(candidate)
            continue
        incoming_not_applicable = row_is_not_applicable(candidate)
        if incoming_not_applicable:
            # A not-applicable candidate never adds a sub-row: skip it once the
            # group already holds a credible cause, and skip duplicate
            # not-applicable placeholders too (the first one already covers it).
            continue
        group_has_credible = any(not row_is_not_applicable(existing_row) for existing_row in group)
        if not group_has_credible:
            # First credible cause for this matrix item: it replaces whatever
            # not-applicable placeholder(s) were seen so far, same as before.
            group[:] = [candidate]
            continue
        # A second (or further) distinct credible cause for the same
        # node+guideword: keep it as its own sub-row instead of flattening its
        # cause/consequence/safeguard/action into the first row's cells.
        group.append(candidate)

    groups: List[List[Dict[str, Any]]] = list(grouped.values())
    if matrix:
        by_key: Dict[str, List[Dict[str, Any]]] = {
            normalize_whitespace(group[0].get("matrix_key")).lower(): group
            for group in groups
            if group and normalize_whitespace(group[0].get("matrix_key"))
        }
        ordered_groups = [
            by_key[normalize_whitespace(item.get("matrix_key")).lower()]
            for item in matrix
            if normalize_whitespace(item.get("matrix_key")).lower() in by_key
        ]
        if not strict_manual_matrix:
            ordered_keys = {
                normalize_whitespace(group[0].get("matrix_key")).lower()
                for group in ordered_groups if group
            }
            ordered_groups.extend(
                group for group in groups
                if group and normalize_whitespace(group[0].get("matrix_key")).lower() not in ordered_keys
            )
        groups = ordered_groups
    else:
        groups.sort(
            key=lambda group: order.get(
                normalize_whitespace(group[0].get("matrix_key")).lower() if group else "", 999
            )
        )

    merged: List[Dict[str, Any]] = []
    for group_index, group in enumerate(groups, 1):
        deduped_group = dedupe_group_causes(group, MAX_CAUSES_PER_MATRIX_ITEM)
        group_row_id = f"{node['node_id']}-R{group_index:03d}"
        group_size = len(deduped_group)
        for cause_index, row in enumerate(deduped_group, 1):
            cause_id = f"{group_row_id}-C{cause_index:02d}"
            row["parent_row_id"] = group_row_id
            row["row_id"] = cause_id
            row["cause_id"] = cause_id
            row["cause_index"] = cause_index
            row["group_size"] = group_size
            row["node_id"] = node["node_id"]
            row["node_name"] = node["node_name"]
            row["node_description"] = row.get("node_description") or node.get("node_description", "")
            row["design_intent"] = row.get("design_intent") or node.get("design_intent", "")
            row["assumptions"] = [] if node.get("manual_source") else sanitize_assumptions(row.get("assumptions"), 20)
            row["source_references"] = sanitize_source_references(row.get("source_references"), 20)
            row.update(complete_worksheet_row(row, node))
            merged.append(row)
    return merged


def ensure_guideword_coverage(rows: List[Dict[str, Any]], node: Dict[str, Any], start_index: int = 1) -> List[Dict[str, Any]]:
    """
    Enforce HAZOP coverage independent of model compliance.
    Every node parameter must have all required guidewords represented.
    Missing combinations are recorded as Not applicable rows so the worksheet
    documents that the guideword was considered.
    """
    rows = merge_rows_by_guideword(rows, node)
    parameters = node_parameters(node)
    required_matrix = guideword_parameter_matrix(node)
    existing = set()
    for row in rows:
        guideword = normalized_guideword(row.get("guideword"))
        parameter = row_parameter(row, parameters)
        matrix_key = normalize_whitespace(row.get("matrix_key")).lower()
        if matrix_key:
            existing.add(matrix_key)
        elif guideword and parameter:
            existing.add(f"{parameter}:{guideword}".lower())

    covered_rows = list(rows)
    next_index = start_index + len(covered_rows)
    for matrix_item in required_matrix:
        parameter = matrix_item["parameter"]
        guideword = matrix_item["guideword"]
        matrix_key = normalize_whitespace(matrix_item.get("matrix_key")).lower()
        key = matrix_key or f"{parameter}:{guideword}".lower()
        if key in existing:
            continue
        deviation_hint = normalize_whitespace(
            matrix_item.get("chair_deviation")
            or matrix_item.get("deviation_hint")
            or matrix_item.get("expected_deviation")
            or matrix_item.get("effect")
        )
        has_usable_deviation_hint = bool(
            deviation_hint
            and deviation_hint.lower().rstrip(".") != "not applicable"
            and not weak_deviation_text(deviation_hint)
            and not is_weak_procedural_effect(deviation_hint, guideword, "")
        )
        # Matrix coverage is an audit statement, not a scenario generator. If
        # no credible row survived Chair finalization, record an explicit
        # Not-applicable placeholder and do not infer a cause chain here.
        coverage_deviation = deviation_hint if has_usable_deviation_hint else "Not applicable"
        covered_rows.append({
                "row_id": f"{node['node_id']}-C{next_index:03d}",
                "node_id": node["node_id"],
                "node_name": node["node_name"],
                "node_description": node.get("node_description", ""),
                "design_intent": node.get("design_intent", ""),
                "guideword": guideword,
                "parameter": parameter,
                "matrix_key": key,
                "expected_deviation": deviation_hint,
                "deviation": coverage_deviation,
                "potential_cause": "Not applicable",
                "cause": "Not applicable",
                "consequence": "Not applicable",
                "safeguard": "Not applicable",
                "safeguards": "Not applicable",
                "necessary_action": "Not applicable",
                "assumptions": [],
                "source_references": [],
                "action_required": "Not applicable",
                "source_round": "Coverage check",
                "row_origin": "auto_coverage",
                "llm_generated_row": False,
                "automatically_added_row": True,
                "fallback_generated_row": False,
        })
        existing.add(key)
        next_index += 1
    return merge_rows_by_guideword(covered_rows, node)


def scenario_to_row(
    scenario: Dict[str, Any],
    node: Dict[str, Any],
    index: int,
    source_agent: str = "",
) -> Dict[str, Any]:
    potential_cause = scenario.get("potential_cause")
    if not potential_cause:
        potential_cause = scenario.get("cause")
    if not potential_cause:
        potential_cause = scenario.get("potential_causes") or scenario.get("causes")
    safeguard = scenario.get("safeguard")
    if not safeguard:
        safeguard = scenario.get("safeguards") or scenario.get("existing_safeguards")
    consequence = scenario.get("consequence") or scenario.get("consequences")
    action = (
        scenario.get("necessary_action")
        or scenario.get("action_required")
        or scenario.get("action_candidate")
        or scenario.get("action_candidates")
    )
    assumptions = sanitize_assumptions(scenario.get("assumptions") or scenario.get("basis"), 12)
    expected_deviation = normalize_whitespace(
        scenario.get("chair_deviation")
        or scenario.get("expected_deviation")
        or scenario.get("effect")
        or scenario.get("deviation_hint")
    )
    matrix_key = normalize_whitespace(scenario.get("matrix_key")).lower()
    if (not expected_deviation or weak_deviation_text(expected_deviation)) and matrix_key:
        for item in guideword_parameter_matrix(node):
            if normalize_whitespace(item.get("matrix_key")).lower() == matrix_key:
                expected_deviation = normalize_whitespace(
                    item.get("chair_deviation")
                    or item.get("expected_deviation")
                    or item.get("effect")
                    or item.get("deviation_hint")
                )
                break
    scenario_deviation = normalize_whitespace(scenario.get("deviation") or scenario.get("effect"))
    if weak_deviation_text(scenario_deviation) and expected_deviation and not weak_deviation_text(expected_deviation):
        scenario_deviation = expected_deviation
    return {
        "row_id": normalize_whitespace(scenario.get("row_id")) or f"{node['node_id']}-F{index:03d}",
        "node_id": node["node_id"],
        "node_name": node["node_name"],
        "node_description": node.get("node_description", ""),
        "design_intent": node.get("design_intent", ""),
        "guideword": scenario.get("guideword", ""),
        "parameter": scenario.get("parameter") or row_parameter(scenario, node_parameters(node)),
        "matrix_key": normalize_whitespace(scenario.get("matrix_key")),
        "expected_deviation": expected_deviation,
        "deviation": scenario_deviation or expected_deviation or "Deviation not identified",
        "potential_cause": potential_cause or "Not identified",
        "cause": potential_cause or "Not identified",
        "consequence": consequence or "Not identified",
        "safeguard": safeguard or "Not identified",
        "safeguards": safeguard or "Not identified",
        "assumptions": assumptions,
        "source_references": sanitize_source_references(scenario.get("source_references"), 12),
        "source_agent": normalize_whitespace(source_agent or scenario.get("source_agent")),
        "source_agents": sanitize_list(scenario.get("source_agents"), 20),
        "source_row_id": normalize_whitespace(scenario.get("source_row_id")),
        "review_id": normalize_whitespace(scenario.get("review_id")),
        "proposal_id": normalize_whitespace(scenario.get("proposal_id")),
        "evidence_basis": sanitize_list(scenario.get("evidence_basis") or scenario.get("basis"), 20),
        "safeguard_evidence": sanitize_list(scenario.get("safeguard_evidence"), 20),
        "necessary_action": action or "",
        "action_required": action or "",
        "source_round": normalize_whitespace(scenario.get("source_round")) or "Expert scenario fallback",
    }


def row_is_not_applicable(row: Dict[str, Any]) -> bool:
    return (
        normalize_whitespace(row.get("potential_cause") or row.get("cause")).lower().rstrip(".") == "not applicable"
        and normalize_whitespace(row.get("consequence")).lower().rstrip(".") == "not applicable"
    )


def field_is_missing(value: Any) -> bool:
    if isinstance(value, list):
        text = normalize_whitespace(" ".join(str(item) for item in value if item))
    elif isinstance(value, dict):
        text = normalize_whitespace(" ".join(str(item) for item in value.values() if item))
    else:
        text = normalize_whitespace(value)
    text = text.lower().rstrip(".:")
    return text in {
        "",
        "not stated",
        "not identified",
        "not provided",
        "unknown",
        "effect not stated",
        "effect not identified",
        "deviation not stated",
        "deviation not identified",
    }


def final_cell_is_missing(value: Any) -> bool:
    text = normalize_whitespace(value).lower().rstrip(".:")
    return field_is_missing(value) or text in {"not applicable", "n/a", "na", "none"}


def default_row_deviation(row: Dict[str, Any], node: Dict[str, Any]) -> str:
    guideword = normalized_guideword(row.get("guideword")) or normalize_whitespace(row.get("guideword"))
    parameter = row_parameter(row, node_parameters(node))
    expected = normalize_whitespace(
        row.get("chair_deviation")
        or row.get("expected_deviation")
        or row.get("effect")
        or row.get("deviation_hint")
    )
    if not expected and node.get("manual_source"):
        matrix_key = normalize_whitespace(row.get("matrix_key")).lower()
        matches: List[Dict[str, Any]] = []
        for item in node.get("guideword_matrix") or []:
            if not isinstance(item, dict):
                continue
            item_key = normalize_whitespace(item.get("matrix_key")).lower()
            item_guideword = normalize_whitespace(item.get("guideword"))
            item_parameter = normalize_whitespace(item.get("parameter"))
            same_key = bool(matrix_key and item_key == matrix_key)
            same_guideword = (
                item_guideword.lower() == guideword.lower()
                or guideword_base_label(item_guideword).lower() == guideword_base_label(guideword).lower()
            )
            same_parameter = not parameter or not item_parameter or item_parameter.lower() == parameter.lower()
            if same_key or (same_guideword and same_parameter):
                matches.append(item)
        if len(matches) == 1:
            expected = normalize_whitespace(
                matches[0].get("chair_deviation")
                or matches[0].get("expected_deviation")
                or matches[0].get("effect")
                or matches[0].get("deviation_hint")
            )
    # Manual-study matrices can contain qualifier text rather than the actual
    # row effect, so reuse the procedural weakness check before using it.
    if expected and not is_incomplete_hazop_fragment(expected) and not is_weak_procedural_effect(expected, guideword, ""):
        return expected
    if node.get("manual_source"):
        sequence_deviation = procedural_sequence_deviation(guideword, parameter, expected, node)
        if sequence_deviation:
            return sequence_deviation
        if row_is_not_applicable(row):
            return "Not applicable"
        # In manual/procedural HAZOP, the guideword is the prompt used to
        # search for deviations; it is not itself the deviation/effect.
        procedural_parameters = {
            "procedure",
            "sequence",
            "rate",
            "magnitude",
            "completion",
            "communication",
            "reverse",
            "other",
            "others",
        }
        if guideword and parameter and parameter.lower() not in procedural_parameters:
            return f"{guideword_base_label(guideword) or guideword} {parameter}".strip()
        return "Deviation not identified"
    if guideword and parameter:
        return f"{guideword} {parameter}".strip()
    return guideword or parameter or "Deviation not identified"


def matrix_item_for_completed_row(row: Dict[str, Any], node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    matrix_key = normalize_whitespace(row.get("matrix_key")).lower()
    guideword = normalized_guideword(row.get("guideword")) or normalize_whitespace(row.get("guideword"))
    parameter = row_parameter(row, node_parameters(node))
    guideword_base = guideword_base_label(guideword).lower()
    row_text = normalize_whitespace(
        " ".join([
            str(row.get("guideword", "")),
            str(row.get("deviation", "")),
            str(row.get("effect", "")),
            str(row.get("expected_deviation", "")),
            str(row.get("deviation_hint", "")),
        ])
    ).lower()
    paren_hint_match = re.search(r"\(([^)]+)\)", guideword)
    guideword_hint = normalize_whitespace(paren_hint_match.group(1) if paren_hint_match else "").lower()
    fallback_match: Optional[Dict[str, Any]] = None
    for item in guideword_parameter_matrix(node):
        item_key = normalize_whitespace(item.get("matrix_key")).lower()
        item_guideword = normalize_whitespace(item.get("guideword"))
        item_parameter = normalize_whitespace(item.get("parameter"))
        if matrix_key and item_key == matrix_key:
            return item
        if (
            item_parameter.lower() == parameter.lower()
            and (
                item_guideword.lower() == guideword.lower()
                or guideword_base_label(item_guideword).lower() == guideword_base
            )
        ):
            item_hint = normalize_whitespace(
                item.get("chair_deviation")
                or item.get("deviation_hint")
                or item.get("expected_deviation")
                or item.get("effect")
            ).lower()
            if item_hint and (item_hint in row_text or item_hint == guideword_hint):
                return item
            if fallback_match is None:
                fallback_match = item
    return fallback_match


def row_has_credible_effect_and_cause(row: Dict[str, Any]) -> bool:
    return (
        not row_is_not_applicable(row)
        and not field_is_missing(row.get("deviation") or row.get("effect"))
        and not field_is_missing(row.get("potential_cause") or row.get("cause"))
    )


def matrix_text_value(item: Optional[Dict[str, Any]], *keys: str) -> str:
    if not isinstance(item, dict):
        return ""
    for key in keys:
        value = item.get(key)
        if isinstance(value, list):
            text = "\n".join(row_text_items(value, 8))
        else:
            text = normalize_whitespace(value)
        if text and not field_is_missing(text):
            return text
    return ""


def default_row_safeguard(row: Dict[str, Any], node: Dict[str, Any]) -> str:
    if row_is_not_applicable(row):
        return "Not applicable"
    return "Not identified"


def complete_worksheet_row(row: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final deterministic worksheet repair.
    This does not invent scenarios; it only prevents blank/generic cells and
    keeps one guideword row readable when the model omits a field.
    """
    row = dict(row)
    guideword = normalized_guideword(row.get("guideword")) or normalize_whitespace(row.get("guideword"))
    row["guideword"] = guideword
    row["parameter"] = row_parameter(row, node_parameters(node))
    expected = normalize_whitespace(
        row.get("chair_deviation")
        or row.get("expected_deviation")
        or row.get("deviation_hint")
        or row.get("effect")
    )
    if expected:
        row["expected_deviation"] = expected

    deviation = normalize_whitespace(row.get("deviation") or row.get("effect"))
    weak_deviation = (
        field_is_missing(deviation)
        or is_weak_procedural_effect(deviation, guideword, expected)
        or deviation.lower() in {
            f"{guideword.lower()} {row.get('parameter', '').lower()}",
            f"{guideword_base_label(guideword).lower()} {row.get('parameter', '').lower()}",
        }
    )
    if weak_deviation:
        row["deviation"] = default_row_deviation(row, node)
    elif node.get("manual_source"):
        cleaned_deviation = clean_procedural_effect(deviation, guideword, expected)
        if cleaned_deviation:
            row["deviation"] = cleaned_deviation
    if node.get("manual_source") and is_weak_procedural_effect(row.get("deviation"), guideword, expected):
        row["deviation"] = default_row_deviation(row, node)

    if node.get("manual_source") and field_is_missing(row.get("deviation")):
        row["deviation"] = "Not applicable"
        for key in ("potential_cause", "cause", "consequence", "safeguard", "safeguards", "necessary_action", "action_required"):
            row[key] = "Not applicable"

    not_applicable = row_is_not_applicable(row)
    if not_applicable:
        row["deviation"] = "Not applicable" if field_is_missing(row.get("deviation")) else row.get("deviation")
        for key in ("potential_cause", "cause", "consequence", "safeguard", "safeguards", "necessary_action", "action_required"):
            if field_is_missing(row.get(key)):
                row[key] = "Not applicable"
        return row

    if final_cell_is_missing(row.get("potential_cause") or row.get("cause")):
        row["potential_cause"] = "Not identified"
        row["cause"] = row["potential_cause"]
    else:
        row["potential_cause"] = row.get("potential_cause") or row.get("cause")
        row["cause"] = row["potential_cause"]

    if final_cell_is_missing(row.get("consequence")):
        row["consequence"] = "Not identified" if row_has_credible_effect_and_cause(row) else "Not applicable"

    if final_cell_is_missing(row.get("safeguard") or row.get("safeguards")):
        row["safeguard"] = default_row_safeguard(row, node)
        row["safeguards"] = row["safeguard"]
    else:
        row["safeguard"] = row.get("safeguard") or row.get("safeguards")
        row["safeguards"] = row["safeguard"]

    action_value = row.get("necessary_action") or row.get("action_required")
    if final_cell_is_missing(action_value) or action_is_generic(action_value):
        row["necessary_action"] = "Not identified"
        row["action_required"] = row["necessary_action"]
    else:
        row["necessary_action"] = row.get("necessary_action") or row.get("action_required")
        row["action_required"] = row["necessary_action"]
    return row


def credible_row_count(rows: List[Dict[str, Any]]) -> int:
    return sum(1 for row in rows if isinstance(row, dict) and not row_is_not_applicable(row))


def best_row_source(*sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [source for source in sources if isinstance(source, list) and source]
    if not candidates:
        return []
    return max(candidates, key=lambda rows: (credible_row_count(rows), len(rows)))


def rows_from_round_one(node: Dict[str, Any], round_one: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for agent in round_one:
        if not isinstance(agent, dict):
            continue
        agent_name = normalize_whitespace(agent.get("agent_name")) or "Expert agent"
        for scenario in agent.get("scenarios") or []:
            if not isinstance(scenario, dict):
                continue
            if scenario.get("credible") is False:
                continue
            row = scenario_to_row(scenario, node, len(rows) + 1, agent_name)
            if not normalize_whitespace(row.get("guideword")):
                continue
            if row_is_not_applicable(row):
                continue
            key = (
                normalized_guideword(row.get("guideword")).lower(),
                row_parameter(row, node_parameters(node)).lower(),
                normalize_whitespace(row.get("deviation")).lower(),
                normalize_whitespace(row.get("potential_cause")).lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) >= MAX_ROWS_PER_NODE:
                return rows
    return rows


def round_one_atomic_candidates(
    node: Dict[str, Any],
    round_one: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Expand Round 1 contributions to one candidate per initiating cause.

    Consequences remain attached to their cause.  Treating every consequence
    bullet as a separate scenario previously inflated the review universe even
    when the initiating event and deviation were identical.
    """
    candidates: List[Dict[str, Any]] = []
    source_index = 0
    for agent in round_one:
        if not isinstance(agent, dict) or agent.get("failed"):
            continue
        source_agent = normalize_whitespace(agent.get("agent_name")) or "Expert agent"
        for scenario in agent.get("scenarios") or []:
            if not isinstance(scenario, dict) or scenario.get("credible") is False:
                continue
            source_index += 1
            source_id = f"{node['node_id']}-R1SRC-{source_index:03d}"
            source_row_id = normalize_whitespace(scenario.get("row_id") or scenario.get("id"))
            base_row = scenario_to_row(scenario, node, source_index, source_agent)
            deviations = row_text_items(scenario.get("deviation") or scenario.get("effect"), 20)
            causes = row_text_items(
                scenario.get("potential_cause")
                or scenario.get("cause")
                or scenario.get("potential_causes")
                or scenario.get("causes"),
                20,
            )
            consequences = row_text_items(
                scenario.get("consequence") or scenario.get("consequences"), 20
            )
            atomic_count = max(1, len(deviations), len(causes))
            for atomic_index in range(atomic_count):
                candidate_id = (
                    source_id
                    if atomic_count == 1
                    else f"{source_id}#C{atomic_index + 1}"
                )
                row = dict(base_row)
                row["source_candidate_id"] = candidate_id
                row["source_candidate_ids"] = [candidate_id]
                row["source_row_id"] = source_row_id
                row["source_row_ids"] = [source_row_id] if source_row_id else []
                row["source_agent"] = source_agent
                row["source_agents"] = [source_agent]
                row["round_1_claimed_credible"] = scenario.get("credible", True)
                if len(deviations) > 1:
                    row["deviation"] = (
                        deviations[atomic_index]
                        if atomic_index < len(deviations) else deviations[0]
                    )
                if len(causes) > 1:
                    cause = causes[atomic_index] if atomic_index < len(causes) else causes[0]
                    row["potential_cause"] = cause
                    row["cause"] = cause
                # Pair consequences only when the model clearly supplied one
                # consequence per cause; otherwise retain the full consequence
                # set on the single cause-level scenario.
                if atomic_count > 1 and len(consequences) == atomic_count:
                    row["consequence"] = consequences[atomic_index]
                elif consequences:
                    row["consequence"] = consequences
                candidates.append(row)
    return candidates


def round_one_candidate_context(row: Dict[str, Any], node: Dict[str, Any]) -> str:
    matrix_key = normalize_whitespace(row.get("matrix_key")).lower()
    if matrix_key:
        return matrix_key
    return "::".join([
        row_parameter(row, node_parameters(node)).lower(),
        normalized_guideword(row.get("guideword")).lower(),
        normalize_whitespace(row.get("deviation") or row.get("effect")).lower(),
    ])


def round_one_cause_signature(row: Dict[str, Any]) -> Tuple[str, Set[str]]:
    text = compact_hazop_phrase(
        row.get("potential_cause") or row.get("cause") or "",
        "potential_cause",
    ).lower().rstrip(".")
    replacements = {
        "fails": "fail", "failed": "fail", "failure": "fail",
        "closed": "close", "closing": "close", "opened": "open",
        "opening": "open", "blocked": "block", "blockage": "block",
        "incorrect": "wrong", "incorrectly": "wrong",
    }
    stop_words = {
        "the", "a", "an", "is", "of", "to", "for", "and", "or",
        "may", "can", "could", "would", "in", "on", "at", "from",
    }
    tokens = {
        replacements.get(token, token)
        for token in re.findall(r"[a-z0-9]+", text)
        if replacements.get(token, token) not in stop_words
    }
    return text, tokens


def round_one_candidates_match(
    left: Dict[str, Any], right: Dict[str, Any], node: Dict[str, Any]
) -> bool:
    if round_one_candidate_context(left, node) != round_one_candidate_context(right, node):
        return False
    left_text, left_tokens = round_one_cause_signature(left)
    right_text, right_tokens = round_one_cause_signature(right)
    if left_text and left_text == right_text:
        return True
    union = left_tokens | right_tokens
    overlap = len(left_tokens & right_tokens) / len(union) if union else 0.0
    return len(left_tokens & right_tokens) >= 2 and overlap >= 0.75


def merge_round_one_candidate_lineage(
    target: Dict[str, Any], source: Dict[str, Any]
) -> None:
    source_candidate_ids = sanitize_list(source.get("source_candidate_ids"), 200)
    if not source_candidate_ids:
        source_candidate_ids = sanitize_list(source.get("source_candidate_id"), 1)
    source_row_ids = sanitize_list(source.get("source_row_ids"), 200)
    if not source_row_ids:
        source_row_ids = sanitize_list(source.get("source_row_id"), 1)
    source_agents = sanitize_list(source.get("source_agents"), 20)
    if not source_agents:
        source_agents = sanitize_list(source.get("source_agent"), 1)
    target["source_candidate_ids"] = unique_strings(
        sanitize_list(target.get("source_candidate_ids"), 200)
        + source_candidate_ids
    )
    target["source_row_ids"] = unique_strings(
        sanitize_list(target.get("source_row_ids"), 200)
        + source_row_ids
    )
    target["source_agents"] = unique_strings(
        sanitize_list(target.get("source_agents"), 20)
        + source_agents
    )
    target["source_agent"] = ", ".join(target["source_agents"])
    for field in ("consequence", "safeguard", "safeguards", "necessary_action", "action_required"):
        combined = compact_hazop_items(
            row_text_items(target.get(field), 20) + row_text_items(source.get(field), 20),
            field,
            20,
        )
        if combined:
            target[field] = combined


def deduplicate_round_one_candidates(
    node: Dict[str, Any], candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    unique_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        existing = next(
            (
                row for row in unique_rows
                if round_one_candidates_match(row, candidate, node)
            ),
            None,
        )
        if existing is not None:
            merge_round_one_candidate_lineage(existing, candidate)
            continue
        unique_rows.append(dict(candidate))
    return unique_rows


def round_one_review_universe(
    node: Dict[str, Any],
    round_one: List[Dict[str, Any]],
    draft_rows: Optional[List[Dict[str, Any]]] = None,
    raw_candidates: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build the unique cause-level set reviewed in Round 2A.

    The Chair draft is reconciled with the deduplicated expert set before the
    Round 1-to-Round 2 handoff. Raw candidates remain linked as provenance,
    but duplicate contributions are not assigned repeatedly to every reviewer.
    """
    raw_candidates = (
        list(raw_candidates)
        if raw_candidates is not None
        else round_one_atomic_candidates(node, round_one)
    )
    unique_raw_candidates = deduplicate_round_one_candidates(node, raw_candidates)
    consolidated: List[Dict[str, Any]] = []
    matched_source_candidate_ids: Set[str] = set()
    for index, draft in enumerate(draft_rows or [], 1):
        if not isinstance(draft, dict) or not normalize_whitespace(draft.get("guideword")):
            continue
        draft_row = scenario_to_row(draft, node, index, "Panel Chair consolidation")
        causes = row_text_items(
            draft.get("potential_cause") or draft.get("cause")
            or draft.get("potential_causes") or draft.get("causes"),
            20,
        )
        cause_count = max(1, len(causes))
        for cause_index in range(cause_count):
            row = dict(draft_row)
            if causes:
                cause = causes[cause_index]
                row["potential_cause"] = cause
                row["cause"] = cause
            row["source_candidate_ids"] = []
            row["source_row_ids"] = []
            row["source_agents"] = sanitize_list(draft.get("source_agents"), 20)
            for raw in unique_raw_candidates:
                if round_one_candidates_match(row, raw, node):
                    merge_round_one_candidate_lineage(row, raw)
                    matched_source_candidate_ids.update(
                        sanitize_list(raw.get("source_candidate_ids"), 200)
                        + sanitize_list(raw.get("source_candidate_id"), 1)
                    )
            if not row["source_agents"]:
                row["source_agents"] = ["Panel Chair consolidation"]
                row["source_agent"] = "Panel Chair consolidation"
            row["consolidation_status"] = "chair_consolidated"
            consolidated.append(row)

    # A Chair draft should consolidate, not erase, expert findings. Preserve
    # any unique cause-level contribution that could not be conservatively
    # linked to a Chair row and expose that status in the saved JSON.
    if consolidated:
        for raw in unique_raw_candidates:
            source_ids = set(
                sanitize_list(raw.get("source_candidate_ids"), 200)
                + sanitize_list(raw.get("source_candidate_id"), 1)
            )
            if source_ids and source_ids <= matched_source_candidate_ids:
                continue
            retained = dict(raw)
            retained["consolidation_status"] = "unique_raw_candidate_preserved"
            consolidated.append(retained)

    review_rows = deduplicate_round_one_candidates(
        node,
        consolidated if consolidated else unique_raw_candidates,
    )
    for index, row in enumerate(review_rows, 1):
        review_id = f"{node['node_id']}-R1-{index:03d}"
        row["review_id"] = review_id
        row["row_id"] = review_id
        row["review_unit"] = "unique_reconciled_round_1_cause"
    return review_rows


def rows_from_challenges(node: Dict[str, Any], challenges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for agent in challenges:
        if not isinstance(agent, dict):
            continue
        agent_name = normalize_whitespace(agent.get("agent_name")) or "Expert agent"
        for scenario in (
            (agent.get("consensus_recommendations") or [])
            + (agent.get("new_scenarios") or agent.get("missing_scenarios") or [])
        ):
            if isinstance(scenario, dict):
                if "recommended_deviation" in scenario or "recommended_potential_cause" in scenario:
                    scenario = {
                        "guideword": scenario.get("guideword", ""),
                        "parameter": scenario.get("parameter", ""),
                        "deviation": scenario.get("recommended_deviation", ""),
                        "potential_cause": scenario.get("recommended_potential_cause", ""),
                        "consequence": scenario.get("recommended_consequence", ""),
                        "safeguard": scenario.get("recommended_safeguard", ""),
                        "necessary_action": scenario.get("recommended_action", ""),
                        "assumptions": [scenario.get("basis", "")] if scenario.get("basis") else [],
                    }
                rows.append(scenario_to_row(scenario, node, len(rows) + 1, agent_name))
    return rows[:MAX_ROWS_PER_NODE]


def rows_from_new_scenario_deliberations(
    node: Dict[str, Any],
    deliberations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert only panel-supported Round 2 additions into fallback rows."""
    rows: List[Dict[str, Any]] = []
    for item in deliberations or []:
        if not isinstance(item, dict) or not item.get("panel_supported"):
            continue
        proposal = item.get("proposal")
        if not isinstance(proposal, dict):
            continue
        source_agents = sanitize_list(proposal.get("source_agents"), 10)
        source = ", ".join(source_agents) or "Round 2 panel"
        rows.append(scenario_to_row(proposal, node, len(rows) + 1, source))
    return rows[:MAX_ROWS_PER_NODE]


def attach_final_row_lineage(
    node: Dict[str, Any],
    final_rows: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
    round_two_outputs: List[Dict[str, Any]],
    new_scenario_deliberations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Attach auditable Round 1/2 provenance to each final atomic row.

    Matching is deliberately conservative: rows must share the matrix item and
    have the same normalized cause or substantial cause-token overlap. Unclear
    matches remain unlinked rather than receiving fabricated provenance.
    """

    def matrix_identity(row: Dict[str, Any]) -> str:
        explicit = normalize_whitespace(row.get("matrix_key")).lower()
        if explicit:
            return explicit
        return "::".join([
            normalize_whitespace(row.get("parameter")).lower(),
            normalized_guideword(row.get("guideword")).lower(),
        ])

    def cause_identity(row: Dict[str, Any]) -> str:
        return compact_hazop_phrase(
            row.get("potential_cause") or row.get("cause") or "",
            "potential_cause",
        ).lower().rstrip(".")

    def cause_tokens(row: Dict[str, Any]) -> Set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", cause_identity(row))
            if len(token) > 2 and token not in {"the", "and", "for", "with", "from"}
        }

    def matches(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        if matrix_identity(left) != matrix_identity(right):
            return False
        left_cause = cause_identity(left)
        right_cause = cause_identity(right)
        if not left_cause or not right_cause:
            return False
        if left_cause == right_cause:
            return True
        left_tokens = cause_tokens(left)
        right_tokens = cause_tokens(right)
        union = left_tokens | right_tokens
        return bool(union) and len(left_tokens & right_tokens) / len(union) >= 0.65

    reviews_by_agent = {
        normalize_whitespace(output.get("agent_name")): {
            normalize_whitespace(review.get("review_id")): review
            for review in output.get("scenario_reviews", []) or []
            if isinstance(review, dict) and normalize_whitespace(review.get("review_id"))
        }
        for output in round_two_outputs
        if isinstance(output, dict)
    }
    proposals = [
        item for item in new_scenario_deliberations
        if isinstance(item, dict) and isinstance(item.get("proposal"), dict)
    ]
    linked_rows: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    for raw_row in final_rows:
        row = dict(raw_row)
        matched_reviews = [
            source for source in review_rows
            if isinstance(source, dict) and matches(row, source)
        ]
        matched_proposals = [
            item for item in proposals if matches(row, item["proposal"])
        ]
        review_ids = unique_strings([
            normalize_whitespace(item.get("review_id")) for item in matched_reviews
        ])
        proposal_ids = unique_strings([
            normalize_whitespace(item.get("proposal_id")) for item in matched_proposals
        ])
        source_agents = unique_strings(
            [normalize_whitespace(item.get("source_agent")) for item in matched_reviews]
            + [
                normalize_whitespace(agent)
                for item in matched_proposals
                for agent in (item.get("proposal", {}).get("source_agents") or [])
            ]
        )
        decisions = {
            agent_name: {
                review_id: normalized_round_two_decision(agent_reviews.get(review_id, {}).get("decision"))
                for review_id in review_ids
                if review_id in agent_reviews
            }
            for agent_name, agent_reviews in reviews_by_agent.items()
        }
        decisions = {agent: values for agent, values in decisions.items() if values}
        row["source_review_ids"] = review_ids
        row["source_proposal_ids"] = proposal_ids
        row["source_agents"] = source_agents
        row["round_2_decisions"] = decisions
        row["lineage_status"] = "linked" if review_ids or proposal_ids else "unlinked"
        linked_rows.append(row)
        lineage.append({
            "row_id": row.get("row_id"),
            "parent_row_id": row.get("parent_row_id"),
            "cause_id": row.get("cause_id"),
            "node_id": node.get("node_id"),
            "matrix_key": row.get("matrix_key"),
            "source_review_ids": review_ids,
            "source_proposal_ids": proposal_ids,
            "source_agents": source_agents,
            "round_2_decisions": decisions,
            "chair_adjudication_id": row.get("chair_adjudication_id", ""),
            "chair_criterion_assessment": row.get("chair_criterion_assessment", {}),
            "round_2_panel_basis": row.get("round_2_panel_basis", {}),
            "chair_suggested_decision": row.get("chair_suggested_decision", ""),
            "chair_decision": row.get("chair_decision", ""),
            "chair_decision_overridden_by_rubric": bool(
                row.get("chair_decision_overridden_by_rubric", False)
            ),
            "chair_final_treatment": row.get("chair_final_treatment", ""),
            "chair_evidence_basis": row.get("chair_evidence_basis", []),
            "chair_revision_summary": row.get("chair_revision_summary", ""),
            "chair_review_status": row.get("chair_review_status", ""),
            "chair_resolution": row.get("chair_resolution", ""),
            "chair_authorized": bool(row.get("chair_authorized", True)),
            "lineage_status": row["lineage_status"],
        })
    return linked_rows, lineage


def repair_not_applicable_rows_from_fallbacks(
    final_rows: List[Dict[str, Any]],
    node: Dict[str, Any],
    *fallback_sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Preserve the final authority's applicability decisions.

    Older versions replaced final ``Not applicable`` rows with earlier draft
    candidates. That reversed the Chair/Scribe decision and allowed rejected
    scenarios back into the worksheet. Fallback sources remain in the
    signature for saved-call compatibility but are intentionally not applied.
    """
    # Applicability belongs to the matrix-coverage audit, not to the HAZOP
    # worksheet. Never manufacture or retain a worksheet row whose technical
    # cells merely say "Not applicable".
    return [
        dict(row) for row in final_rows
        if isinstance(row, dict) and not row_is_not_applicable(row)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting and HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_provider_slot(provider: str) -> None:
    interval = PROVIDER_MIN_INTERVAL_SECONDS.get(provider, 0.0)
    if interval <= 0:
        return
    lock = PROVIDER_RATE_LIMIT_LOCKS.setdefault(provider, threading.Lock())
    with lock:
        now = time.monotonic()
        next_allowed_at = PROVIDER_NEXT_ALLOWED_AT.get(provider, 0.0)
        if next_allowed_at > now:
            time.sleep(next_allowed_at - now)
        PROVIDER_NEXT_ALLOWED_AT[provider] = time.monotonic() + interval


def request_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    provider: str = "",
    retries: int = 0,
    max_retries: int = 4,
) -> Dict[str, Any]:
    wait_for_provider_slot(provider)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        limiter = PROVIDER_ACTIVE_REQUEST_LIMITS.get(provider)
        if limiter:
            with limiter:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    return json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="ignore")
        if error.code in {429, 500, 502, 503, 504} and retries < max_retries:
            retry_after = error.headers.get("Retry-After") if error.headers else ""
            try:
                delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                delay = 0.0
            if delay <= 0:
                delay = min(60.0, (2 ** retries) * 4.0) + random.uniform(0, 1.5)
            time.sleep(delay)
            return request_json(url, headers, payload, provider, retries + 1, max_retries=max_retries)
        raise RuntimeError(f"HTTP {error.code} from {provider}: {details[:800]}")
    except urllib.error.URLError as error:
        reason = normalize_whitespace(getattr(error, "reason", ""))
        if retries < max_retries and (
            "timed out" in reason.lower()
            or "timeout" in reason.lower()
            or "temporarily unavailable" in reason.lower()
        ):
            delay = min(90.0, (2 ** retries) * 4.0) + random.uniform(0, 1.5)
            time.sleep(delay)
            return request_json(url, headers, payload, provider, retries + 1, max_retries=max_retries)
        raise RuntimeError(f"Network error contacting {provider}: {error.reason}")
    except (TimeoutError, socket.timeout) as error:
        if retries < max_retries:
            delay = min(90.0, (2 ** retries) * 4.0) + random.uniform(0, 1.5)
            time.sleep(delay)
            return request_json(url, headers, payload, provider, retries + 1, max_retries=max_retries)
        raise RuntimeError(f"Network timeout contacting {provider}: {error}")


def unique_strings(items: List[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        text = normalize_whitespace(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def is_transient_provider_error(error: Exception, provider: str) -> bool:
    message = str(error)
    if f"Network error contacting {provider}" in message:
        return True
    return any(f"HTTP {code} from {provider}" in message for code in (429, 500, 502, 503, 504))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CALLERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_openai_responses_text(response: Dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(response.get("output", []))
    return "\n".join(chunks).strip()


def call_openai_responses(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    thinking_level: str,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        ],
        "max_output_tokens": 16000,
        "text": {"format": {"type": "json_object"}},
        "store": False,
    }
    if openai_supports_temperature(model):
        payload["temperature"] = temperature
    if openai_supports_reasoning_effort(model):
        effort = openai_reasoning_effort(thinking_level, model)
        if effort:
            payload["reasoning"] = {"effort": effort}
    response = request_json(
        "https://api.openai.com/v1/responses",
        {"content-type": "application/json", "authorization": f"Bearer {api_key}"},
        payload,
        provider="openai",
    )
    text = extract_openai_responses_text(response)
    if text:
        return text
    raise RuntimeError("OpenAI Responses API response did not include output text")


def call_openai(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    thinking_level: str,
) -> str:
    try:
        return call_openai_responses(api_key, model, system_prompt, user_prompt, temperature, thinking_level)
    except RuntimeError as error:
        if openai_responses_only(model):
            raise
        message = str(error)
        if "HTTP 400 from openai" not in message and "HTTP 404 from openai" not in message:
            raise
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 16000,
        "response_format": {"type": "json_object"},
    }
    if openai_supports_temperature(model):
        payload["temperature"] = temperature
    response = request_json(
        "https://api.openai.com/v1/chat/completions",
        {"content-type": "application/json", "authorization": f"Bearer {api_key}"},
        payload,
        provider="openai",
    )
    choices = response.get("choices") or []
    if choices and choices[0].get("message"):
        return choices[0]["message"].get("content", "").strip()
    raise RuntimeError("OpenAI response did not include output text")


def call_anthropic(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    thinking_level: str,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": 16000,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    response = request_anthropic_messages(api_key, payload, thinking_level, model)
    text = extract_anthropic_text(response)
    if text:
        return text
    raise RuntimeError("Anthropic response did not include output text")


def request_anthropic_messages(
    api_key: str,
    payload: Dict[str, Any],
    thinking_level: str,
    model: str,
) -> Dict[str, Any]:
    thinking = anthropic_thinking_payload(thinking_level, model)
    if thinking:
        payload["thinking"] = thinking
        if thinking.get("type") == "adaptive":
            effort = anthropic_output_effort(thinking_level)
            if effort:
                payload["output_config"] = {"effort": effort}
        elif thinking.get("type") == "enabled":
            # Classic (budget_tokens) extended thinking is incompatible with a custom
            # temperature on these models — omit it rather than send any value.
            payload.pop("temperature", None)
    if not anthropic_supports_temperature(model):
        payload.pop("temperature", None)
    try:
        response = request_json(
            "https://api.anthropic.com/v1/messages",
            {
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            payload,
            provider="anthropic",
        )
    except RuntimeError as error:
        message = str(error)
        if thinking and "HTTP 400 from anthropic" in message:
            payload.pop("thinking", None)
            payload.pop("output_config", None)
            response = request_json(
                "https://api.anthropic.com/v1/messages",
                {
                    "content-type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                payload,
                provider="anthropic",
            )
        else:
            raise
    return response


def extract_anthropic_text(response: Dict[str, Any]) -> str:
    parts = [part.get("text", "") for part in response.get("content", []) if part.get("type") == "text"]
    return "\n".join(part for part in parts if part).strip()


def extract_gemini_text(response: Dict[str, Any]) -> str:
    chunks: List[str] = []

    output_text = response.get("output_text") or response.get("text")
    if isinstance(output_text, str) and output_text.strip():
        chunks.append(output_text)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for key in ("output", "outputs", "steps", "candidates"):
        if key in response:
            walk(response[key])

    seen = set()
    unique_chunks = []
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique_chunks.append(stripped)
    return "\n".join(unique_chunks).strip()


def call_gemini_generate_content(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    thinking_level: str,
) -> str:
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "responseMimeType": "application/json",
        "maxOutputTokens": GEMINI_TEXT_OUTPUT_TOKENS,
    }
    thinking_config = gemini_generate_content_thinking_config(thinking_level, model)
    if thinking_config:
        generation_config["thinkingConfig"] = thinking_config
    response = request_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"content-type": "application/json", "x-goog-api-key": api_key},
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        },
        provider="gemini",
    )
    text = extract_gemini_text(response)
    if text:
        return text
    raise RuntimeError("Gemini generateContent response did not include generated text")


def call_ollama(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    """
    Ollama: prepend system prompt into the user prompt when using
    the /api/generate endpoint, since some Ollama models ignore the system field.
    Also combine system + user into a single prompt to improve JSON compliance.
    """
    # Merge system + user so local models can't ignore system field
    combined_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    request_payload = {
        "model": model,
        "prompt": combined_prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 16384},
    }
    response = request_ollama_generate(api_key, request_payload)
    text = response.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError("Ollama response did not include generated text")


def request_ollama_generate(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    local_url = "http://localhost:11434/api/generate"
    local_headers = {"content-type": "application/json"}
    if not api_key:
        return request_json(local_url, local_headers, payload, provider="ollama")
    cloud_headers = {"content-type": "application/json", "authorization": f"Bearer {api_key}"}
    try:
        return request_json("https://ollama.com/api/generate", cloud_headers, payload, provider="ollama")
    except RuntimeError as error:
        message = str(error)
        if "HTTP 401 from ollama" in message or "HTTP 403 from ollama" in message:
            return request_json(local_url, local_headers, payload, provider="ollama")
        raise


def split_data_url(data_url: str) -> Dict[str, str]:
    match = re.match(r"data:(image/[a-zA-Z0-9.+-]+);base64,(.+)", data_url or "", re.DOTALL)
    if not match:
        raise ValueError("Image payload must be a data:image/...;base64 URL.")
    return {"media_type": match.group(1), "data": match.group(2)}


def call_openai_vision(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[str],
    temperature: float,
    thinking_level: str,
) -> str:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
    for image in images:
        content.append({"type": "input_image", "image_url": image})
    payload: Dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 24000,
        "text": {"format": {"type": "json_object"}},
        "store": False,
    }
    if openai_supports_temperature(model):
        payload["temperature"] = temperature
    if openai_supports_reasoning_effort(model):
        effort = openai_reasoning_effort(thinking_level, model)
        if effort:
            payload["reasoning"] = {"effort": effort}
    try:
        response = request_json(
            "https://api.openai.com/v1/responses",
            {"content-type": "application/json", "authorization": f"Bearer {api_key}"},
            payload,
            provider="openai",
        )
        text = extract_openai_responses_text(response)
        if text:
            return text
        raise RuntimeError("OpenAI vision Responses API response did not include output text")
    except RuntimeError as error:
        if openai_responses_only(model):
            raise
        message = str(error)
        if "HTTP 400 from openai" in message and "temperature" in message.lower():
            raise
        if "HTTP 400 from openai" not in message and "HTTP 404 from openai" not in message:
            raise

    content = [{"type": "text", "text": user_prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image, "detail": "high"}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": 24000,
    }
    if openai_supports_temperature(model):
        payload["temperature"] = temperature
    response = request_json(
        "https://api.openai.com/v1/chat/completions",
        {"content-type": "application/json", "authorization": f"Bearer {api_key}"},
        payload,
        provider="openai",
    )
    choices = response.get("choices") or []
    if choices and choices[0].get("message"):
        return choices[0]["message"].get("content", "").strip()
    raise RuntimeError("OpenAI vision response did not include output text")


def call_anthropic_vision(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[str],
    temperature: float,
    thinking_level: str,
) -> str:
    content: List[Dict[str, Any]] = []
    for image in images:
        parsed = split_data_url(image)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": parsed["media_type"], "data": parsed["data"]},
        })
    content.append({"type": "text", "text": user_prompt})
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": 24000,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
    }
    response = request_anthropic_messages(api_key, payload, thinking_level, model)
    text = extract_anthropic_text(response)
    if text:
        return text
    raise RuntimeError("Anthropic vision response did not include output text")


def call_gemini_vision_generate_content(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[str],
    temperature: float,
    thinking_level: str,
) -> str:
    parts: List[Dict[str, Any]] = [{"text": user_prompt}]
    for image in images:
        parsed = split_data_url(image)
        parts.append({"inline_data": {"mime_type": parsed["media_type"], "data": parsed["data"]}})
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "responseMimeType": "application/json",
        "maxOutputTokens": GEMINI_VISION_OUTPUT_TOKENS,
    }
    thinking_config = gemini_generate_content_thinking_config(thinking_level, model)
    if thinking_config:
        generation_config["thinkingConfig"] = thinking_config
    response = request_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"content-type": "application/json", "x-goog-api-key": api_key},
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        },
        provider="gemini",
    )
    text = extract_gemini_text(response)
    if text:
        return text
    raise RuntimeError("Gemini vision generateContent response did not include generated text")


def call_ollama_vision(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[str],
    temperature: float,
) -> str:
    image_payload = [split_data_url(image)["data"] for image in images]
    combined_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    response = request_ollama_generate(
        api_key,
        {
            "model": model,
            "prompt": combined_prompt,
            "images": image_payload,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 16384},
        },
    )
    text = response.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError("Ollama vision response did not include generated text")


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER SELECTION & MODEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def select_provider(agent_id: str, preferred: str, keys: Dict[str, str]) -> str:
    preferred = (preferred or "auto").lower()
    if preferred in {"openai", "anthropic", "gemini", "ollama"}:
        if preferred != "ollama" and not keys.get(preferred):
            raise ValueError(f"{agent_id} is configured for {preferred}, but no API key was provided.")
        return preferred
    for provider in AUTO_PROVIDER_ORDER.get(agent_id, ["openai", "anthropic", "gemini", "ollama"]):
        if provider == "ollama" or keys.get(provider):
            return provider
    raise ValueError("No API key was supplied and Ollama was not selected.")


def provider_for_model(model: str) -> str:
    model = normalize_whitespace(model)
    if not model:
        return ""
    for provider, models in AVAILABLE_MODELS.items():
        if model in models:
            return provider
    lower = model.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("gemini"):
        return "gemini"
    return ""


def resolve_runtime(agent_id: str, config: Dict[str, Any], keys: Dict[str, str]) -> Dict[str, str]:
    requested_provider = normalize_whitespace(config.get("provider", "auto")).lower() or "auto"
    requested_model = normalize_whitespace(config.get("model"))
    if requested_model.startswith("models/"):
        requested_model = requested_model.split("/", 1)[1]
    inferred_provider = provider_for_model(requested_model)
    if requested_provider == "auto" and inferred_provider:
        provider = select_provider(agent_id, inferred_provider, keys)
    else:
        provider = select_provider(agent_id, requested_provider, keys)
    if requested_model and inferred_provider and inferred_provider != provider:
        raise ValueError(
            f"{agent_id} selected model '{requested_model}' belongs to {inferred_provider}, "
            f"but provider is set to {provider}. Set provider to {inferred_provider} or choose a {provider} model."
        )
    if provider == "gemini" and requested_model in GEMINI_UNAVAILABLE_FOR_THIS_APP:
        requested_model = DEFAULT_MODELS["gemini"]
    model = requested_model or DEFAULT_MODELS[provider]
    return {"provider": provider, "model": model}


def model_call_record(
    agent_id: str,
    config: Dict[str, Any],
    provider: str,
    model: str,
    temperature: float,
    thinking_level: str,
    system_prompt: str,
    user_prompt: str,
    trace: Optional[Dict[str, Any]],
    default_attempt_role: str,
    image_count: Optional[int] = None,
) -> Dict[str, Any]:
    record = {
        "call_id": uuid.uuid4().hex,
        "agent_id": agent_id,
        "stage": normalize_whitespace((trace or {}).get("stage")),
        "node_id": normalize_whitespace((trace or {}).get("node_id")),
        "round": normalize_whitespace((trace or {}).get("round")),
        "attempt_role": normalize_whitespace((trace or {}).get("attempt_role")) or default_attempt_role,
        "requested_provider": normalize_whitespace(config.get("provider", "auto")).lower() or "auto",
        "requested_model": normalize_whitespace(config.get("model")),
        "actual_provider": provider,
        "actual_model": model,
        "requested_temperature": temperature,
        "temperature": temperature if provider != "openai" or openai_supports_temperature(model) else None,
        "temperature_parameter_used": provider != "openai" or openai_supports_temperature(model),
        "thinking_level": thinking_level,
        "system_prompt_hash": short_hash(system_prompt),
        "user_prompt_hash": short_hash(user_prompt),
        "system_prompt_chars": len(system_prompt or ""),
        "user_prompt_chars": len(user_prompt or ""),
        "estimated_input_tokens": estimated_tokens(system_prompt) + estimated_tokens(user_prompt),
        "started_at": utc_now(),
    }
    if image_count is not None:
        record["image_count"] = image_count
    return record


def finish_model_call(
    job_id: str,
    record: Dict[str, Any],
    start: float,
    status: str,
    response: str = "",
    error: Optional[Exception] = None,
) -> None:
    completed = dict(record)
    completed["completed_at"] = utc_now()
    completed["latency_seconds"] = round(max(0.0, time.monotonic() - start), 4)
    completed["status"] = status
    completed["response_chars"] = len(response or "")
    completed["estimated_output_tokens"] = estimated_tokens(response)
    completed["response_hash"] = short_hash(response) if response else ""
    if error is not None:
        completed["error_type"] = type(error).__name__
        completed["error_message"] = str(error)[:1000]
    append_model_call(job_id, completed)


def run_logged_model_call(job_id: str, record: Dict[str, Any], call: Callable[[], str]) -> str:
    start = time.monotonic()
    try:
        response = call()
        finish_model_call(job_id, record, start, "success", response=response)
        return response
    except Exception as error:
        finish_model_call(job_id, record, start, "error", error=error)
        raise


def run_model(
    agent_id: str,
    config: Dict[str, Any],
    keys: Dict[str, str],
    system_prompt: str,
    user_prompt: str,
    trace: Optional[Dict[str, Any]] = None,
) -> str:
    runtime = resolve_runtime(agent_id, config, keys)
    provider = runtime["provider"]
    model = runtime["model"]
    temperature = clamp_temperature(config.get("temperature"), default_agent_temperature(agent_id))
    thinking_level = agent_thinking_level(agent_id, config.get("thinking_level"))
    job_id = normalize_whitespace((trace or {}).get("job_id"))
    base_record = model_call_record(
        agent_id,
        config,
        provider,
        model,
        temperature,
        thinking_level,
        system_prompt,
        user_prompt,
        trace,
        "primary",
    )

    if provider == "openai":
        return run_logged_model_call(
            job_id,
            base_record,
            lambda: call_openai(keys["openai"], model, system_prompt, user_prompt, temperature, thinking_level),
        )
    if provider == "anthropic":
        return run_logged_model_call(
            job_id,
            base_record,
            lambda: call_anthropic(keys["anthropic"], model, system_prompt, user_prompt, temperature, thinking_level),
        )
    if provider == "gemini":
        last_error: Optional[Exception] = None
        models = unique_strings([model] + TEXT_MODEL_FALLBACKS.get("gemini", []))
        for attempt_index, candidate_model in enumerate(models, 1):
            record = dict(base_record)
            record["actual_model"] = candidate_model
            record["provider_fallback_attempt"] = attempt_index
            record["provider_fallback_used"] = candidate_model != model
            try:
                def gemini_call(candidate: str = candidate_model) -> str:
                    return call_gemini_generate_content(
                        keys["gemini"],
                        candidate,
                        system_prompt,
                        user_prompt,
                        temperature,
                        thinking_level,
                    )

                return run_logged_model_call(
                    job_id,
                    record,
                    gemini_call,
                )
            except RuntimeError as error:
                last_error = error
                if not is_transient_provider_error(error, "gemini"):
                    raise
                continue
        if last_error:
            raise last_error
        raise RuntimeError("Gemini call failed before a model candidate was attempted.")
    return run_logged_model_call(
        job_id,
        base_record,
        lambda: call_ollama(keys.get("ollama", ""), model, system_prompt, user_prompt, temperature),
    )


def run_vision_model(
    agent_id: str,
    config: Dict[str, Any],
    keys: Dict[str, str],
    system_prompt: str,
    user_prompt: str,
    images: List[str],
    trace: Optional[Dict[str, Any]] = None,
) -> str:
    runtime = resolve_runtime(agent_id, config, keys)
    provider = runtime["provider"]
    model = runtime["model"]
    temperature = 0.0 if agent_id == "pid_image_parser" else clamp_temperature(config.get("temperature"), default_agent_temperature(agent_id))
    thinking_level = agent_thinking_level(agent_id, config.get("thinking_level"))
    job_id = normalize_whitespace((trace or {}).get("job_id"))
    record = model_call_record(
        agent_id,
        config,
        provider,
        model,
        temperature,
        thinking_level,
        system_prompt,
        user_prompt,
        trace,
        "vision",
        image_count=len(images or []),
    )
    if not record["stage"]:
        record["stage"] = "pid_image_parse"
    if provider == "openai":
        call = lambda: call_openai_vision(keys["openai"], model, system_prompt, user_prompt, images, temperature, thinking_level)
    elif provider == "anthropic":
        call = lambda: call_anthropic_vision(keys["anthropic"], model, system_prompt, user_prompt, images, temperature, thinking_level)
    elif provider == "gemini":
        call = lambda: call_gemini_vision_generate_content(keys["gemini"], model, system_prompt, user_prompt, images, temperature, thinking_level)
    else:
        call = lambda: call_ollama_vision(keys.get("ollama", ""), model, system_prompt, user_prompt, images, temperature)
    return run_logged_model_call(job_id, record, call)


def vision_runtime_supported(agent_id: str, config: Dict[str, Any], keys: Dict[str, str]) -> bool:
    runtime = resolve_runtime(agent_id, config, keys)
    provider = runtime["provider"]
    model = runtime["model"].lower()
    if provider in {"openai", "anthropic", "gemini"}:
        return True
    if provider == "ollama":
        vision_markers = (
            "vision", "llava", "bakllava", "moondream", "minicpm",
            "qwen2-vl", "qwen2.5vl", "qwen-vl", "qwen3.5", "qwen3.6", "qwen3.8", "qwen3-vl",
            "gemma4", "gemma-4", "minimax-m3", "kimi-k3", "mistral-large-3", "llama4",
        )
        return any(marker in model for marker in vision_markers)
    return False


def vision_fallback_configs(agent_id: str, config: Dict[str, Any], keys: Dict[str, str]) -> List[Dict[str, Any]]:
    preferred = normalize_whitespace(config.get("provider")).lower() or "auto"
    base_temperature = clamp_temperature(config.get("temperature"), default_agent_temperature(agent_id))
    base_thinking = agent_thinking_level(agent_id, config.get("thinking_level"))

    if preferred == "auto":
        providers = AUTO_PROVIDER_ORDER.get(agent_id, ["openai", "anthropic", "gemini", "ollama"])
    else:
        providers = [preferred]

    candidates: List[Dict[str, Any]] = []
    seen = set()
    for provider in providers:
        if provider not in {"openai", "anthropic", "gemini", "ollama"}:
            continue
        if provider != "ollama" and not keys.get(provider):
            continue
        selected_model = normalize_whitespace(config.get("model")) if provider == preferred else ""
        models = [selected_model or DEFAULT_MODELS.get(provider, "")]
        models.extend(VISION_MODEL_FALLBACKS.get(provider, []))
        for model in models:
            model = normalize_whitespace(model)
            if not model:
                continue
            candidate = {
                "provider": provider,
                "model": model,
                "temperature": base_temperature,
                "thinking_level": base_thinking,
            }
            key = (provider, model)
            if key in seen:
                continue
            seen.add(key)
            try:
                if vision_runtime_supported(agent_id, candidate, keys):
                    candidates.append(candidate)
            except Exception:
                continue
    return candidates


def filter_pid_parse(parsed: Dict[str, Any]) -> Dict[str, Any]:
    credible_no_digit_instruments = {
        "FRC", "FIC", "FC", "FI", "FT",
        "PIC", "PC", "PI", "PT",
        "TIC", "TC", "TI", "TT",
        "LIC", "LC", "LI", "LT",
        "PSV", "PRV", "ESD", "SIS",
        "XV", "FV", "PV", "TV", "LV",
    }

    assumed_markers = (
        "assumed",
        "inferred from text",
        "text only",
        "problem statement",
        "not visible",
        "not shown",
        "external",
        "semantic",
        "llm_added_without_image_evidence",
    )

    def item_has_visible_evidence(item: Dict[str, Any]) -> bool:
        evidence_text = " ".join(
            normalize_whitespace(item.get(key))
            for key in (
                "image_evidence",
                "visible_evidence",
                "basis",
                "location",
                "description",
                "label",
                "name",
                "tag",
            )
        )
        if any(marker in evidence_text.lower() for marker in assumed_markers):
            return False
        return bool(evidence_text)

    def generated_id_without_image_evidence(identifier: str, item: Dict[str, Any]) -> bool:
        key = normalize_whitespace(identifier).lower()
        if not key:
            return False
        generated_patterns = [
            r"^stream[-_ ]?\d{1,4}$",
            r"^tk[-_ ]?\d{1,4}$",
            r"^t[-_ ]?\d{2,4}$",
            r"^r[-_ ]?\d{2,4}$",
            r"^l[-_ ]?\d{1,4}$",
            r"^(reactor|product_storage)$",
            r"^[a-z0-9_]+_source$",
        ]
        if not any(re.match(pattern, key, flags=re.I) for pattern in generated_patterns):
            return False
        direct_image_evidence = " ".join(
            normalize_whitespace(item.get(field))
            for field in ("image_evidence", "visible_evidence", "basis")
        ).lower()
        return key not in direct_image_evidence

    def keep_instrument(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if not item_has_visible_evidence(item):
            return False
        tag = normalize_whitespace(item.get("tag")).upper()
        if generated_id_without_image_evidence(tag, item):
            return False
        if not tag or tag == "UNKNOWN":
            return False
        if re.search(r"\d", tag):
            return True
        return tag in credible_no_digit_instruments

    def keep_tagged_item(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if not item_has_visible_evidence(item):
            return False
        tag = normalize_whitespace(item.get("tag")).upper()
        if generated_id_without_image_evidence(tag, item):
            return False
        if tag and tag != "UNKNOWN":
            return True
        return bool(normalize_whitespace(
            item.get("image_evidence")
            or item.get("visible_evidence")
            or item.get("label")
            or item.get("name")
            or item.get("description")
            or item.get("location")
        ))

    def keep_evidence_line(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if not item_has_visible_evidence(item):
            return False
        line_id = normalize_whitespace(item.get("line_id") or item.get("id"))
        if generated_id_without_image_evidence(line_id, item):
            return False
        start = normalize_whitespace(item.get("from") or item.get("source"))
        end = normalize_whitespace(item.get("to") or item.get("destination"))
        service = normalize_whitespace(item.get("material_or_service") or item.get("service"))
        connected = item.get("connected_tags")
        has_connected = isinstance(connected, list) and any(normalize_whitespace(x) for x in connected)
        raw_geometry_id = bool(re.fullmatch(r"[HV]-\d{3,}", line_id or ""))
        if raw_geometry_id and not any([start, end, service, has_connected]):
            return False
        return bool(any([line_id, start, end, service, has_connected]))

    def keep_candidate_node(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if not item_has_visible_evidence(item):
            return False
        node_name = normalize_whitespace(item.get("node_name") or item.get("name")).lower()
        boundary = normalize_whitespace(item.get("boundary") or item.get("boundaries")).lower()
        equipment_tags = item.get("equipment_tags") or item.get("equipment") or []
        instrument_tags = item.get("instrument_tags") or item.get("instruments") or []
        lines = item.get("line_tags") or item.get("lines") or []
        has_specific_tags = any(isinstance(x, str) and re.search(r"\d", x) for x in list(equipment_tags) + list(instrument_tags) + list(lines))
        has_specific_boundary = bool(boundary and boundary not in {"entire p&id", "entire pid", "entire process", "overall process"})
        generic_name = node_name in {"overall process", "process", "entire p&id", "entire pid"}
        return bool(has_specific_tags or (has_specific_boundary and not generic_name))

    if isinstance(parsed.get("equipment"), list):
        parsed["equipment"] = [item for item in parsed["equipment"] if keep_tagged_item(item)]
    for key in ("instruments", "instruments_and_controls"):
        if isinstance(parsed.get(key), list):
            parsed[key] = [item for item in parsed[key] if keep_instrument(item)]
    for key in ("valves", "valves_and_relief_devices"):
        if isinstance(parsed.get(key), list):
            parsed[key] = [item for item in parsed[key] if keep_tagged_item(item) or keep_instrument(item)]
    if isinstance(parsed.get("lines"), list):
        parsed["lines"] = [item for item in parsed["lines"] if keep_evidence_line(item)]
    for key in ("candidate_nodes", "candidate_hazop_nodes"):
        if isinstance(parsed.get(key), list):
            parsed[key] = [item for item in parsed[key] if keep_candidate_node(item)]
    if not any([
        parsed.get("equipment"),
        parsed.get("lines"),
        parsed.get("instruments_and_controls"),
        parsed.get("valves_and_relief_devices"),
        parsed.get("candidate_hazop_nodes"),
        parsed.get("candidate_nodes"),
    ]):
        parsed["parsed_pid_summary"] = (
            normalize_whitespace(parsed.get("parsed_pid_summary"))
            or "The parser did not identify reliable visible equipment, line, instrument, or node evidence from the uploaded P&ID."
        )
    allowed_keys = {
        "parsed_pid_summary",
        "context_used",
        "context_not_used",
        "equipment",
        "lines",
        "process_streams",
        "instruments",
        "instruments_and_controls",
        "valves",
        "valves_and_relief_devices",
        "candidate_nodes",
        "candidate_hazop_nodes",
        "uncertainties",
        "raw_text",
        "parse_source",
        "vision_parse_succeeded",
    }
    return {key: value for key, value in parsed.items() if key in allowed_keys}


def lines_from_process_streams(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    streams = parsed.get("process_streams")
    if not isinstance(streams, list):
        return []
    lines: List[Dict[str, Any]] = []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        start = normalize_whitespace(stream.get("from") or stream.get("source"))
        end = normalize_whitespace(stream.get("to") or stream.get("destination"))
        material = normalize_whitespace(stream.get("material") or stream.get("name") or stream.get("service"))
        if not any([start, end, material]):
            continue
        fallback_line_id = material or "process connection"
        lines.append({
            "line_id": normalize_whitespace(stream.get("line_id")) or fallback_line_id,
            "from": start,
            "to": end,
            "material_or_service": material,
            "flow_direction": normalize_whitespace(stream.get("flow_direction")) or "from source to destination",
        })
    return lines


def split_tags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_whitespace(item) for item in value if normalize_whitespace(item)]
    text = normalize_whitespace(value)
    if not text:
        return []
    return [normalize_whitespace(item) for item in re.split(r"[,;]\s*|\s+\band\b\s+", text, flags=re.I) if normalize_whitespace(item)]


def requested_hazop_parameters(documents: Dict[str, Any]) -> Optional[List[str]]:
    text = " ".join(
        normalize_whitespace(value).lower()
        for value in documents.values()
        if isinstance(value, str)
    )
    parameter_aliases = {
        "flow": ["flow", "rate", "feed rate", "throughput"],
        "pressure": ["pressure", "overpressure", "vacuum"],
        "temperature": ["temperature", "temp", "heating", "cooling"],
        "level": ["level", "inventory"],
        "composition": ["composition", "concentration", "contamination", "purity", "wrong material"],
    }
    scope_phrases = [
        r"\bonly\b.{0,80}\b(?:parameter|parameters|condition|conditions|case|scenario|mode)\b",
        r"\b(?:parameter|parameters)\b.{0,80}\bonly\b",
        r"\bconsider\b.{0,30}\bonly\b.{0,80}",
        r"\bstudy\b.{0,30}\bonly\b.{0,80}",
        r"\bperform\b.{0,30}\bonly\b.{0,80}",
        r"\blimit(?:ed)?\b.{0,80}\b(?:to|for)\b.{0,80}",
    ]
    scoped_text = " ".join(match.group(0) for pattern in scope_phrases for match in re.finditer(pattern, text))
    if not scoped_text:
        return None
    found = []
    for parameter, aliases in parameter_aliases.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", scoped_text) for alias in aliases):
            found.append(parameter)
    return found or None


def canonical_scope_parameter(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    aliases = {
        "flow": "flow",
        "rate": "flow",
        "flowrate": "flow",
        "level": "level",
        "pressure": "pressure",
        "temperature": "temperature",
        "temp": "temperature",
        "contamination": "contamination",
        "composition": "composition",
        "concentration": "composition",
        "purity": "composition",
    }
    return aliases.get(text, text)


def requested_guideword_scope(documents: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    text = " ".join(
        normalize_whitespace(value)
        for value in documents.values()
        if isinstance(value, str)
    )
    if not re.search(r"\bguide\s*words?\b|\bguidewords?\b", text, flags=re.I):
        return {}
    text = text.replace("–", "-").replace("—", "-")
    parameter_pattern = re.compile(
        r"\b(FLOW|LEVEL|PRESSURE|TEMPERATURE|TEMP|CONTAMINATION|COMPOSITION|CONCENTRATION|PURITY)\b\s*[-:]\s*",
        flags=re.I,
    )
    matches = list(parameter_pattern.finditer(text))
    if not matches:
        return {}

    guideword_pattern = re.compile(
        r"\b(HIGH|LOW|ZERO|REVERSE|NO|LESS|MORE|PART\s+OF|AS\s+WELL\s+AS|OTHER\s+THAN|CONTAMINATION)\b",
        flags=re.I,
    )
    scoped: Dict[str, List[Dict[str, str]]] = {}
    for index, match in enumerate(matches):
        parameter = canonical_scope_parameter(match.group(1))
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else min(len(text), match.end() + 260)
        segment = text[match.end():segment_end]
        found: List[Dict[str, str]] = []
        seen: Set[str] = set()
        for token_match in guideword_pattern.finditer(segment):
            guideword = normalized_guideword(token_match.group(1))
            key = guideword.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({
                "guideword": guideword,
                "meaning": guideword_meaning(guideword, parameter),
            })
        if parameter == "contamination" and not found:
            found.append({
                "guideword": "Contamination",
                "meaning": guideword_meaning("Contamination", parameter),
            })
        if found:
            scoped[parameter] = found
    return scoped


def supplied_document_text(documents: Dict[str, Any]) -> str:
    """Combine user-supplied text without vision-parser narrative fields."""
    parts: List[str] = []
    seen: Set[str] = set()
    for key in (
        "hazop_setup",
        "process_description",
        "safeguards",
        "operating_conditions",
        "pid",
    ):
        value = documents.get(key)
        if isinstance(value, str) and value.strip():
            fingerprint = hashlib.sha256(normalize_whitespace(value).encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            parts.append(value)
    return "\n\n".join(parts)


def combined_document_text(documents: Dict[str, Any]) -> str:
    parts = [supplied_document_text(documents)]
    parser_output = documents.get("pid_image_parser_output")
    if isinstance(parser_output, str) and parser_output.strip():
        parts.append(parser_output)
    return "\n\n".join(part for part in parts if part)


def parse_manual_worksheet_columns(text: str) -> List[str]:
    match = re.search(r"\bHAZOP\s+WORKSHEETS?\b\s*(.+)$", text, flags=re.I | re.S)
    if not match:
        return []
    raw_lines = [
        normalize_whitespace(line).strip(" •-*:\t")
        for line in match.group(1).splitlines()
    ]
    aliases = {
        "guidewords": "Guideword",
        "guideword": "Guideword",
        "effects": "Deviation / Effect",
        "effect": "Deviation / Effect",
        "deviation": "Deviation / Effect",
        "cause": "Potential Cause",
        "potential cause": "Potential Cause",
        "consequence": "Consequence",
        "safegaurds": "Safeguard",
        "safeguards": "Safeguard",
        "safeguard": "Safeguard",
        "actions": "Necessary Action",
        "action": "Necessary Action",
        "necessary action": "Necessary Action",
    }
    columns: List[str] = []
    seen: Set[str] = set()
    for line in raw_lines:
        key = line.lower()
        if key in aliases and aliases[key].lower() not in seen:
            columns.append(aliases[key])
            seen.add(aliases[key].lower())
    return columns


def procedural_parameter_for_guideword(category_label: str, option: str = "") -> str:
    text = normalize_whitespace(f"{category_label} {option}").lower()
    if "communication" in text:
        return "communication"
    if "pressure" in text:
        return "pressure"
    if "reverse" in text:
        return "flow"
    if "rate" in text or "too fast" in text or "too slow" in text:
        return "rate"
    if "magnitude" in text or re.search(r"\b(more|less)\b", text):
        return "magnitude"
    if "out of sequence" in text or "too early" in text or "too late" in text:
        return "sequence"
    if "incomplete" in text:
        return "completion"
    if "contamination" in text:
        return "contamination"
    return "other"


def procedural_guideword_label(category_label: str, option: str = "") -> str:
    guideword = normalized_guideword_no_lookup(category_label)
    qualifier = normalize_whitespace(option)
    if not qualifier:
        return guideword
    if guideword.lower() in {"other", "others"}:
        return guideword
    if guideword.lower() in qualifier.lower() or qualifier.lower() in guideword.lower():
        return guideword
    return f"{guideword} ({qualifier})"


def parse_manual_guideword_entries(text: str) -> List[Dict[str, str]]:
    match = re.search(
        r"\bGuide\s+words?\s*:\s*(.+?)(?:\n\s*\n|\bHAZOP\s+WORKSHEETS?\b|$)",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return []
    block = match.group(1).replace("—", "-").replace("–", "-")
    entries: List[Dict[str, str]] = []
    seen: Set[str] = set()
    procedural_mode = bool(re.search(r"\b(?:Procedural\s+HAZOP|DETAILED\s+PROPOSED\s+SEQUENCE)\b", text, flags=re.I))
    for raw_line in block.splitlines():
        line = normalize_whitespace(raw_line).strip(" •-*:\t")
        if not line:
            continue
        if re.search(r"\bguide\s+words?\b.*\b(?:used|were)\b", line, flags=re.I):
            continue
        if "-" in line:
            category, options_text = [part.strip() for part in line.split("-", 1)]
            category_label = normalized_guideword_no_lookup(category)
            parameter = canonical_scope_parameter(category)
            standard_parameter_category = category.strip().lower() in {
                "flow", "level", "pressure", "temperature", "temp", "contamination", "composition", "concentration", "purity"
            }
            is_process_parameter = parameter in {
                "flow", "level", "pressure", "temperature", "contamination", "composition"
            } and standard_parameter_category and not procedural_mode
            if parameter == "contamination" and standard_parameter_category:
                guideword = "Contamination"
                key = f"{parameter}:{guideword}".lower()
                if key not in seen:
                    entries.append({
                        "guideword": guideword,
                        "parameter": parameter,
                        "meaning": normalize_whitespace(options_text) or guideword_meaning(guideword, parameter),
                        "deviation_hint": "Contamination",
                        "expected_deviation": "Contamination",
                        "effect": "Contamination",
                        "matrix_key": f"{parameter}:{guideword}:Contamination",
                    })
                    seen.add(key)
                continue
            options = [normalize_whitespace(item) for item in re.split(r",|/|\bor\b", options_text, flags=re.I)]
            for option in options:
                if not option:
                    continue
                if is_process_parameter:
                    guideword = normalized_guideword(option)
                    entry_parameter = parameter
                    meaning = guideword_meaning(guideword, entry_parameter)
                    deviation_hint = f"{guideword} {entry_parameter}"
                else:
                    entry_parameter = procedural_parameter_for_guideword(category_label, option)
                    deviation_hint = normalize_whitespace(option)
                    if category_label.lower() == "reverse" and re.search(r"\bincomplete\b", deviation_hint, flags=re.I):
                        deviation_hint = normalize_whitespace(re.sub(r"\bincomplete\b.*$", "", deviation_hint, flags=re.I))
                    guideword = procedural_guideword_label(category_label, deviation_hint)
                    meaning = "Apply this procedural guideword item to the supplied node design intention."
                key = f"{entry_parameter}:{guideword}:{deviation_hint}".lower()
                if key not in seen:
                    entry = {
                        "guideword": guideword,
                        "guideword_category": category_label,
                        "parameter": entry_parameter,
                        "meaning": meaning,
                        "qualifier": deviation_hint,
                        "deviation_hint": deviation_hint,
                        "matrix_key": key,
                    }
                    if is_process_parameter:
                        entry["expected_deviation"] = deviation_hint
                        entry["effect"] = deviation_hint
                    else:
                        entry["expected_deviation"] = ""
                        entry["effect"] = ""
                    entries.append(entry)
                    seen.add(key)
        else:
            guideword = normalize_whitespace(line)
            parameter = procedural_parameter_for_guideword(guideword)
            key = f"{parameter}:{guideword}".lower()
            if key not in seen:
                entries.append({
                    "guideword": guideword,
                    "parameter": parameter,
                    "meaning": f"Procedural HAZOP guideword from the supplied study setup: {guideword}.",
                    "matrix_key": key,
                })
                seen.add(key)
    return entries


def manual_study_focus_from_matrix(guideword_matrix: List[Dict[str, str]]) -> List[str]:
    focus: List[str] = []
    seen: Set[str] = set()

    def add(label: str) -> None:
        label = normalize_whitespace(label)
        key = label.lower()
        if label and key not in seen:
            seen.add(key)
            focus.append(label)

    for item in guideword_matrix:
        if not isinstance(item, dict):
            continue
        parameter = normalize_whitespace(item.get("parameter")).lower()
        guideword = normalized_guideword_no_lookup(item.get("guideword"))
        effect = normalize_whitespace(
            item.get("chair_deviation")
            or item.get("effect")
            or item.get("expected_deviation")
            or item.get("deviation_hint")
        )
        text = normalize_whitespace(" ".join([parameter, guideword, effect])).lower()
        if parameter and parameter != "procedure":
            add(parameter)
            continue
        if "out of sequence" in text or "too early" in text or "too late" in text:
            add("sequence / timing")
        elif "reverse" in text:
            add("reverse flow")
        elif "rate" in text or "flow" in text or "too fast" in text or "too slow" in text or "magnitude" in text:
            add("flow / rate")
        elif "pressure" in text:
            add("pressure")
        elif "communication" in text:
            add("communication")
        elif "incomplete" in text:
            add("incomplete execution")
        elif "contamination" in text:
            add("contamination")
        elif "others" in text or "other" in text:
            add("other deviations")
        else:
            add(guideword or effect or "other deviations")
    return focus


def parse_numbered_procedure_steps(text: str) -> Dict[str, Any]:
    match = re.search(
        r"\bDETAILED\s+PROPOSED\s+SEQUENCE\b\s*(.+?)(?:\bDivision\s+into\s+nodes\s*:|$)",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return {"initial_setup": "", "steps": []}
    section = match.group(1)
    first_step = re.search(r"(?m)^\s*1\.\s+", section)
    initial_setup = normalize_whitespace(section[: first_step.start()]) if first_step else normalize_whitespace(section)
    steps: List[Dict[str, Any]] = []
    for step_match in re.finditer(
        r"(?ms)^\s*(\d{1,3})\.\s*(.+?)(?=^\s*\d{1,3}\.\s*|\Z)",
        section,
    ):
        number = int(step_match.group(1))
        instruction = normalize_whitespace(step_match.group(2))
        if instruction:
            steps.append({"step_number": number, "instruction": instruction})
    return {"initial_setup": initial_setup, "steps": steps}


def procedure_steps_for_node(step_label: str, procedure_steps: List[Dict[str, Any]]) -> List[str]:
    label = step_label or ""
    numbers = [int(value) for value in re.findall(r"\d+", label)]
    if not numbers:
        return []
    # Only treat this as an inclusive range when there are exactly two
    # numbers joined by an explicit range separator (e.g. "Steps 1-2" or
    # "Steps 1 to 2"). Previously any label with 2+ numbers used just the
    # first two as a range boundary, so "Steps 1, 3, 5" silently became
    # {1,2,3} -- wrongly pulling in step 2 (never requested) and dropping
    # step 5 (explicitly requested), corrupting which procedure content this
    # node sees.
    if len(numbers) == 2 and re.search(r"\d\s*(?:-|–|to)\s*\d", label, flags=re.IGNORECASE):
        start, end = min(numbers), max(numbers)
        wanted = set(range(start, end + 1))
    else:
        wanted = set(numbers)
    selected: List[str] = []
    for item in procedure_steps:
        step_number = item.get("step_number")
        instruction = normalize_whitespace(item.get("instruction"))
        if step_number in wanted and instruction:
            selected.append(f"{step_number}. {instruction}")
    return selected


def text_between_markers(text: str, start_pattern: str, end_patterns: List[str]) -> str:
    start = re.search(start_pattern, text, flags=re.I)
    if not start:
        return ""
    end_index = len(text)
    for pattern in end_patterns:
        end = re.search(pattern, text[start.end():], flags=re.I)
        if end:
            end_index = min(end_index, start.end() + end.start())
    return text[start.end():end_index].strip()


def parse_manual_context(text: str) -> Dict[str, Any]:
    background_end = re.search(r"\bDETAILED\s+PROPOSED\s+SEQUENCE\b", text, flags=re.I)
    background = text[: background_end.start()].strip() if background_end else text.strip()
    sequence = text_between_markers(
        text,
        r"\bDETAILED\s+PROPOSED\s+SEQUENCE\b",
        [r"\bDivision\s+into\s+nodes\s*:", r"\bGuide\s+words?\s*:", r"\bHAZOP\s+WORKSHEETS?\b"],
    )
    node_definition = text_between_markers(
        text,
        r"\bDivision\s+into\s+nodes\s*:",
        [r"\bGuide\s+words?\s*:", r"\bHAZOP\s+WORKSHEETS?\b"],
    )
    guideword_block = text_between_markers(
        text,
        r"\bGuide\s+words?\s*:",
        [r"\bHAZOP\s+WORKSHEETS?\b"],
    )
    worksheet_block = text_between_markers(
        text,
        r"\bHAZOP\s+WORKSHEETS?\b\s*:?",
        [],
    )
    return {
        "full_manual_text": clamp_text(text, MAX_CONTEXT_CHARS),
        "background_and_process_description": normalize_whitespace(background),
        "detailed_proposed_sequence": normalize_whitespace(sequence),
        "node_definition_section": normalize_whitespace(node_definition),
        "guideword_section": normalize_whitespace(guideword_block),
        "worksheet_heading_section": normalize_whitespace(worksheet_block),
    }


def split_complete_statements(text: str, limit: int = 14, max_chars: int = 420) -> List[str]:
    """Split supplied study text into complete, readable statements for the Study Basis tab."""
    cleaned = normalize_whitespace(text)
    cleaned = re.sub(r"^(?:BACKGROUND|PROCESS\s+DESCRIPTION|DESCRIPTION\s+OF\s+THE\s+FACILITY)\s*:?\s*", "", cleaned, flags=re.I)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    statements: List[str] = []
    for part in parts:
        item = normalize_whitespace(part)
        if not item:
            continue
        if len(item) > max_chars:
            chunks = re.split(r";\s+|,\s+(?=(?:and|with|which|where|including|before|after)\b)", item)
            current = ""
            for chunk in chunks:
                chunk = normalize_whitespace(chunk)
                if not chunk:
                    continue
                candidate = normalize_whitespace(f"{current}; {chunk}" if current else chunk)
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        statements.append(current.rstrip(";,"))
                    current = chunk
            if current:
                statements.append(current.rstrip(";,"))
        else:
            statements.append(item)
        if len(statements) >= limit:
            break
    return statements[:limit]


def parse_manual_hazop_setup(documents: Dict[str, Any]) -> Dict[str, Any]:
    text = combined_document_text(documents)
    if not re.search(r"\bDivision\s+into\s+nodes\s*:", text, flags=re.I):
        return {"nodes": [], "guideword_matrix": [], "worksheet_columns": []}

    manual_context = parse_manual_context(text)
    guideword_matrix = parse_manual_guideword_entries(text)
    worksheet_columns = parse_manual_worksheet_columns(text)
    known_safeguards = extract_known_safeguards_from_text(
        supplied_document_text(documents),
        "process_description_procedure_and_drawing_text",
    )
    procedure = parse_numbered_procedure_steps(text)
    procedure_initial_setup = normalize_whitespace(procedure.get("initial_setup", ""))
    procedure_steps = procedure.get("steps", [])
    division_match = re.search(r"\bDivision\s+into\s+nodes\s*:\s*(.+)$", text, flags=re.I | re.S)
    if not division_match:
        return {"nodes": [], "guideword_matrix": guideword_matrix, "worksheet_columns": worksheet_columns}
    node_section = division_match.group(1)
    node_section = re.split(r"\bGuide\s+words?\s*:", node_section, flags=re.I, maxsplit=1)[0]
    node_section = re.split(r"\bHAZOP\s+WORKSHEETS?\b", node_section, flags=re.I, maxsplit=1)[0]
    node_matches = list(re.finditer(r"(?im)^\s*Node\s+(\d+)(?:\s*:\s*(.+?))?\s*$", node_section))
    nodes: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for index, match in enumerate(node_matches):
        node_number = int(match.group(1))
        block_start = match.end()
        block_end = node_matches[index + 1].start() if index + 1 < len(node_matches) else len(node_section)
        block = node_section[block_start:block_end].strip()
        heading_detail = normalize_whitespace(match.group(2) or "")
        if "not completed" in normalize_whitespace(f"{heading_detail} {block}").lower():
            warnings.append(f"N-{node_number:03d} is marked not completed in the supplied manual setup.")
            continue
        bullet_lines = [
            normalize_whitespace(line).strip(" •-*:\t")
            for line in block.splitlines()
            if normalize_whitespace(line).strip(" •-*:\t")
        ]
        step_line = next((line for line in bullet_lines if re.match(r"Steps?\s+\d+", line, flags=re.I)), "")
        design_line = next((line for line in bullet_lines if re.match(r"Design\s+intention\s*:", line, flags=re.I)), "")
        other_lines = [line for line in bullet_lines if line not in {step_line, design_line}]

        steps = ""
        node_name = heading_detail or f"Node {node_number}"
        boundaries = heading_detail
        node_description = heading_detail
        if step_line:
            step_match = re.match(r"(Steps?\s+[^:]+):\s*(.+)", step_line, flags=re.I)
            if step_match:
                steps = normalize_whitespace(step_match.group(1))
                node_description = normalize_whitespace(step_match.group(2))
            else:
                node_description = step_line
            valve_tags = re.findall(r"\bV\d+\b", node_description, flags=re.I)
            tags = valve_tags or re.findall(r"\bV\d+\b|\bJ\d+\b|\bLCV\b|\bIST\b|\bN2\b", node_description, flags=re.I)
            if len(tags) >= 2:
                boundaries = f"{tags[0].upper()} to {tags[-1].upper()}"
            node_name = re.sub(
                r"\s*\.?\s+(?:\b(?:V\d+|J\d+|LCV|N2)\b\s*(?:-|to)?\s*){1,4}\.?$",
                "",
                node_description,
                flags=re.I,
            )
            node_name = normalize_whitespace(node_name).strip(" .-")
        elif other_lines:
            if heading_detail:
                node_description = normalize_whitespace(f"{heading_detail}: {other_lines[0]}")
                node_name = normalize_whitespace(other_lines[0])
            else:
                node_description = other_lines[0]
                node_name = node_description

        design_intent = re.sub(r"^Design\s+intention\s*:\s*", "", design_line, flags=re.I).strip()
        if not design_intent:
            design_intent = "Not completed in supplied manual node list." if "not completed" in block.lower() else node_description

        equipment = sorted(set(tag.upper() for tag in re.findall(r"\bV\d+\b|\bJ\d+\b|\bLCV\b|\bIST\b|\bN2\b", block, flags=re.I)))
        if heading_detail:
            equipment = sorted(set(equipment + re.findall(r"\b[A-Z]{1,3}\d+\b|\bL\d+\b|\bVO\b|\bV0\b|\bLIC\b|\bTIC\b|\bFIC\b", heading_detail + " " + block, flags=re.I)))
        manual_parameters: List[str] = []
        for item in guideword_matrix:
            if not isinstance(item, dict):
                continue
            parameter = normalize_whitespace(item.get("parameter")).lower()
            if parameter and parameter not in manual_parameters:
                manual_parameters.append(parameter)
        if not manual_parameters:
            manual_parameters = ["other"]
        raw_node = {
            "node_id": f"N-{node_number:03d}",
            "node_name": node_name or f"Node {node_number}",
            "node_description": node_description or block,
            "boundaries": boundaries,
            "design_intent": normalize_whitespace(design_intent),
            "equipment": equipment,
            "lines": sanitize_list(re.findall(r"\bV\d+\s*(?:to|-)\s*V\d+\b", block, flags=re.I), 20),
            "materials": sanitize_list(re.findall(r"\bC6\b|\bN2\b|nitrogen|petroleum feedstock", block, flags=re.I), 20),
            "parameters": manual_parameters,
            "study_focus": manual_study_focus_from_matrix(guideword_matrix) or manual_parameters,
            "steps": steps,
            "procedure_initial_setup": procedure_initial_setup,
            "procedure_steps": procedure_steps_for_node(steps, procedure_steps),
            "manual_context": manual_context,
            "guideword_matrix": list(guideword_matrix),
            # Safeguards are node-scoped. A control mentioned elsewhere in the
            # study must not be propagated to every node.
            "known_safeguards": extract_known_safeguards_from_text(
                "\n".join([block, node_description, normalize_whitespace(design_intent)]),
                "node_definition",
            ),
            "manual_source": "Division into nodes and guide words supplied by user.",
            "worksheet_columns": worksheet_columns,
        }
        nodes.append(sanitize_node(raw_node, len(nodes) + 1))

    return {
        "nodes": nodes,
        "guideword_matrix": guideword_matrix,
        "worksheet_columns": worksheet_columns,
        "manual_context": manual_context,
        "known_safeguards": known_safeguards,
        "warnings": warnings,
        "source": "manual_hazop_setup",
    }


def requested_study_conditions(documents: Dict[str, Any]) -> List[str]:
    text = " ".join(
        normalize_whitespace(value)
        for value in documents.values()
        if isinstance(value, str)
    )
    condition_patterns = [
        r"\b(?:only|specifically|focus(?:ed)? on|limited to|during|under|for)\s+([^.;\n]{3,140})",
        r"\b(?:startup|start-up|shutdown|shut-down|normal operation|abnormal operation|maintenance|commissioning|decommissioning|emergency|blocked outlet|loss of cooling|loss of power|high pressure|low pressure|high temperature|low temperature)\b",
    ]
    conditions: List[str] = []
    for pattern in condition_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            value = normalize_whitespace(match.group(1) if match.groups() else match.group(0))
            value = re.sub(r"\b(the|this|that)\b\s*", "", value, flags=re.I).strip(" ,.;:")
            if re.search(r"\b(parameter|parameters|deviation|deviations)\b", value, flags=re.I):
                continue
            if value and value.lower() not in {c.lower() for c in conditions}:
                conditions.append(value)
            if len(conditions) >= 8:
                return conditions
    return conditions


def study_scope_summary(documents: Dict[str, Any]) -> Dict[str, Any]:
    parameters = requested_hazop_parameters(documents)
    guideword_scope = requested_guideword_scope(documents)
    conditions = requested_study_conditions(documents)
    manual_setup = parse_manual_hazop_setup(documents)
    notes = []
    if manual_setup.get("nodes"):
        notes.append(
            f"Manual HAZOP setup detected: {len(manual_setup['nodes'])} supplied node(s) "
            f"and {len(manual_setup.get('guideword_matrix') or [])} supplied procedural guideword item(s)."
        )
    elif guideword_scope:
        scoped_params = ", ".join(guideword_scope.keys())
        notes.append(f"Study-specific guideword scope detected for: {scoped_params}.")
    elif parameters:
        notes.append(f"Parameter scope requested by user/problem statement: {', '.join(parameters)}.")
    if conditions:
        notes.append(f"Specific condition/scenario focus: {'; '.join(conditions)}.")
    if not notes:
        notes.append("No explicit parameter or condition restriction detected; perform full HAZOP coverage for relevant parameters.")
    return {"parameters": parameters or [], "guideword_scope": guideword_scope, "conditions": conditions, "notes": notes}


def node_text_for_scope(node: Dict[str, Any]) -> str:
    return normalize_whitespace(" ".join(
        str(node.get(key, ""))
        for key in ("node_id", "node_name", "node_description", "boundaries", "design_intent", "operating_conditions")
    ) + " " + " ".join(node.get("equipment", []) or []) + " " + " ".join(node.get("lines", []) or [])).lower()


def parameter_relevant_to_node(parameter: str, node: Dict[str, Any]) -> bool:
    inferred = set(node_parameters(node))
    if parameter in inferred:
        return True
    text = node_text_for_scope(node)
    if parameter == "contamination":
        return any(term in text for term in ("contamin", "hot oil", "heating fluid", "surge", "moisture", "h1", "h2", "p1"))
    if parameter == "flow":
        return any(term in text for term in ("line", "feed", "pump", "valve", "transfer", "flow", "inlet", "outlet"))
    if parameter == "temperature":
        return any(term in text for term in ("heat", "heater", "furnace", "reboil", "condenser", "cool", "temperature", "tic"))
    if parameter == "level":
        return any(term in text for term in ("tank", "vessel", "drum", "receiver", "column", "level", "lic"))
    if parameter == "pressure":
        return any(term in text for term in ("pressure", "relief", "vent", "psv", "prv", "column", "vessel", "condenser"))
    return False


def apply_study_scope(
    nodes: List[Dict[str, Any]],
    parameters: Optional[List[str]],
    guideword_scope: Optional[Dict[str, List[Dict[str, str]]]],
) -> List[Dict[str, Any]]:
    guideword_scope = guideword_scope or {}
    if not parameters and not guideword_scope:
        return nodes
    scoped = []
    for node in nodes:
        updated = dict(node)
        if parameters:
            selected_parameters = list(parameters)
        elif guideword_scope:
            selected_parameters = [
                parameter for parameter in guideword_scope
                if parameter_relevant_to_node(parameter, updated)
            ]
            if not selected_parameters:
                selected_parameters = list(guideword_scope.keys())
        else:
            selected_parameters = node_parameters(updated)
        updated["parameters"] = selected_parameters
        if guideword_scope:
            updated["guidewords_by_parameter"] = {
                parameter: guideword_scope[parameter]
                for parameter in selected_parameters
                if parameter in guideword_scope
            }
            updated["guideword_matrix"] = [
                {
                    "parameter": parameter,
                    "guideword": entry["guideword"],
                    "meaning": entry.get("meaning") or guideword_meaning(entry["guideword"], parameter),
                }
                for parameter in selected_parameters
                for entry in guideword_scope.get(parameter, [])
            ]
        scoped.append(updated)
    return scoped


def normalize_pid_node_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    valve_re = re.compile(r"^V[- ]?\d{1,3}[A-Z]?$", re.I)
    for key in ("candidate_nodes", "candidate_hazop_nodes"):
        nodes = parsed.get(key)
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            instrument_tags = node.get("instrument_tags")
            if not isinstance(instrument_tags, list):
                continue
            valves = [tag for tag in instrument_tags if isinstance(tag, str) and valve_re.match(tag.strip())]
            if not valves:
                continue
            node["instrument_tags"] = [tag for tag in instrument_tags if tag not in valves]
            existing_valves = [
                str(tag) for tag in (node.get("valve_tags") or [])
                if isinstance(tag, str)
            ]
            node["valve_tags"] = sorted(set(existing_valves + valves))
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Document context
# ─────────────────────────────────────────────────────────────────────────────

def document_context(documents: Dict[str, str]) -> str:
    seen: Set[str] = set()

    def unique_text(key: str) -> str:
        value = str(documents.get(key, "") or "")
        if not value.strip():
            return ""
        fingerprint = hashlib.sha256(normalize_whitespace(value).encode("utf-8")).hexdigest()
        if fingerprint in seen:
            return ""
        seen.add(fingerprint)
        return value

    payload = {
        "hazop_setup_manual_nodes_and_guidewords": unique_text("hazop_setup"),
        "pid_or_drawing_text": unique_text("pid"),
        "pid_image_count": len(documents.get("pid_images", [])) if isinstance(documents.get("pid_images"), list) else 0,
        "pid_image_parser_output": unique_text("pid_image_parser_output"),
        "shared_process_understanding": unique_text("process_understanding"),
        "process_description": unique_text("process_description"),
        "operating_conditions": unique_text("operating_conditions"),
        "safeguards_controls_procedures": unique_text("safeguards"),
        "study_scope": study_scope_summary(documents),
    }
    return clamp_text(json.dumps(payload, ensure_ascii=True, indent=2), MAX_CONTEXT_CHARS)


def round_three_evidence_manifest(
    node: Dict[str, Any],
    documents: Dict[str, Any],
) -> Dict[str, Any]:
    """Record exactly which case-evidence categories were available to the Chair."""
    source_keys = (
        "pid",
        "pid_image_parser_output",
        "process_understanding",
        "process_description",
        "operating_conditions",
        "safeguards",
        "hazop_setup",
    )
    sources: Dict[str, Any] = {}
    for key in source_keys:
        value = documents.get(key, "")
        text = json.dumps(value, ensure_ascii=True) if not isinstance(value, str) else value
        sources[key] = {
            "available": bool(normalize_whitespace(text)),
            "character_count": len(text),
            "sha256_16": short_hash(text) if text else "",
        }
    return {
        "node": {
            key: json_safe(node.get(key))
            for key in (
                "node_id",
                "node_name",
                "node_description",
                "boundaries",
                "design_intent",
                "operating_conditions",
                "equipment",
                "lines",
                "instrument_tags",
                "procedure_initial_setup",
                "procedure_steps",
            )
        },
        "source_manifest": sources,
        "evidence_hierarchy": [
            "node boundary and design intent",
            "current-case P&ID parse and shared process understanding",
            "current-case process/procedure/operating/safeguard evidence",
            "Round 2B rubric evidence and technical arguments",
        ],
        "generic_knowledge_rule": (
            "Generic engineering knowledge may test plausibility but cannot create an undocumented case fact."
        ),
    }


configure_prompt_helpers(
    agents=AGENTS,
    compact_json_fn=compact_json,
    clamp_text_fn=clamp_text,
    study_scope_summary_fn=study_scope_summary,
    document_context_fn=document_context,
    guideword_matrix_text_fn=guideword_matrix_text,
    guideword_parameter_matrix_fn=guideword_parameter_matrix,
    node_scope_gate_text_fn=node_scope_gate_text,
    normalize_whitespace_fn=normalize_whitespace,
)


# ─────────────────────────────────────────────────────────────────────────────
# Job state management
# ─────────────────────────────────────────────────────────────────────────────

def initial_result() -> Dict[str, Any]:
    return {
        "report_inputs": {},
        "performance": {
            "model_calls": [],
            "row_lineage": [],
            "experiment_metadata": {},
        },
        "nodes": [],
        "pid_parse": None,
        "process_understanding": {},
        "planning": {},
        "review_methodology": json_safe(ROUND_TWO_DECISION_METHOD),
        "chair_finalization_methodology": json_safe(ROUND_THREE_DECISION_METHOD),
        "node_results": [],
        "worksheet": [],
        "worksheet_display": [],
        "worksheet_atomic": [],
        "review_decision_trace": [],
        "round_3_adjudication_trace": [],
        "round_3_adjudication_summary": {},
        "round_3_escalated_proposals": [],
        "round_3_new_proposal_deliberations": [],
        "action_register": [],
        "report": None,
    }


def report_input_snapshot(documents: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pid_images": list(documents.get("pid_images", [])[:2]) if isinstance(documents.get("pid_images"), list) else [],
        "pid_text": clamp_text(documents.get("pid", ""), 20000),
        "hazop_setup": clamp_text(documents.get("hazop_setup", ""), 30000),
        "process_description": clamp_text(documents.get("process_description", ""), 30000),
        "safeguards": clamp_text(documents.get("safeguards", ""), 20000),
    }


def effective_agent_settings_snapshot(agents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for agent_id in AGENT_IDS:
        config = agents.get(agent_id) or {}
        configured_temperature = clamp_temperature(
            config.get("temperature"),
            default_agent_temperature(agent_id),
        )
        effective_temperature = 0.0 if agent_id == "pid_image_parser" else configured_temperature
        snapshot[agent_id] = {
            "provider": normalize_whitespace(config.get("provider")),
            "model": normalize_whitespace(config.get("model")),
            "configured_temperature": configured_temperature,
            "effective_temperature": effective_temperature,
            "thinking_level": agent_thinking_level(agent_id, config.get("thinking_level")),
        }
    return snapshot


def create_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = initial_result()
    result["report_inputs"] = report_input_snapshot(payload["documents"])
    return {
        "id": uuid.uuid4().hex,
        "status": "queued",
        "stop_requested": False,
        "submitted_at": utc_now(),
        "updated_at": utc_now(),
        # Incremented only when user-facing result data changes. Progress,
        # event, and model-call telemetry updates do not force the browser to
        # download the complete (potentially very large) result document.
        "result_revision": 0,
        "error": "",
        "steps": [],
        "events": [],
        "progress": {"phase": "queued", "percent": 0, "message": "Queued", "current_node": "", "current_round": ""},
        "result": result,
        "request": {
            "document_lengths": {
                key: (len(value) if not isinstance(value, list) else len(value))
                for key, value in payload["documents"].items()
            },
            "agents": payload["agents"],
            "effective_agent_settings": effective_agent_settings_snapshot(payload["agents"]),
            "settings": payload["settings"],
        },
    }


def job_snapshot_for_performance(job: Dict[str, Any], reason: str) -> Dict[str, Any]:
    snapshot = json_safe(job)
    result = snapshot.get("result") if isinstance(snapshot.get("result"), dict) else {}

    # Keep one canonical copy of each audit dataset in the persisted JSON.
    # The live job retains convenience projections for the web/Word renderers,
    # but storing those deterministic aliases again makes large Round 2 runs
    # unnecessarily difficult to inspect and compare.
    worksheet = result.get("worksheet")
    if result.get("worksheet_display") == worksheet:
        result.pop("worksheet_display", None)

    node_results = [
        item for item in result.get("node_results", [])
        if isinstance(item, dict)
    ]
    canonical_trace_pairs = (
        ("review_decision_trace", "round_2_review_trace"),
        ("round_3_adjudication_trace", "round_3_adjudication_trace"),
        ("round_3_escalated_proposals", "round_3_escalated_proposals"),
        (
            "round_3_new_proposal_deliberations",
            "round_3_new_proposal_deliberations",
        ),
    )
    removed_node_trace_fields: List[str] = []
    for global_key, node_key in canonical_trace_pairs:
        flattened = [
            entry
            for node_result in node_results
            for entry in (node_result.get(node_key) or [])
            if isinstance(entry, dict)
        ]
        if result.get(global_key) == flattened:
            for node_result in node_results:
                node_result.pop(node_key, None)
            removed_node_trace_fields.append(node_key)

    round_two_method = result.get("review_methodology")
    round_three_method = result.get("chair_finalization_methodology")
    for node_result in node_results:
        if node_result.get("round_2_review_methodology") == round_two_method:
            node_result.pop("round_2_review_methodology", None)
        if node_result.get("round_3_chair_methodology") == round_three_method:
            node_result.pop("round_3_chair_methodology", None)

        for agent_output in node_result.get("round_2", []) or []:
            if not isinstance(agent_output, dict):
                continue
            for review_field in ("initial_scenario_reviews", "scenario_reviews"):
                for review in agent_output.get(review_field, []) or []:
                    if isinstance(review, dict):
                        # Source scenario content is canonical in
                        # round_1_review_universe and linked by review_id.
                        review.pop("reviewed_scenario", None)
            # These are deterministic projections of scenario_reviews or are
            # copied into the canonical new-scenario deliberation records.
            for derived_key in (
                "agreements",
                "scenario_challenges",
                "discussion_refinements",
                "new_scenarios",
                "new_scenario_reviews",
            ):
                agent_output.pop(derived_key, None)
            for empty_legacy_key in (
                "safeguard_challenges", "challenges", "agreement_notes"
            ):
                if not agent_output.get(empty_legacy_key):
                    agent_output.pop(empty_legacy_key, None)

    result["storage_schema"] = {
        "version": "haznova-performance-v2-canonical",
        "canonical_final_worksheet": "result.worksheet",
        "canonical_atomic_worksheet": "result.worksheet_atomic",
        "canonical_round_2_trace": "result.review_decision_trace",
        "canonical_round_3_trace": "result.round_3_adjudication_trace",
        "canonical_round_3_escalated_proposals": "result.round_3_escalated_proposals",
        "canonical_round_3_new_proposal_deliberations": (
            "result.round_3_new_proposal_deliberations"
        ),
        "removed_exact_or_reconstructable_aliases": [
            "result.worksheet_display when equal to result.worksheet",
            *[f"result.node_results[].{field}" for field in removed_node_trace_fields],
            "result.node_results[].round_2[].reviewed_scenario",
            "result.node_results[].round_2[].agreements",
            "result.node_results[].round_2[].scenario_challenges",
            "result.node_results[].round_2[].discussion_refinements",
            "result.node_results[].round_2[].new_scenarios",
            "result.node_results[].round_2[].new_scenario_reviews",
            "per-node copies of the global Round 2 and Round 3 methodology",
        ],
        "reconstruction_note": (
            "Removed views are deterministic projections linked by review_id, proposal_id, "
            "adjudication_id, row_id, and node_id; no unique agent decision, criterion "
            "assessment, evidence statement, worksheet row, or lineage identifier is removed."
        ),
    }
    snapshot["snapshot_type"] = "haznova_job_performance_data"
    snapshot["snapshot_reason"] = reason
    snapshot["snapshot_saved_at"] = utc_now()
    snapshot["performance_data_note"] = (
        "This file is generated automatically after a Haznova run reaches a terminal state. "
        "It contains job configuration, model selections, temperature/thinking settings, events, "
        "P&ID parse output, node results, discussion rounds, worksheet rows, actions, "
        "report data, the literature-grounded review methodology, and criterion-level Round 2A/2B "
        "decision traces for later LLM/agent performance evaluation. Exact or deterministically "
        "reconstructable aliases are omitted according to result.storage_schema. API keys are not stored."
    )
    snapshot.pop("api_keys", None)
    return snapshot


def save_job_performance_snapshot(job_id: str, reason: str) -> str:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return ""
        snapshot = job_snapshot_for_performance(job, reason)
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    final_path = PERFORMANCE_DIR / f"haznova_job_{job_id}.json"
    temp_path = PERFORMANCE_DIR / f".haznova_job_{job_id}.json.tmp"
    temp_path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2), encoding="utf-8")
    temp_path.replace(final_path)
    return str(final_path)


def update_job(job_id: str, mutator) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        mutator(job)
        job["updated_at"] = utc_now()


def job_status_snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return the small, immutable payload used by the live status poll."""
    return json_safe({
        "id": job.get("id", ""),
        "status": job.get("status", ""),
        "stop_requested": bool(job.get("stop_requested", False)),
        "submitted_at": job.get("submitted_at", ""),
        "updated_at": job.get("updated_at", ""),
        "error": job.get("error", ""),
        "steps": job.get("steps", []),
        "events": job.get("events", []),
        "progress": job.get("progress", {}),
        "result_revision": int(job.get("result_revision", 0)),
    })


def add_event(job_id: str, phase: str, message: str) -> None:
    def mutator(job):
        job["events"].append({"time": utc_now(), "phase": phase, "message": message})
        job["events"] = job["events"][-250:]
    update_job(job_id, mutator)


def set_status(job_id: str, status: str, error: str = "") -> None:
    def mutator(job):
        job["status"] = status
        job["error"] = error
    update_job(job_id, mutator)


def request_stop_job(job_id: str) -> Dict[str, Any]:
    def mutator(job):
        if job["status"] in {"completed", "failed", "stopped"}:
            return
        job["stop_requested"] = True
        job["status"] = "stopping"
        job["progress"]["message"] = "Stop requested. Waiting for current step to finish."
        job["events"].append({"time": utc_now(), "phase": "stop", "message": "Stop requested by user."})
        job["events"] = job["events"][-250:]
    update_job(job_id, mutator)
    with JOBS_LOCK:
        return job_status_snapshot(JOBS[job_id])


def job_stop_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return bool(job and job.get("stop_requested"))


def ensure_not_stopped(job_id: str) -> None:
    if job_stop_requested(job_id):
        raise JobStopped()


def mark_job_stopped(job_id: str, message: str = "HAZOP stopped. Partial results are available.") -> None:
    def mutator(job):
        job["status"] = "stopped"
        job["progress"]["phase"] = "stopped"
        job["progress"]["message"] = message
        for step in job["steps"]:
            if step["status"] == "running":
                step["status"] = "stopped"
        job["events"].append({"time": utc_now(), "phase": "stopped", "message": message})
        job["events"] = job["events"][-250:]
    update_job(job_id, mutator)


def set_progress(job_id: str, phase: str, message: str, percent: float, current_node: str = "", current_round: str = "") -> None:
    def mutator(job):
        job["progress"] = {
            "phase": phase,
            "message": message,
            "percent": round(max(0.0, min(100.0, percent)), 1),
            "current_node": current_node,
            "current_round": current_round,
        }
    update_job(job_id, mutator)
    add_step(job_id, phase, message, "running")


def add_step(job_id: str, step_id: str, label: str, status: str) -> None:
    def mutator(job):
        for step in job["steps"]:
            if step["id"] == step_id and step["label"] == label:
                step["status"] = status
                return
        job["steps"].append({"id": step_id, "label": label, "status": status})
        job["steps"] = job["steps"][-80:]
    update_job(job_id, mutator)


def complete_latest_step(job_id: str, step_id: str, label: str) -> None:
    """Complete the most recent step with ``step_id``.

    The step ID is the stable workflow identity. Labels are presentation text
    and may legitimately become more concise between start and completion, so
    requiring an exact label match can leave a finished step marked running.
    """
    def mutator(job):
        for step in reversed(job["steps"]):
            if step["id"] == step_id:
                step["status"] = "completed"
                return
    update_job(job_id, mutator)


def set_result(job_id: str, path: List[str], value: Any) -> None:
    def mutator(job):
        target = job["result"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        job["result_revision"] = int(job.get("result_revision", 0)) + 1
    update_job(job_id, mutator)


def append_result(job_id: str, path: List[str], value: Any) -> None:
    def mutator(job):
        target = job["result"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]].append(value)
        job["result_revision"] = int(job.get("result_revision", 0)) + 1
    update_job(job_id, mutator)


def append_model_call(job_id: str, record: Dict[str, Any]) -> None:
    if not job_id:
        return
    def mutator(job):
        performance = job.setdefault("result", {}).setdefault("performance", {})
        calls = performance.setdefault("model_calls", [])
        calls.append(json_safe(record))
        performance["model_call_count"] = len(calls)
    update_job(job_id, mutator)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    raw_docs = payload.get("documents") or {}
    pid_images_raw = raw_docs.get("pid_images")
    pid_images: List[str] = []
    if isinstance(pid_images_raw, list):
        for item in pid_images_raw[:8]:
            if isinstance(item, str) and item.startswith("data:image/") and ";base64," in item:
                pid_images.append(item)
    documents = {
        "hazop_setup": clamp_text(raw_docs.get("hazop_setup", ""), MAX_DOCUMENT_CHARS),
        "pid": clamp_text(raw_docs.get("pid", ""), MAX_DOCUMENT_CHARS),
        "pid_images": pid_images,
        "pid_image_parser_output": "",
        "process_description": clamp_text(raw_docs.get("process_description", ""), MAX_DOCUMENT_CHARS),
        "safeguards": clamp_text(raw_docs.get("safeguards", ""), MAX_DOCUMENT_CHARS),
        "operating_conditions": clamp_text(raw_docs.get("operating_conditions", ""), MAX_DOCUMENT_CHARS),
    }
    text_values = [value for value in documents.values() if isinstance(value, str)]
    if len("\n".join(text_values).strip()) < 120 and not pid_images:
        raise ValueError("Add at least 120 characters of process, P&ID, safeguard, or operating-condition text.")
    keys_raw = payload.get("api_keys") or {}
    keys = {
        "openai": normalize_whitespace(keys_raw.get("openai")),
        "anthropic": normalize_whitespace(keys_raw.get("anthropic")),
        "gemini": normalize_whitespace(keys_raw.get("gemini")),
        "ollama": normalize_whitespace(keys_raw.get("ollama")),
    }
    settings_raw = payload.get("settings") or {}
    try:
        max_nodes = int(settings_raw.get("max_nodes", MAX_NODES_DEFAULT))
    except (TypeError, ValueError):
        max_nodes = MAX_NODES_DEFAULT
    settings = {"max_nodes": max(1, min(MAX_NODES_HARD_LIMIT, max_nodes))}
    agents_raw = payload.get("agents") or {}
    agents: Dict[str, Dict[str, Any]] = {}
    for agent_id in AGENT_IDS:
        config = agents_raw.get(agent_id) or {}
        provider = normalize_whitespace(config.get("provider")).lower() or "auto"
        if provider not in PROVIDERS:
            raise ValueError(f"Invalid provider for {agent_id}.")
        agents[agent_id] = {
            "provider": provider,
            "model": normalize_whitespace(config.get("model")),
            "temperature": clamp_temperature(config.get("temperature"), default_agent_temperature(agent_id)),
            "thinking_level": agent_thinking_level(agent_id, config.get("thinking_level")),
        }
        runtime = resolve_runtime(agent_id, agents[agent_id], keys)
        agents[agent_id]["provider"] = runtime["provider"]
        agents[agent_id]["model"] = runtime["model"]
    return {"api_keys": keys, "documents": documents, "agents": agents, "settings": settings}


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW STEPS
# ─────────────────────────────────────────────────────────────────────────────

def parse_pid_images(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse uploaded P&ID images directly with the configured vision model."""
    images = payload["documents"].get("pid_images")
    if not isinstance(images, list) or not images:
        return None

    candidates = vision_fallback_configs(
        "pid_image_parser",
        payload["agents"]["pid_image_parser"],
        payload["api_keys"],
    )

    if candidates:
        parsed = None
        attempt_errors: List[str] = []
        for candidate in candidates:
            provider = candidate.get("provider", "")
            model = candidate.get("model", "")
            try:
                raw = run_vision_model(
                    "pid_image_parser",
                    candidate,
                    payload["api_keys"],
                    pid_image_parser_system_prompt(),
                    pid_image_parser_prompt(payload["documents"]),
                    images,
                    trace={
                        "job_id": payload.get("_job_id", ""),
                        "stage": "pid_image_parse",
                        "round": "P&ID Parse",
                        "attempt_role": f"vision_parse_{provider}_{model}",
                    },
                )
                parsed = extract_json_object(raw)
                parsed["raw_text"] = raw
                parsed["parse_source"] = "vision_model_direct"
                parsed["vision_provider"] = provider
                parsed["vision_model"] = model
                parsed["vision_parse_succeeded"] = True
                parsed = normalize_pid_node_fields(parsed)
                parsed = filter_pid_parse(parsed)
                if attempt_errors:
                    parsed.setdefault("uncertainties", []).append(
                        "Primary vision parse attempt failed; fallback parser used. "
                        + " | ".join(attempt_errors[:3])
                    )
                break
            except Exception as error:
                attempt_errors.append(f"{provider}/{model}: {str(error)[:500]}")

        if parsed is None:
            error_msg = " | ".join(attempt_errors) or "No vision parser attempt returned structured JSON."
            parsed = {
                "parsed_pid_summary": (
                    "Vision parse did not return structured JSON. "
                    "No P&ID image objects were accepted because the parser must be grounded only in visible image evidence."
                ),
                "equipment": [],
                "lines": [],
                "instruments_and_controls": [],
                "valves_and_relief_devices": [],
                "candidate_hazop_nodes": [],
                "uncertainties": [f"Vision parser error: {error_msg}"],
                "raw_text": "",
                "parse_source": "vision_error",
                "vision_parse_succeeded": False,
                "vision_parse_error": error_msg,
            }
    else:
        parsed = {
            "parsed_pid_summary": "P&ID image parsing requires a vision-capable model for the P&ID Image Parser agent.",
            "equipment": [],
            "lines": [],
            "instruments_and_controls": [],
            "valves_and_relief_devices": [],
            "candidate_hazop_nodes": [],
            "uncertainties": ["Select OpenAI, Gemini, Anthropic, or an Ollama vision model for the P&ID Image Parser agent."],
            "raw_text": "",
            "parse_source": "vision_model_required",
            "vision_parse_succeeded": False,
        }

    parsed = filter_pid_parse(parsed)
    parsed.setdefault("equipment", [])
    parsed.setdefault("lines", [])
    if not parsed.get("lines"):
        parsed["lines"] = lines_from_process_streams(parsed)
    parsed.setdefault("instruments_and_controls", parsed.get("instruments", []))
    parsed.setdefault("valves_and_relief_devices", parsed.get("valves", []))
    parsed.setdefault("candidate_hazop_nodes", parsed.get("candidate_nodes", []))
    if not any([
        parsed.get("equipment"),
        parsed.get("lines"),
        parsed.get("process_streams"),
        parsed.get("instruments_and_controls"),
        parsed.get("valves_and_relief_devices"),
        parsed.get("candidate_hazop_nodes"),
    ]):
        parsed.setdefault("uncertainties", []).append(
            "The P&ID parser did not identify enough visible tagged objects or explicit connections to support image-derived HAZOP node candidates. Supplied text remains available for HAZOP context."
        )

    assert isinstance(parsed, dict)
    enrichment_parts = [
        "=== P&ID IMAGE PARSER OUTPUT ===",
        f"Parse source: {parsed.get('parse_source', 'unknown')}",
        f"Vision parse succeeded: {parsed.get('vision_parse_succeeded', False)}",
        f"Summary: {normalize_whitespace(parsed.get('parsed_pid_summary', ''))}",
    ]
    if parsed.get("equipment"):
        enrichment_parts.append(f"Equipment: {compact_json(parsed.get('equipment', []), 20000)}")
    if parsed.get("lines"):
        enrichment_parts.append(f"Lines: {compact_json(parsed.get('lines', []), 20000)}")
    if parsed.get("pipes"):
        raw_pipes = parsed.get("pipes")
        pipes: Dict[str, Any] = dict(raw_pipes) if isinstance(raw_pipes, dict) else {}
        enrichment_parts.append(
            "Pipe geometry summary: "
            f"{pipes.get('horizontal_count', 0)} horizontal + "
            f"{pipes.get('vertical_count', 0)} vertical pipe segments"
        )
    if parsed.get("instruments_and_controls"):
        enrichment_parts.append(f"Instruments and controls: {compact_json(parsed.get('instruments_and_controls', []), 20000)}")
    if parsed.get("valves_and_relief_devices"):
        enrichment_parts.append(f"Valves and relief devices: {compact_json(parsed.get('valves_and_relief_devices', []), 20000)}")
    if parsed.get("candidate_hazop_nodes"):
        enrichment_parts.append(f"Candidate HAZOP nodes: {compact_json(parsed.get('candidate_hazop_nodes', []), 20000)}")
    if parsed.get("uncertainties"):
        enrichment_parts.append(f"Uncertainties: {compact_json(parsed.get('uncertainties', []), 8000)}")
    payload["documents"]["pid_image_parser_output"] = "\n".join(enrichment_parts)
    # Keep parser output in its own evidence channel. Appending it to ``pid``
    # duplicated the same evidence in every downstream model prompt.
    return parsed


def plan_nodes(payload: Dict[str, Any]) -> Dict[str, Any]:
    manual_setup = parse_manual_hazop_setup(payload["documents"])
    if manual_setup.get("nodes"):
        nodes = manual_setup["nodes"][: payload["settings"]["max_nodes"]]
        return {
            "nodes": nodes,
            "study_scope": "Manual HAZOP setup supplied by the user was used as the authoritative basis for node definition and guideword coverage.",
            "assumptions": [],
            "warnings": manual_setup.get("warnings", []),
            "worksheet_columns": manual_setup.get("worksheet_columns", []),
            "guideword_matrix": manual_setup.get("guideword_matrix", []),
            "raw_text": "Used supplied manual HAZOP setup.",
        }

    scoped_parameters = requested_hazop_parameters(payload["documents"])
    guideword_scope = requested_guideword_scope(payload["documents"])

    raw = run_model(
        "panel_chair",
        payload["agents"]["panel_chair"],
        payload["api_keys"],
        system_prompt_for("panel_chair"),
        planning_prompt(payload["documents"], payload["settings"]["max_nodes"]),
    )
    parsed = extract_json_object(raw)
    nodes_raw: List[Any] = list(parsed.get("nodes") or []) if isinstance(parsed.get("nodes"), list) else []
    nodes = [
        sanitize_node(node, index + 1)
        for index, node in enumerate(nodes_raw[: payload["settings"]["max_nodes"]])
        if isinstance(node, dict)
    ]
    nodes = apply_study_scope(nodes, scoped_parameters, guideword_scope)
    if not nodes:
        nodes = [sanitize_node({
            "node_id": "N-001",
            "node_name": "Overall process",
            "node_description": "Fallback node generated because no manual nodes, image-parser candidate nodes, or model-planned nodes were available.",
            "design_intent": "Operate the supplied process according to the submitted description.",
        }, 1)]
    return {
        "nodes": nodes,
        "study_scope": normalize_whitespace(parsed.get("study_scope")) or "Panel Chair planned nodes from supplied text because no manual node list was supplied.",
        "assumptions": sanitize_assumptions(parsed.get("assumptions"), 20),
        "warnings": sanitize_list(parsed.get("warnings"), 20),
        "raw_text": raw,
    }


def run_agent_json_with_retry(
    agent_id: str,
    label: str,
    payload: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    expected_shape: str,
    trace: Optional[Dict[str, Any]] = None,
    allow_repair_retry: bool = True,
) -> Dict[str, Any]:
    raw = ""
    try:
        raw = run_model(
            agent_id,
            payload["agents"][agent_id],
            payload["api_keys"],
            system_prompt,
            user_prompt,
            trace={**(trace or {}), "attempt_role": (trace or {}).get("attempt_role", "primary")},
        )
        parsed = extract_json_object(raw)
        parsed["raw_text"] = raw
        return parsed
    except Exception as first_error:
        if not allow_repair_retry:
            raise
        retry_prompt = compact_retry_prompt(label, expected_shape, raw or str(first_error))
        try:
            retry_raw = run_model(
                agent_id,
                payload["agents"][agent_id],
                payload["api_keys"],
                system_prompt,
                retry_prompt,
                trace={**(trace or {}), "attempt_role": "json_repair_retry"},
            )
            parsed = extract_json_object(retry_raw)
            parsed["raw_text"] = retry_raw
            parsed["repair_note"] = f"Initial response failed or was invalid and was repaired after retry: {first_error}"
            return parsed
        except Exception as second_error:
            if agent_id == "mechanical_engineer":
                return mechanical_fallback_response(label, first_error, second_error)
            raise second_error


def mechanical_fallback_response(label: str, first_error: Exception, second_error: Exception) -> Dict[str, Any]:
    return {
        "agent_name": label,
        "opening_statement": (
            "Mechanical review fallback used because the model returned invalid JSON. "
            "No mechanical scenarios were generated by fallback because unsupported generic causes must not be inserted into an arbitrary P&ID."
        ),
        "domain_findings": [
            "Mechanical Engineer output could not be parsed as valid JSON after retry, so the Chair should rely on successfully parsed expert statements and visible/supplied evidence only.",
            "Mechanical integrity topics such as equipment limits, line blockage, leakage, backflow, relief basis, corrosion, and material compatibility should be carried as verification needs only when supported by the uploaded P&ID or documents."
        ],
        "scenarios": [],
        "not_applicable": [],
        "fallback_used": True,
        "failed": True,
        "repair_note": f"Initial parse failed: {first_error}; retry failed: {second_error}",
    }


def mechanical_round_two_fallback(label: str, error: Exception) -> Dict[str, Any]:
    return {
        "agent_name": label,
        "round_2_position": (
            "Mechanical review fallback retained credible deviations and requested verification of undocumented mechanical safeguards, equipment limits, blockage, backflow, relief, leakage, and material compatibility."
        ),
        "agreements": [
            {
                "row_id": "credible flow deviation rows",
                "agrees_with_agent": "Process Engineer",
                "reason": "Flow loss, low flow, high flow, reverse flow, and wrong material are mechanically credible when consistent with the node boundary.",
            }
        ],
        "safeguard_challenges": [
            {
                "row_id": "all applicable rows",
                "responds_to_agent": "Panel Chair",
                "issue": "Do not assume mechanical safeguards or equipment condition unless visible or documented.",
                "importance": "material",
                "proposed_revision": "Carry missing check valve, relief basis, corrosion allowance, line inspection, valve maintenance, and material compatibility as specific verification actions.",
            }
        ],
        "scenario_challenges": [],
        "challenges": [],
        "new_scenarios": [],
        "not_applicable_reviews": [],
        "agreement_notes": [],
        "fallback_used": True,
        "failed": False,
        "repair_note": f"Mechanical fallback reason: {error}",
    }


def chair_define_effects_for_node(payload: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    raw = ""
    try:
        raw = run_model(
            "panel_chair",
            payload["agents"]["panel_chair"],
            payload["api_keys"],
            system_prompt_for("panel_chair"),
            chair_effect_definition_prompt(node, payload["documents"]),
            trace={
                "job_id": payload.get("_job_id", ""),
                "stage": "chair_effect_definition",
                "node_id": node.get("node_id", ""),
                "round": "Chair setup",
                "attempt_role": "chair_effect_definition",
            },
        )
        parsed = extract_json_object(raw)
    except Exception as error:
        parsed = {
            "chair_effect_definition_summary": [
                f"Chair effect-definition fallback used because the Chair model response was incomplete: {error}",
                "The fallback preserved the required matrix order and created compact effect/deviation wording from the supplied matrix, node boundary, design intent, parameter, and guideword."
            ],
            "effect_definitions": [],
        }
    definitions = parsed.get("effect_definitions")
    if not isinstance(definitions, list):
        definitions = []
    updated_node = apply_chair_effect_definitions(node, definitions)
    parsed["effect_definitions"] = definitions
    parsed["node"] = updated_node
    parsed["raw_text"] = raw
    return parsed


def round_one_for_node(payload: Dict[str, Any], node: Dict[str, Any]) -> List[Dict[str, Any]]:
    contributors = [
        ("process_engineer", "Process Engineer"),
        ("instrumentation_control", "Instrumentation & Control Engineer"),
        ("safety_consequence", "Safety Engineer"),
        ("maintenance_reliability", "Maintenance Engineer"),
        ("mechanical_engineer", "Mechanical Engineer"),
    ]
    futures = {}
    for agent_id, label in contributors:
        futures[EXECUTOR.submit(
            run_agent_json_with_retry,
            agent_id,
            label,
            payload,
            system_prompt_for(agent_id),
            expert_round_one_prompt(agent_id, label, node, payload["documents"]),
            round_one_json_shape(label),
            trace={
                "job_id": payload.get("_job_id", ""),
                "stage": "round_1_expert_elicitation",
                "node_id": node.get("node_id", ""),
                "round": "Round 1",
                "attempt_role": "expert_elicitation",
            },
        )] = (agent_id, label)
    outputs = []
    for future in as_completed(futures):
        agent_id, label = futures[future]
        try:
            parsed = future.result()
            outputs.append(parsed)
        except Exception as error:
            if agent_id == "mechanical_engineer":
                outputs.append(mechanical_fallback_response(label, error, error))
            else:
                outputs.append({"agent_name": label, "opening_statement": f"Agent failed: {error}", "scenarios": [], "failed": True})
    outputs.sort(key=lambda item: normalize_whitespace(item.get("agent_name", "")))
    return outputs


def merge_round_one(payload: Dict[str, Any], node: Dict[str, Any], round_one: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw = ""
    try:
        raw = run_model(
            "panel_chair", payload["agents"]["panel_chair"], payload["api_keys"],
            system_prompt_for("panel_chair"),
            chair_merge_prompt(node, round_one, payload["documents"]),
            trace={
                "job_id": payload.get("_job_id", ""),
                "stage": "round_1_chair_merge",
                "node_id": node.get("node_id", ""),
                "round": "Round 1",
                "attempt_role": "chair_consolidation",
            },
        )
        parsed = extract_json_object(raw)
    except Exception as error:
        parsed = {
            "round_1_summary": f"Panel Chair merge fallback used because the Chair model response was incomplete: {error}",
            "draft_rows": [],
        }
    parsed["draft_rows"] = parsed.get("draft_rows") if isinstance(parsed.get("draft_rows"), list) else []
    if not isinstance(parsed.get("round_1_chair_opening"), list):
        opening = normalize_whitespace(parsed.get("round_1_chair_opening", ""))
        parsed["round_1_chair_opening"] = [opening] if opening else []
    if credible_row_count(parsed["draft_rows"]) == 0:
        fallback_rows = rows_from_round_one(node, round_one)
        if fallback_rows:
            parsed["draft_rows"] = fallback_rows
            parsed["round_1_chair_opening"] = parsed.get("round_1_chair_opening") or [
                "The Panel Chair opened the node by confirming the node boundary, design intent, and the required guideword-parameter coverage before consolidating the expert findings.",
                "The Panel Chair retained credible expert scenarios for Round 2 discussion because every guideword must be assessed and unsupported Not applicable conclusions cannot replace a credible cause and consequence.",
            ]
            parsed["round_1_summary"] = (
                "Round 1 fallback preserved credible expert scenarios because the Chair merge did not return usable worksheet rows."
            )
    parsed["raw_text"] = raw
    return parsed


def normalized_round_two_decision(value: Any) -> str:
    text = normalize_whitespace(value).lower().replace("-", "_").replace(" ", "_")
    if text in {"agree", "agreed"}:
        return "agree"
    if text in {"partially_agree", "partial_agree", "partly_agree", "partially_agreed"}:
        return "partially_agree"
    if text in {"disagree", "disagreed"}:
        return "disagree"
    return ""


def normalized_review_criterion_rating(value: Any) -> str:
    """Normalize one criterion rating to the closed review vocabulary."""
    if isinstance(value, dict):
        value = value.get("rating") or value.get("status") or value.get("assessment")
    text = normalize_whitespace(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pass": "pass",
        "passed": "pass",
        "meets": "pass",
        "met": "pass",
        "acceptable": "pass",
        "revision_required": "revision_required",
        "requires_revision": "revision_required",
        "needs_revision": "revision_required",
        "revise": "revision_required",
        "partial": "revision_required",
        "partially_meets": "revision_required",
        "fail": "fail",
        "failed": "fail",
        "does_not_meet": "fail",
        "not_met": "fail",
        "unacceptable": "fail",
    }
    return aliases.get(text, "")


def normalize_round_two_criterion_assessment(value: Any) -> Dict[str, Dict[str, str]]:
    """Validate and normalize all criterion-level ratings returned by an agent.

    Every criterion requires a closed rating, a technical reason, and an
    evidence statement. Returning an incomplete rubric is treated as an
    incomplete review assignment and triggers the existing retry mechanism.
    """
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, Dict[str, str]] = {}
    for criterion in ROUND_TWO_REVIEW_CRITERIA:
        raw = value.get(criterion)
        if not isinstance(raw, dict):
            return {}
        rating = normalized_review_criterion_rating(raw.get("rating"))
        reason = normalize_whitespace(raw.get("reason"))
        evidence = normalize_whitespace(raw.get("evidence"))
        if not rating or not reason or not evidence:
            return {}
        normalized[criterion] = {
            "rating": rating,
            "reason": reason,
            "evidence": evidence,
        }
    return normalized


def derive_round_two_decision(
    criterion_assessment: Dict[str, Dict[str, str]],
) -> str:
    """Derive one scenario-validity outcome without using safeguard adequacy.

    Safeguards are existing protections to be identified and audited; their
    absence or an action-field correction does not decide whether the hazard
    scenario itself is credible.  Only the four scenario-validity checks drive
    Agree, Partially agree, or Disagree.
    """
    if any(criterion not in criterion_assessment for criterion in ROUND_TWO_REVIEW_CRITERIA):
        return ""
    core_ratings = [
        criterion_assessment[criterion].get("rating", "")
        for criterion in ROUND_TWO_CORE_CRITERIA
    ]
    if any(rating == "fail" for rating in core_ratings):
        return "disagree"
    if any(rating != "pass" for rating in core_ratings):
        return "partially_agree"
    return "agree"


def criterion_affected_fields(
    criterion_assessment: Dict[str, Dict[str, str]],
) -> List[str]:
    """Map non-passing criterion results to auditable worksheet fields."""
    field_map = {
        "node_guideword_relevance": "deviation",
        "physical_causal_credibility": "cause/deviation",
        "consequence_validity": "consequence",
        "scenario_distinctness": "cause/deviation",
        "safeguard_action_validity": "safeguard/necessary_action",
    }
    return unique_strings([
        field_map[criterion]
        for criterion, assessment in criterion_assessment.items()
        if assessment.get("rating") != "pass" and criterion in field_map
    ])


def criterion_rating_summary(value: Any) -> List[str]:
    """Return compact, ordered criterion ratings for human-facing reports.

    Full criterion reasons and evidence remain in ``review_decision_trace`` and
    the nested Round 2 records. This summary keeps the DOCX audit readable
    without discarding the machine-readable trace.
    """
    if not isinstance(value, dict):
        return []
    labels = {
        "node_guideword_relevance": "Node/guideword relevance",
        "physical_causal_credibility": "Physical/causal credibility",
        "consequence_validity": "Consequence validity",
        "safeguard_action_validity": "Safeguard/action validity",
    }
    return [
        f"{labels.get(criterion, criterion)}: "
        f"{normalize_whitespace((value.get(criterion) or {}).get('rating')) or 'not completed'}"
        for criterion in ROUND_TWO_REVIEW_CRITERIA
    ]


def rubric_decision_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return the normalized rubric, derived decision, and audit comparison."""
    criteria = normalize_round_two_criterion_assessment(
        item.get("criterion_assessment") or item.get("criteria")
    )
    derived = derive_round_two_decision(criteria)
    suggested = normalized_round_two_decision(
        item.get("suggested_decision")
        or item.get("recommended_decision")
        or item.get("final_decision")
        or item.get("decision")
    )
    return {
        "criterion_assessment": criteria,
        "decision": derived,
        "agent_suggested_decision": suggested,
        "decision_overridden_by_rubric": bool(suggested and derived and suggested != derived),
        "decision_method": "deterministic_scenario_validity_rubric",
    }


def unique_round_two_items(items: List[Any]) -> List[Any]:
    output: List[Any] = []
    seen: Set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def meaningful_round_two_recommendation(item: Any) -> bool:
    """Reject empty model-generated recommendation shells.

    Some batch responses repeat the JSON example with every technical field
    empty. Those objects are not findings and should not be saved or rendered
    as consensus recommendations.
    """
    if not isinstance(item, dict):
        return bool(normalize_whitespace(item))
    identity_fields = (
        "guideword", "parameter", "matrix_key", "recommended_deviation",
        "deviation", "recommended_potential_cause", "potential_cause",
        "cause", "recommended_consequence", "consequence",
    )
    placeholders = {
        "", "n/a", "none", "not applicable", "not identified",
        "requires verification", "refer to engineering team for further study",
    }
    for field in identity_fields:
        values = row_text_items(item.get(field), 20)
        if any(normalize_whitespace(value).lower() not in placeholders for value in values):
            return True
    return False


def round_two_compatibility_lists(
    reviews: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build the legacy agreement/challenge views from the canonical decisions."""
    agreements = [
        {
            "row_id": review["review_id"],
            "agrees_with_agent": review.get("source_agent", ""),
            "reason": review.get("reason", ""),
            "evidence_basis": review.get("evidence_basis", ""),
        }
        for review in reviews
        if review.get("decision") == "agree"
    ]
    challenges = [
        {
            "row_id": review["review_id"],
            "responds_to_agent": review.get("source_agent", ""),
            "issue": review.get("reason", ""),
            "reason": review.get("reason", ""),
            "evidence_basis": review.get("evidence_basis", ""),
            "importance": "material" if review.get("decision") == "disagree" else "minor",
            "proposed_revision": review.get("proposed_revision", ""),
            "agreement_or_disagreement": review.get("decision", ""),
        }
        for review in reviews
        if review.get("decision") in {"partially_agree", "disagree"}
    ]
    return agreements, challenges


def missing_round_two_review(
    source: Dict[str, Any], attempts: int, *, id_field: str = "review_id"
) -> Dict[str, Any]:
    """Preserve an exhausted review assignment without fabricating a decision."""
    item_id = str(source.get(id_field, ""))
    missing = {
        id_field: item_id,
        "decision": "",
        "agent_suggested_decision": "",
        "criterion_assessment": {},
        "decision_method": "deterministic_scenario_validity_rubric",
        "decision_overridden_by_rubric": False,
        "reason": "",
        "evidence_basis": "",
        "affected_fields": [],
        "proposed_revision": "",
        "review_status": "missing_after_3_attempts",
        "review_attempts": attempts,
    }
    if id_field == "review_id":
        missing.update({
            "row_id": item_id,
            "source_agent": source.get("source_agent", ""),
            "source_row_id": source.get("source_row_id", ""),
            "agreement_or_disagreement": "",
            "reviewed_scenario": {
                key: source.get(key, "")
                for key in (
                    "review_id", "row_id", "source_agent", "source_row_id",
                    "node_id", "guideword", "parameter", "matrix_key",
                    "expected_deviation", "deviation", "potential_cause",
                    "consequence", "safeguard", "necessary_action",
                )
            },
        })
    return missing


def run_exhaustive_round_two_agent(
    agent_id: str,
    label: str,
    payload: Dict[str, Any],
    node: Dict[str, Any],
    draft_rows: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    batches = [
        review_rows[index:index + ROUND_TWO_REVIEW_BATCH_SIZE]
        for index in range(0, len(review_rows), ROUND_TWO_REVIEW_BATCH_SIZE)
    ]
    if not batches:
        raise RuntimeError(f"{label} cannot perform Round 2 because Round 1 produced no reviewable scenarios.")

    reviews_by_id: Dict[str, Dict[str, Any]] = {}
    positions: List[str] = []
    new_scenarios: List[Any] = []
    not_applicable_reviews: List[Any] = []
    consensus_recommendations: List[Any] = []
    review_errors: List[str] = []

    for batch_index, batch in enumerate(batches, start=1):
        pending = list(batch)
        completeness_attempt = 0
        while pending and completeness_attempt < ROUND_TWO_BATCH_RETRY_ATTEMPTS:
            completeness_attempt += 1
            ensure_not_stopped(payload.get("_job_id", ""))
            requested_rows = pending
            try:
                parsed = run_agent_json_with_retry(
                    agent_id,
                    label,
                    payload,
                    system_prompt_for(agent_id),
                    expert_round_two_exhaustive_prompt(
                        agent_id,
                        label,
                        node,
                        draft_rows,
                        requested_rows,
                        payload["documents"],
                        batch_index,
                        len(batches),
                    ),
                    exhaustive_round_two_json_shape(label),
                    trace={
                        "job_id": payload.get("_job_id", ""),
                        "stage": "round_2_exhaustive_cross_review",
                        "node_id": node.get("node_id", ""),
                        "round": "Round 2",
                        "attempt_role": f"expert_review_batch_{batch_index}_attempt_{completeness_attempt}",
                    },
                    allow_repair_retry=False,
                )
            except Exception as error:
                review_errors.append(
                    f"Batch {batch_index}, attempt {completeness_attempt}: {error}"
                )
                parsed = {}
            position = normalize_whitespace(parsed.get("round_2_position"))
            if position:
                positions.append(position)
            new_scenarios.extend(parsed.get("new_scenarios") or [])
            not_applicable_reviews.extend(parsed.get("not_applicable_reviews") or [])
            consensus_recommendations.extend(parsed.get("consensus_recommendations") or [])

            pending_by_id = {str(row["review_id"]): row for row in requested_rows}
            for item in parsed.get("scenario_reviews") or []:
                if not isinstance(item, dict):
                    continue
                review_id = normalize_whitespace(item.get("review_id"))
                source = pending_by_id.get(review_id)
                rubric = rubric_decision_record(item)
                decision = rubric["decision"]
                reason = normalize_whitespace(item.get("reason"))
                evidence_basis = normalize_whitespace(item.get("evidence_basis"))
                proposed_revision = normalize_whitespace(item.get("proposed_revision"))
                if (
                    source is None
                    or not decision
                    or not reason
                    or not evidence_basis
                    or (decision != "agree" and not proposed_revision)
                ):
                    continue
                if decision == "agree" and not proposed_revision:
                    proposed_revision = "No revision required"
                reviews_by_id[review_id] = {
                    "review_id": review_id,
                    "row_id": review_id,
                    "source_agent": source.get("source_agent", ""),
                    "source_row_id": source.get("source_row_id", ""),
                    "decision": decision,
                    "agreement_or_disagreement": decision,
                    "agent_suggested_decision": rubric["agent_suggested_decision"],
                    "criterion_assessment": rubric["criterion_assessment"],
                    "decision_method": rubric["decision_method"],
                    "decision_overridden_by_rubric": rubric["decision_overridden_by_rubric"],
                    "reason": reason,
                    "evidence_basis": evidence_basis,
                    "affected_fields": unique_strings(
                        sanitize_list(item.get("affected_fields"), 8)
                        + criterion_affected_fields(rubric["criterion_assessment"])
                    ),
                    "proposed_revision": proposed_revision,
                    "review_status": "completed",
                    "review_attempts": completeness_attempt,
                    "reviewed_scenario": {
                        key: source.get(key, "")
                        for key in (
                            "review_id", "row_id", "source_agent", "source_row_id",
                            "node_id", "guideword", "parameter", "matrix_key",
                            "expected_deviation", "deviation", "potential_cause",
                            "consequence", "safeguard", "necessary_action",
                        )
                    },
                }
            pending = [row for row in pending if str(row["review_id"]) not in reviews_by_id]

        for source in pending:
            review_id = str(source["review_id"])
            reviews_by_id[review_id] = missing_round_two_review(
                source, ROUND_TWO_BATCH_RETRY_ATTEMPTS
            )

    ordered_reviews = [reviews_by_id[str(row["review_id"])] for row in review_rows]
    missing_review_ids = [
        str(review.get("review_id", ""))
        for review in ordered_reviews
        if review.get("review_status") != "completed"
    ]
    reviewed_count = len(ordered_reviews) - len(missing_review_ids)
    agreements, scenario_challenges = round_two_compatibility_lists(ordered_reviews)
    return {
        "agent_name": label,
        "round_2_position": " ".join(unique_round_two_items(positions)),
        "round_2_position_batches": unique_round_two_items(positions),
        "scenario_reviews": ordered_reviews,
        "review_complete": not missing_review_ids,
        "reviewed_scenario_count": reviewed_count,
        "required_scenario_count": len(review_rows),
        "missing_review_count": len(missing_review_ids),
        "missing_review_ids": missing_review_ids,
        "review_attempt_limit": ROUND_TWO_BATCH_RETRY_ATTEMPTS,
        "review_errors": review_errors,
        "agreements": agreements,
        "safeguard_challenges": [],
        "scenario_challenges": scenario_challenges,
        "challenges": [],
        "new_scenarios": unique_round_two_items(new_scenarios),
        "not_applicable_reviews": unique_round_two_items(not_applicable_reviews),
        "consensus_recommendations": [
            item for item in unique_round_two_items(consensus_recommendations)
            if meaningful_round_two_recommendation(item)
        ],
        "agreement_notes": [],
        "failed": False,
    }


def build_round_two_contested_packets(
    review_rows: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect every unresolved Round 2A scenario for focused discussion.

    Only a completed unanimous-Agree profile with passing scenario-validity
    checks bypasses Round 2B.  Unanimous Partially agree or Disagree still
    enters discussion so the panel can refine the evidence and report one of
    the three conclusions without a mixed category.
    """
    source_by_id = {str(row.get("review_id", "")): row for row in review_rows}
    reviews_by_agent: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for output in outputs:
        agent_name = normalize_whitespace(output.get("agent_name"))
        reviews_by_agent[agent_name] = {
            str(review.get("review_id", "")): review
            for review in output.get("scenario_reviews", []) or []
            if isinstance(review, dict)
        }

    packets: List[Dict[str, Any]] = []
    for source in review_rows:
        review_id = str(source.get("review_id", ""))
        initial_reviews = []
        decisions: Set[str] = set()
        completed_decisions: List[str] = []
        criterion_ratings: Dict[str, Set[str]] = {
            criterion: set() for criterion in ROUND_TWO_CORE_CRITERIA
        }
        for output in outputs:
            agent_name = normalize_whitespace(output.get("agent_name"))
            review = reviews_by_agent.get(agent_name, {}).get(review_id, {})
            decision = normalized_round_two_decision(review.get("decision"))
            if decision:
                decisions.add(decision)
                completed_decisions.append(decision)
            assessment = review.get("criterion_assessment") or {}
            if isinstance(assessment, dict):
                for criterion in ROUND_TWO_CORE_CRITERIA:
                    rating = normalized_review_criterion_rating(
                        (assessment.get(criterion) or {}).get("rating")
                        if isinstance(assessment.get(criterion), dict)
                        else assessment.get(criterion)
                    )
                    if rating:
                        criterion_ratings[criterion].add(rating)
            initial_reviews.append({
                "agent_name": agent_name,
                "decision": decision,
                "agent_suggested_decision": normalize_whitespace(
                    review.get("agent_suggested_decision")
                ),
                "criterion_assessment": review.get("criterion_assessment", {}),
                "decision_overridden_by_rubric": bool(
                    review.get("decision_overridden_by_rubric", False)
                ),
                "reason": normalize_whitespace(review.get("reason")),
                "evidence_basis": normalize_whitespace(review.get("evidence_basis")),
                "proposed_revision": normalize_whitespace(review.get("proposed_revision")),
            })
        criterion_conflict = any(
            len(ratings) > 1 for ratings in criterion_ratings.values()
        )
        requires_discussion = bool(
            len(decisions) > 1
            or any(decision != "agree" for decision in completed_decisions)
            or criterion_conflict
        )
        # Missing assignments remain a coverage issue and do not themselves
        # create a technical position.  A fully passing unanimous Agree profile
        # is already converged and does not require a second model call.
        if not requires_discussion:
            continue
        packets.append({
            "review_id": review_id,
            "reviewed_scenario": {
                key: source_by_id.get(review_id, {}).get(key, "")
                for key in (
                    "review_id", "row_id", "source_agent", "source_row_id",
                    "node_id", "guideword", "parameter", "matrix_key",
                    "expected_deviation", "deviation", "potential_cause",
                    "consequence", "safeguard", "necessary_action",
                )
            },
            "initial_reviews": initial_reviews,
            "round_2a_decision_categories": sorted(decisions),
            "round_2a_criterion_conflict": criterion_conflict,
        })
    return packets


def run_round_two_refinement_agent(
    agent_id: str,
    label: str,
    payload: Dict[str, Any],
    node: Dict[str, Any],
    initial_output: Dict[str, Any],
    contested_packets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run one evidence-based rebuttal pass over every contested Round 2A item."""
    initial_by_id = {
        str(review.get("review_id", "")): review
        for review in initial_output.get("scenario_reviews", []) or []
        if isinstance(review, dict)
    }
    batches = [
        contested_packets[index:index + ROUND_TWO_REFINEMENT_BATCH_SIZE]
        for index in range(0, len(contested_packets), ROUND_TWO_REFINEMENT_BATCH_SIZE)
    ]
    refinements_by_id: Dict[str, Dict[str, Any]] = {}
    refinement_errors: List[str] = []

    for batch_index, batch in enumerate(batches, start=1):
        pending = list(batch)
        completeness_attempt = 0
        while pending and completeness_attempt < ROUND_TWO_BATCH_RETRY_ATTEMPTS:
            completeness_attempt += 1
            ensure_not_stopped(payload.get("_job_id", ""))
            requested_packets = pending
            try:
                parsed = run_agent_json_with_retry(
                    agent_id,
                    label,
                    payload,
                    system_prompt_for(agent_id),
                    expert_round_two_refinement_prompt(
                        agent_id,
                        label,
                        node,
                        requested_packets,
                        payload["documents"],
                        batch_index,
                        len(batches),
                    ),
                    round_two_refinement_json_shape(label),
                    trace={
                        "job_id": payload.get("_job_id", ""),
                        "stage": "round_2b_cross_agent_discussion",
                        "node_id": node.get("node_id", ""),
                        "round": "Round 2B",
                        "attempt_role": f"discussion_batch_{batch_index}_attempt_{completeness_attempt}",
                    },
                    allow_repair_retry=False,
                )
            except Exception as error:
                refinement_errors.append(
                    f"Batch {batch_index}, attempt {completeness_attempt}: {error}"
                )
                parsed = {}
            pending_by_id = {str(packet["review_id"]): packet for packet in requested_packets}
            for item in parsed.get("refined_reviews") or []:
                if not isinstance(item, dict):
                    continue
                review_id = normalize_whitespace(item.get("review_id"))
                if review_id not in pending_by_id or review_id not in initial_by_id:
                    continue
                rubric = rubric_decision_record(item)
                final_decision = rubric["decision"]
                technical_response = normalize_whitespace(item.get("technical_response"))
                evidence_basis = normalize_whitespace(item.get("evidence_basis"))
                change_reason = normalize_whitespace(item.get("change_reason"))
                proposed_resolution = normalize_whitespace(item.get("proposed_resolution"))
                if not all((final_decision, technical_response, evidence_basis, change_reason, proposed_resolution)):
                    continue
                initial_decision = normalized_round_two_decision(
                    initial_by_id[review_id].get("decision")
                )
                refinements_by_id[review_id] = {
                    "review_id": review_id,
                    "initial_decision": initial_decision,
                    "final_decision": final_decision,
                    "agent_suggested_final_decision": rubric["agent_suggested_decision"],
                    "criterion_assessment": rubric["criterion_assessment"],
                    "decision_method": rubric["decision_method"],
                    "decision_overridden_by_rubric": rubric["decision_overridden_by_rubric"],
                    "responds_to_agents": sanitize_list(item.get("responds_to_agents"), 12),
                    "technical_response": technical_response,
                    "evidence_basis": evidence_basis,
                    "position_changed": final_decision != initial_decision,
                    "change_reason": change_reason,
                    "proposed_resolution": proposed_resolution,
                    "review_status": "completed",
                    "review_attempts": completeness_attempt,
                }
            pending = [
                packet for packet in pending
                if str(packet["review_id"]) not in refinements_by_id
            ]

        for packet in pending:
            review_id = str(packet["review_id"])
            refinements_by_id[review_id] = {
                "review_id": review_id,
                "initial_decision": normalized_round_two_decision(
                    initial_by_id.get(review_id, {}).get("decision")
                ),
                "final_decision": "",
                "agent_suggested_final_decision": "",
                "criterion_assessment": {},
                "decision_method": "deterministic_scenario_validity_rubric",
                "decision_overridden_by_rubric": False,
                "responds_to_agents": [],
                "technical_response": "",
                "evidence_basis": "",
                "position_changed": False,
                "change_reason": "",
                "proposed_resolution": "",
                "review_status": "missing_after_3_attempts",
                "review_attempts": ROUND_TWO_BATCH_RETRY_ATTEMPTS,
            }

    return [refinements_by_id[str(packet["review_id"])] for packet in contested_packets]


def apply_round_two_refinements(
    output: Dict[str, Any],
    refinements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Preserve Round 2A decisions and replace the canonical view with Round 2B decisions."""
    initial_reviews = [dict(review) for review in output.get("scenario_reviews", []) or []]
    refinement_by_id = {
        str(item.get("review_id", "")): item
        for item in refinements
        if isinstance(item, dict)
    }
    final_reviews: List[Dict[str, Any]] = []
    for initial in initial_reviews:
        final = dict(initial)
        refinement = refinement_by_id.get(str(initial.get("review_id", "")))
        if refinement:
            final_decision = normalized_round_two_decision(refinement.get("final_decision"))
            final["decision"] = final_decision
            final["agreement_or_disagreement"] = final_decision
            final["criterion_assessment"] = refinement.get("criterion_assessment", {})
            final["agent_suggested_decision"] = refinement.get(
                "agent_suggested_final_decision", ""
            )
            final["decision_method"] = refinement.get(
                "decision_method", "deterministic_scenario_validity_rubric"
            )
            final["decision_overridden_by_rubric"] = bool(
                refinement.get("decision_overridden_by_rubric", False)
            )
            final["reason"] = refinement.get("technical_response", "")
            final["evidence_basis"] = refinement.get("evidence_basis", "")
            final["proposed_revision"] = refinement.get("proposed_resolution", "")
            final["round_2a_decision"] = refinement["initial_decision"]
            final["round_2b_position_changed"] = refinement["position_changed"]
            final["round_2b_change_reason"] = refinement["change_reason"]
            final["round_2b_responds_to_agents"] = refinement["responds_to_agents"]
            final["round_2b_review_status"] = refinement.get("review_status", "completed")
            final["round_2b_review_attempts"] = refinement.get("review_attempts", 1)
        final_reviews.append(final)

    missing_refinement_ids = [
        str(item.get("review_id", ""))
        for item in refinements
        if item.get("review_status") == "missing_after_3_attempts"
    ]
    completed_refinement_count = len(refinements) - len(missing_refinement_ids)
    agreements, challenges = round_two_compatibility_lists(final_reviews)
    updated = dict(output)
    updated.update({
        "initial_scenario_reviews": initial_reviews,
        "scenario_reviews": final_reviews,
        "discussion_refinements": refinements,
        "round_2b_complete": not missing_refinement_ids,
        "refined_scenario_count": completed_refinement_count,
        "required_round_2b_review_count": len(refinements),
        "missing_round_2b_review_count": len(missing_refinement_ids),
        "missing_round_2b_review_ids": missing_refinement_ids,
        "round_2b_review_attempt_limit": ROUND_TWO_BATCH_RETRY_ATTEMPTS,
        "agreements": agreements,
        "scenario_challenges": challenges,
    })
    return updated


def _proposal_fingerprint(proposal: Dict[str, Any]) -> str:
    values = [
        proposal.get("guideword"),
        proposal.get("parameter"),
        proposal.get("matrix_key"),
        proposal.get("deviation") or proposal.get("effect"),
        proposal.get("potential_cause") or proposal.get("cause"),
        proposal.get("consequence"),
    ]
    return "|".join(
        re.sub(r"[^a-z0-9]+", " ", normalize_whitespace(value).lower()).strip()
        for value in values
    )


def build_round_two_new_scenario_proposals(
    node: Dict[str, Any],
    outputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign stable IDs and collapse exact or near-duplicate proposals."""
    proposals: List[Dict[str, Any]] = []
    by_fingerprint: Dict[str, Dict[str, Any]] = {}

    def proposal_context(item: Dict[str, Any]) -> str:
        return normalize_whitespace(
            item.get("matrix_key")
            or f"{item.get('guideword', '')}|{item.get('parameter', '')}"
        ).lower()

    def proposal_tokens(item: Dict[str, Any]) -> Set[str]:
        text = normalize_whitespace(" ".join([
            str(item.get("deviation", "")),
            str(item.get("potential_cause", "")),
            " ".join(item.get("consequence", []) or []),
        ])).lower()
        replacements = {
            "fails": "fail", "failed": "fail", "failure": "fail",
            "blocked": "block", "blockage": "block",
            "closed": "close", "closing": "close",
            "opened": "open", "opening": "open",
            "incorrect": "wrong", "incorrectly": "wrong",
        }
        stop_words = {
            "the", "a", "an", "is", "of", "to", "for", "and", "or",
            "may", "can", "could", "would", "in", "on", "at", "from",
        }
        return {
            replacements.get(token, token)
            for token in re.findall(r"[a-z0-9]+", text)
            if replacements.get(token, token) not in stop_words
        }

    def find_near_duplicate(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        context = proposal_context(item)
        tokens = proposal_tokens(item)
        for existing in proposals:
            if context != proposal_context(existing):
                continue
            prior_tokens = proposal_tokens(existing)
            union = tokens | prior_tokens
            overlap = len(tokens & prior_tokens) / len(union) if union else 0.0
            if len(tokens & prior_tokens) >= 4 and overlap >= 0.78:
                return existing
        return None

    for output in outputs:
        agent_name = normalize_whitespace(output.get("agent_name")) or "Expert agent"
        for raw in output.get("new_scenarios", []) or []:
            if not isinstance(raw, dict):
                continue
            proposal = {
                "node_id": node.get("node_id", ""),
                "guideword": normalize_whitespace(raw.get("guideword")),
                "parameter": normalize_whitespace(raw.get("parameter")),
                "matrix_key": normalize_whitespace(raw.get("matrix_key")),
                "expected_deviation": normalize_whitespace(raw.get("expected_deviation")),
                "deviation": normalize_whitespace(raw.get("deviation") or raw.get("effect")),
                "potential_cause": normalize_whitespace(
                    raw.get("potential_cause") or raw.get("cause")
                ),
                "consequence": sanitize_list(raw.get("consequence") or raw.get("consequences"), 8),
                "safeguard": sanitize_list(raw.get("safeguard") or raw.get("safeguards"), 8),
                "necessary_action": sanitize_list(
                    raw.get("necessary_action") or raw.get("recommended_action"), 8
                ),
                "basis": normalize_whitespace(raw.get("basis") or raw.get("reason")),
                "source_agents": [agent_name],
            }
            fingerprint = _proposal_fingerprint(proposal)
            if not fingerprint.replace("|", "").strip():
                continue
            existing = by_fingerprint.get(fingerprint)
            if not existing:
                existing = find_near_duplicate(proposal)
            if existing:
                if agent_name not in existing["source_agents"]:
                    existing["source_agents"].append(agent_name)
                continue
            proposal_id = f"{node.get('node_id', 'N-000')}-R2N-{len(proposals) + 1:03d}"
            proposal["proposal_id"] = proposal_id
            proposal["row_id"] = proposal_id
            proposal["proposal_fingerprint"] = fingerprint
            proposals.append(proposal)
            by_fingerprint[fingerprint] = proposal
    return proposals


def run_round_two_new_scenario_review_agent(
    agent_id: str,
    label: str,
    payload: Dict[str, Any],
    node: Dict[str, Any],
    proposals: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Require one explicit panel decision for every proposed Round 2 addition."""
    batches = [
        proposals[index:index + ROUND_TWO_REFINEMENT_BATCH_SIZE]
        for index in range(0, len(proposals), ROUND_TWO_REFINEMENT_BATCH_SIZE)
    ]
    reviews_by_id: Dict[str, Dict[str, Any]] = {}
    for batch_index, batch in enumerate(batches, start=1):
        pending = list(batch)
        completeness_attempt = 0
        while pending and completeness_attempt < ROUND_TWO_BATCH_RETRY_ATTEMPTS:
            completeness_attempt += 1
            ensure_not_stopped(payload.get("_job_id", ""))
            requested_proposals = pending
            try:
                parsed = run_agent_json_with_retry(
                    agent_id,
                    label,
                    payload,
                    system_prompt_for(agent_id),
                    expert_round_two_new_scenario_review_prompt(
                        agent_id,
                        label,
                        node,
                        requested_proposals,
                        review_rows,
                        payload["documents"],
                        batch_index,
                        len(batches),
                    ),
                    round_two_new_scenario_review_json_shape(label),
                    trace={
                        "job_id": payload.get("_job_id", ""),
                        "stage": "round_2b_new_scenario_cross_review",
                        "node_id": node.get("node_id", ""),
                        "round": "Round 2B",
                        "attempt_role": f"new_scenario_batch_{batch_index}_attempt_{completeness_attempt}",
                    },
                    allow_repair_retry=False,
                )
            except Exception:
                parsed = {}
            pending_by_id = {
                str(item["proposal_id"]): item for item in requested_proposals
            }
            for item in parsed.get("new_scenario_reviews") or []:
                if not isinstance(item, dict):
                    continue
                proposal_id = normalize_whitespace(item.get("proposal_id"))
                rubric = rubric_decision_record(item)
                decision = rubric["decision"]
                reason = normalize_whitespace(item.get("reason"))
                evidence_basis = normalize_whitespace(item.get("evidence_basis"))
                proposed_revision = normalize_whitespace(item.get("proposed_revision"))
                if (
                    proposal_id not in pending_by_id
                    or not decision
                    or not reason
                    or not evidence_basis
                    or (decision != "agree" and not proposed_revision)
                ):
                    continue
                if decision == "agree" and not proposed_revision:
                    proposed_revision = "No revision required"
                reviews_by_id[proposal_id] = {
                    "proposal_id": proposal_id,
                    "decision": decision,
                    "agent_suggested_decision": rubric["agent_suggested_decision"],
                    "criterion_assessment": rubric["criterion_assessment"],
                    "decision_method": rubric["decision_method"],
                    "decision_overridden_by_rubric": rubric["decision_overridden_by_rubric"],
                    "reason": reason,
                    "evidence_basis": evidence_basis,
                    "affected_fields": unique_strings(
                        sanitize_list(item.get("affected_fields"), 8)
                        + criterion_affected_fields(rubric["criterion_assessment"])
                    ),
                    "proposed_revision": proposed_revision,
                    "review_status": "completed",
                    "review_attempts": completeness_attempt,
                }
            pending = [
                proposal for proposal in pending
                if str(proposal["proposal_id"]) not in reviews_by_id
            ]
        for proposal in pending:
            proposal_id = str(proposal["proposal_id"])
            reviews_by_id[proposal_id] = missing_round_two_review(
                proposal,
                ROUND_TWO_BATCH_RETRY_ATTEMPTS,
                id_field="proposal_id",
            )
    return [reviews_by_id[str(proposal["proposal_id"])] for proposal in proposals]


def aggregate_round_two_new_scenario_decisions(
    proposals: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Record one three-category panel conclusion for each proposed scenario."""
    deliberations: List[Dict[str, Any]] = []
    reviewer_count = len(outputs)
    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id", ""))
        reviews = []
        for output in outputs:
            agent_name = normalize_whitespace(output.get("agent_name"))
            review = next(
                (
                    item for item in output.get("new_scenario_reviews", []) or []
                    if str(item.get("proposal_id", "")) == proposal_id
                ),
                None,
            )
            if review:
                reviews.append({"agent_name": agent_name, **review})
        completed_reviews = [
            review for review in reviews
            if review.get("review_status") != "missing_after_3_attempts"
            and normalized_round_two_decision(review.get("decision"))
        ]
        counts = {
            status: sum(review.get("decision") == status for review in reviews)
            for status in ("agree", "partially_agree", "disagree")
        }
        panel_result = panel_decision_classification(
            [review.get("decision") for review in completed_reviews],
            expected_review_count=reviewer_count,
        )
        outcome = panel_result.get("panel_conclusion", "")
        deliberations.append({
            "proposal_id": proposal_id,
            "proposal": proposal,
            "review_count": len(completed_reviews),
            "required_review_count": reviewer_count,
            "missing_review_count": reviewer_count - len(completed_reviews),
            "decision_counts": counts,
            "panel_outcome": outcome,
            "panel_conclusion": outcome,
            "panel_conclusion_basis": panel_result.get("conclusion_basis", ""),
            "panel_review_complete": panel_result.get("review_complete", False),
            "panel_dissent_present": panel_result.get("dissent_present", False),
            "panel_supported": outcome in {"agree", "partially_agree"},
            "requires_chair_adjudication": outcome in {"partially_agree", "disagree"}
            or not panel_result.get("review_complete", False),
            "reviews": reviews,
        })
    return deliberations


def round_two_agreement_statistics(
    outputs: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
    review_field: str,
) -> Dict[str, Any]:
    """Calculate agreement on complete panels without fabricating missing ratings.

    Fleiss' kappa assumes the same number of ratings for every included item.
    Scenarios with a blank agent response are therefore reported as incomplete
    and excluded from the agreement denominator instead of being silently
    treated as either agreement or disagreement.
    """
    reviewer_count = len(outputs)
    pair_count = math.comb(reviewer_count, 2) if reviewer_count >= 2 else 0
    pairwise_scores: List[float] = []
    unanimous_count = 0
    contested_count = 0
    incomplete_scenario_count = 0
    category_totals = {status: 0 for status in ("agree", "partially_agree", "disagree")}
    for source in review_rows:
        review_id = str(source.get("review_id", ""))
        decisions = []
        for output in outputs:
            review = next(
                (
                    item for item in output.get(review_field, []) or []
                    if str(item.get("review_id", "")) == review_id
                ),
                None,
            )
            decision = normalized_round_two_decision(review.get("decision")) if review else ""
            if decision:
                decisions.append(decision)
        if len(decisions) != reviewer_count:
            incomplete_scenario_count += 1
            continue
        counts = {status: decisions.count(status) for status in ("agree", "partially_agree", "disagree")}
        for status, count in counts.items():
            category_totals[status] += count
        agreeing_pairs = sum(math.comb(count, 2) for count in counts.values() if count >= 2)
        pairwise_scores.append(agreeing_pairs / pair_count if pair_count else 1.0)
        if len(set(decisions)) == 1:
            unanimous_count += 1
        else:
            contested_count += 1
    complete_scenario_count = len(pairwise_scores)
    observed_agreement = (
        sum(pairwise_scores) / complete_scenario_count
        if complete_scenario_count else 0.0
    )
    assignment_count = complete_scenario_count * reviewer_count
    category_proportions = {
        status: count / assignment_count if assignment_count else 0.0
        for status, count in category_totals.items()
    }
    expected_agreement = sum(value * value for value in category_proportions.values())
    if not complete_scenario_count:
        fleiss_kappa = 0.0
    elif math.isclose(expected_agreement, 1.0):
        fleiss_kappa = 1.0 if math.isclose(observed_agreement, 1.0) else 0.0
    else:
        fleiss_kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return {
        "pairwise_agreement_rate": observed_agreement,
        "expected_agreement_rate": expected_agreement,
        "fleiss_kappa": fleiss_kappa,
        "decision_distribution": category_totals,
        "complete_scenario_count": complete_scenario_count,
        "incomplete_scenario_count": incomplete_scenario_count,
        "unanimous_count": unanimous_count,
        "unanimous_rate": (
            unanimous_count / complete_scenario_count
            if complete_scenario_count else 0.0
        ),
        "contested_count": contested_count,
    }


def build_round_two_deliberation_summary(
    outputs: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    initial = round_two_agreement_statistics(outputs, review_rows, "initial_scenario_reviews")
    final = round_two_agreement_statistics(outputs, review_rows, "scenario_reviews")
    refinements = [
        refinement
        for output in outputs
        for refinement in output.get("discussion_refinements", []) or []
        if isinstance(refinement, dict)
    ]
    changed_count = sum(bool(item.get("position_changed")) for item in refinements)
    discussed_scenario_count = len({
        normalize_whitespace(item.get("review_id"))
        for item in refinements
        if normalize_whitespace(item.get("review_id"))
    })
    round_2a_required = len(outputs) * len(review_rows)
    round_2a_missing = sum(
        review.get("review_status") == "missing_after_3_attempts"
        for output in outputs
        for review in output.get("initial_scenario_reviews", []) or []
        if isinstance(review, dict)
    )
    round_2b_required = len(refinements)
    round_2b_missing = sum(
        item.get("review_status") == "missing_after_3_attempts"
        for item in refinements
    )
    return {
        "reviewer_count": len(outputs),
        "scenario_count": len(review_rows),
        "round_2a_required_assignment_count": round_2a_required,
        "round_2a_missing_review_count": round_2a_missing,
        "round_2a_review_coverage_rate": (
            (round_2a_required - round_2a_missing) / round_2a_required
            if round_2a_required else 1.0
        ),
        "round_2b_required_assignment_count": round_2b_required,
        "round_2b_discussed_scenario_count": discussed_scenario_count,
        "round_2b_missing_review_count": round_2b_missing,
        "round_2b_review_coverage_rate": (
            (round_2b_required - round_2b_missing) / round_2b_required
            if round_2b_required else 1.0
        ),
        "round_2a_pairwise_agreement_rate": initial["pairwise_agreement_rate"],
        "round_2b_pairwise_agreement_rate": final["pairwise_agreement_rate"],
        "pairwise_agreement_improvement": (
            final["pairwise_agreement_rate"] - initial["pairwise_agreement_rate"]
        ),
        "round_2a_fleiss_kappa": initial["fleiss_kappa"],
        "round_2b_fleiss_kappa": final["fleiss_kappa"],
        "fleiss_kappa_improvement": final["fleiss_kappa"] - initial["fleiss_kappa"],
        "round_2a_decision_distribution": initial["decision_distribution"],
        "round_2b_decision_distribution": final["decision_distribution"],
        "round_2a_metric_scenario_count": initial["complete_scenario_count"],
        "round_2a_incomplete_scenario_count": initial["incomplete_scenario_count"],
        "round_2b_metric_scenario_count": final["complete_scenario_count"],
        "round_2b_incomplete_scenario_count": final["incomplete_scenario_count"],
        "round_2a_unanimous_count": initial["unanimous_count"],
        "round_2a_unanimous_rate": initial["unanimous_rate"],
        "round_2b_unanimous_count": final["unanimous_count"],
        "round_2b_unanimous_rate": final["unanimous_rate"],
        "unanimous_scenario_gain": final["unanimous_count"] - initial["unanimous_count"],
        "round_2a_contested_count": initial["contested_count"],
        "round_2b_remaining_contested_count": final["contested_count"],
        "agent_decision_change_count": changed_count,
        "agent_decision_change_rate": changed_count / len(refinements) if refinements else 0.0,
    }


def panel_decision_classification(
    decisions: List[Any],
    expected_review_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Return one of the three post-discussion panel conclusions.

    Individual decisions are retained for traceability.  A unique plurality
    determines the conclusion; an exact tie is conservatively reported as
    Partially agree because the scenario remains unresolved but is not silently
    rejected or accepted.  Missing reviews are reported as coverage and never
    converted into votes.
    """
    normalized = [
        normalized_round_two_decision(decision)
        for decision in decisions
    ]
    normalized = [decision for decision in normalized if decision]
    counts = {
        decision: normalized.count(decision)
        for decision in ("agree", "partially_agree", "disagree")
    }
    expected = max(0, int(expected_review_count or 0))
    incomplete = bool(expected and len(normalized) < expected)
    if not normalized:
        conclusion = ""
        conclusion_basis = "no_completed_reviews"
    else:
        highest = max(counts.values())
        leaders = [decision for decision, count in counts.items() if count == highest]
        if len(leaders) == 1:
            conclusion = leaders[0]
            if highest == len(normalized):
                conclusion_basis = "unanimous"
            elif highest > len(normalized) / 2:
                conclusion_basis = "majority"
            else:
                conclusion_basis = "plurality"
        else:
            conclusion = "partially_agree"
            conclusion_basis = "tie_resolved_as_partially_agree"
    return {
        "classification": conclusion,
        "panel_conclusion": conclusion,
        "conclusion_basis": conclusion_basis,
        "decision_counts": counts,
        "review_count": len(normalized),
        "expected_review_count": expected or len(normalized),
        "missing_review_count": max(0, expected - len(normalized)),
        "review_complete": not incomplete,
        "dissent_present": len(set(normalized)) > 1,
        "decision_categories_present": sorted(set(normalized)),
    }


def aggregate_panel_criteria(
    assessments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return criterion-level Round 2B distributions for Chair evidence use."""
    output: Dict[str, Any] = {}
    ratings_order = ("pass", "revision_required", "fail")
    for criterion in ROUND_TWO_REVIEW_CRITERIA:
        ratings: List[str] = []
        non_pass_reasons: List[str] = []
        evidence: List[str] = []
        for assessment in assessments:
            item = assessment.get(criterion) if isinstance(assessment, dict) else None
            if not isinstance(item, dict):
                continue
            rating = normalized_review_criterion_rating(item.get("rating"))
            if not rating:
                continue
            ratings.append(rating)
            if rating != "pass":
                non_pass_reasons.append(normalize_whitespace(item.get("reason")))
            evidence.append(normalize_whitespace(item.get("evidence")))
        counts = {rating: ratings.count(rating) for rating in ratings_order}
        if not ratings:
            panel_rating = "not_assessed"
        elif counts["pass"] == len(ratings):
            panel_rating = "pass"
        elif counts["fail"] == len(ratings):
            panel_rating = "fail"
        else:
            panel_rating = "revision_required"
        output[criterion] = {
            "panel_rating": panel_rating,
            "rating_counts": counts,
            "non_pass_reasons": unique_strings(non_pass_reasons),
            "evidence": unique_strings(evidence),
        }
    return output


def build_round_two_chair_digest(
    outputs: List[Dict[str, Any]],
    new_scenario_deliberations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a scenario-centric Round 2 record small enough for Chair adjudication."""
    initial_by_agent = {
        normalize_whitespace(output.get("agent_name")): {
            str(review.get("review_id", "")): review
            for review in output.get("initial_scenario_reviews", []) or []
            if isinstance(review, dict)
        }
        for output in outputs
    }
    final_by_agent = {
        normalize_whitespace(output.get("agent_name")): {
            str(review.get("review_id", "")): review
            for review in output.get("scenario_reviews", []) or []
            if isinstance(review, dict)
        }
        for output in outputs
    }
    ordered_ids: List[str] = []
    for output in outputs:
        for review in output.get("scenario_reviews", []) or []:
            review_id = str(review.get("review_id", ""))
            if review_id and review_id not in ordered_ids:
                ordered_ids.append(review_id)

    scenario_deliberations: List[Dict[str, Any]] = []
    for review_id in ordered_ids:
        initial_decisions = {
            agent: normalized_round_two_decision(reviews.get(review_id, {}).get("decision"))
            for agent, reviews in initial_by_agent.items()
        }
        final_decisions = {
            agent: normalized_round_two_decision(reviews.get(review_id, {}).get("decision"))
            for agent, reviews in final_by_agent.items()
        }
        final_criterion_assessments = [
            reviews.get(review_id, {}).get("criterion_assessment", {})
            for reviews in final_by_agent.values()
        ]
        initial_completed = {
            decision for decision in initial_decisions.values() if decision
        }
        final_completed = {
            decision for decision in final_decisions.values() if decision
        }
        initial_contested = len(initial_completed) > 1
        final_contested = len(final_completed) > 1
        initial_incomplete = any(not decision for decision in initial_decisions.values())
        final_incomplete = any(not decision for decision in final_decisions.values())
        packet: Dict[str, Any] = {
            "review_id": review_id,
            "round_2a_decisions": initial_decisions,
            "round_2b_decisions": final_decisions,
            "round_2a_contested": initial_contested,
            "round_2b_contested": final_contested,
            "round_2a_incomplete": initial_incomplete,
            "round_2b_incomplete": final_incomplete,
            "round_2b_panel_classification": panel_decision_classification(
                list(final_decisions.values()),
                expected_review_count=len(final_decisions),
            ),
            "round_2b_criterion_summary": aggregate_panel_criteria(
                final_criterion_assessments
            ),
        }
        final_statuses = set(final_decisions.values())
        if (
            initial_contested or final_contested
            or initial_incomplete or final_incomplete
            or final_statuses != {"agree"}
        ):
            arguments = []
            for agent in final_by_agent:
                initial = initial_by_agent.get(agent, {}).get(review_id, {})
                final = final_by_agent.get(agent, {}).get(review_id, {})
                arguments.append({
                    "agent_name": agent,
                    "initial_decision": initial_decisions.get(agent, ""),
                    "initial_criterion_assessment": initial.get(
                        "criterion_assessment", {}
                    ),
                    "initial_reason": normalize_whitespace(initial.get("reason")),
                    "final_decision": final_decisions.get(agent, ""),
                    "final_criterion_assessment": final.get(
                        "criterion_assessment", {}
                    ),
                    "agent_suggested_final_decision": normalize_whitespace(
                        final.get("agent_suggested_decision")
                    ),
                    "decision_overridden_by_rubric": bool(
                        final.get("decision_overridden_by_rubric", False)
                    ),
                    "final_reason": normalize_whitespace(final.get("reason")),
                    "evidence_basis": normalize_whitespace(final.get("evidence_basis")),
                    "position_changed": bool(final.get("round_2b_position_changed", False)),
                    "change_reason": normalize_whitespace(final.get("round_2b_change_reason")),
                    "proposed_resolution": normalize_whitespace(final.get("proposed_revision")),
                })
            packet["agent_arguments"] = arguments
        scenario_deliberations.append(packet)

    enriched_new_scenario_deliberations: List[Dict[str, Any]] = []
    for item in new_scenario_deliberations or []:
        if not isinstance(item, dict):
            continue
        enriched = dict(item)
        reviews = [
            review for review in item.get("reviews", []) or []
            if isinstance(review, dict)
        ]
        enriched["round_2b_panel_classification"] = panel_decision_classification(
            [review.get("decision") for review in reviews],
            expected_review_count=int(
                item.get("required_review_count", len(reviews)) or len(reviews)
            ),
        )
        enriched["round_2b_criterion_summary"] = aggregate_panel_criteria([
            review.get("criterion_assessment", {}) for review in reviews
        ])
        enriched_new_scenario_deliberations.append(enriched)

    return {
        "scenario_deliberations": scenario_deliberations,
        "new_scenario_deliberations": enriched_new_scenario_deliberations,
    }


def build_round_two_review_trace(
    node: Dict[str, Any],
    review_rows: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    new_scenario_deliberations: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Flatten all Round 2 judgments into analysis-ready JSON audit records."""
    source_by_id = {
        str(row.get("review_id", "")): row
        for row in review_rows
        if isinstance(row, dict)
    }
    panel_by_id: Dict[str, Dict[str, Any]] = {}
    for review_id in source_by_id:
        decisions = []
        for output in outputs:
            review = next((
                item for item in output.get("scenario_reviews", []) or []
                if isinstance(item, dict) and str(item.get("review_id", "")) == review_id
            ), {})
            decisions.append(review.get("decision"))
        panel_by_id[review_id] = panel_decision_classification(
            decisions, expected_review_count=len(outputs)
        )
    trace: List[Dict[str, Any]] = []
    for output in outputs:
        agent_name = normalize_whitespace(output.get("agent_name"))
        initial_by_id = {
            str(item.get("review_id", "")): item
            for item in output.get("initial_scenario_reviews", []) or []
            if isinstance(item, dict)
        }
        final_by_id = {
            str(item.get("review_id", "")): item
            for item in output.get("scenario_reviews", []) or []
            if isinstance(item, dict)
        }
        refinement_by_id = {
            str(item.get("review_id", "")): item
            for item in output.get("discussion_refinements", []) or []
            if isinstance(item, dict)
        }
        ordered_ids = unique_strings(list(source_by_id) + list(final_by_id))
        for review_id in ordered_ids:
            source = source_by_id.get(review_id, {})
            initial = initial_by_id.get(review_id) or final_by_id.get(review_id, {})
            final = final_by_id.get(review_id, initial)
            refinement = refinement_by_id.get(review_id, {})
            trace.append({
                "node_id": node.get("node_id", ""),
                "node_name": node.get("node_name", ""),
                "review_type": "round_1_scenario",
                "item_id": review_id,
                "agent_name": agent_name,
                "source_agent": source.get("source_agent", ""),
                "matrix_key": source.get("matrix_key", ""),
                "guideword": source.get("guideword", ""),
                "parameter": source.get("parameter", ""),
                "deviation": source.get("deviation", ""),
                "potential_cause": source.get("potential_cause", ""),
                "round_2a": {
                    "decision": initial.get("decision", ""),
                    "agent_suggested_decision": initial.get("agent_suggested_decision", ""),
                    "decision_overridden_by_rubric": bool(
                        initial.get("decision_overridden_by_rubric", False)
                    ),
                    "criterion_assessment": initial.get("criterion_assessment", {}),
                    "reason": initial.get("reason", ""),
                    "evidence_basis": initial.get("evidence_basis", ""),
                    "affected_fields": initial.get("affected_fields", []),
                    "proposed_revision": initial.get("proposed_revision", ""),
                    "review_status": initial.get("review_status", ""),
                    "review_attempts": initial.get("review_attempts", 0),
                },
                "round_2b": {
                    "required": bool(refinement),
                    "decision": final.get("decision", ""),
                    "agent_suggested_decision": final.get("agent_suggested_decision", ""),
                    "decision_overridden_by_rubric": bool(
                        final.get("decision_overridden_by_rubric", False)
                    ),
                    "criterion_assessment": final.get("criterion_assessment", {}),
                    "reason": final.get("reason", ""),
                    "evidence_basis": final.get("evidence_basis", ""),
                    "position_changed": bool(
                        final.get("round_2b_position_changed", False)
                    ),
                    "responds_to_agents": final.get(
                        "round_2b_responds_to_agents", []
                    ),
                    "change_reason": final.get("round_2b_change_reason", ""),
                    "proposed_resolution": final.get("proposed_revision", ""),
                    "review_status": final.get(
                        "round_2b_review_status", "not_required" if not refinement else ""
                    ),
                    "review_attempts": final.get("round_2b_review_attempts", 0),
                    "panel_conclusion": panel_by_id.get(review_id, {}).get(
                        "panel_conclusion", ""
                    ),
                    "panel_conclusion_basis": panel_by_id.get(review_id, {}).get(
                        "conclusion_basis", ""
                    ),
                    "panel_decision_counts": panel_by_id.get(review_id, {}).get(
                        "decision_counts", {}
                    ),
                    "panel_review_complete": panel_by_id.get(review_id, {}).get(
                        "review_complete", False
                    ),
                },
                "decision_method": "deterministic_scenario_validity_rubric",
            })

    for deliberation in new_scenario_deliberations or []:
        proposal = deliberation.get("proposal", {})
        for review in deliberation.get("reviews", []) or []:
            if not isinstance(review, dict):
                continue
            trace.append({
                "node_id": node.get("node_id", ""),
                "node_name": node.get("node_name", ""),
                "review_type": "round_2_new_scenario_proposal",
                "item_id": deliberation.get("proposal_id", ""),
                "agent_name": review.get("agent_name", ""),
                "matrix_key": proposal.get("matrix_key", ""),
                "guideword": proposal.get("guideword", ""),
                "parameter": proposal.get("parameter", ""),
                "deviation": proposal.get("deviation", ""),
                "potential_cause": proposal.get("potential_cause", ""),
                "round_2b": {
                    "required": True,
                    "decision": review.get("decision", ""),
                    "agent_suggested_decision": review.get(
                        "agent_suggested_decision", ""
                    ),
                    "decision_overridden_by_rubric": bool(
                        review.get("decision_overridden_by_rubric", False)
                    ),
                    "criterion_assessment": review.get("criterion_assessment", {}),
                    "reason": review.get("reason", ""),
                    "evidence_basis": review.get("evidence_basis", ""),
                    "affected_fields": review.get("affected_fields", []),
                    "proposed_resolution": review.get("proposed_revision", ""),
                    "review_status": review.get("review_status", ""),
                    "review_attempts": review.get("review_attempts", 0),
                },
                "panel_outcome": deliberation.get("panel_outcome", ""),
                "panel_conclusion": deliberation.get("panel_conclusion", ""),
                "panel_conclusion_basis": deliberation.get(
                    "panel_conclusion_basis", ""
                ),
                "panel_review_complete": deliberation.get(
                    "panel_review_complete", False
                ),
                "decision_counts": deliberation.get("decision_counts", {}),
                "panel_supported": bool(deliberation.get("panel_supported", False)),
                "decision_method": "deterministic_scenario_validity_rubric",
            })
    return trace


def round_two_for_node(
    payload: Dict[str, Any], node: Dict[str, Any],
    draft_rows: List[Dict[str, Any]], round_one: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    challengers = [
        ("panel_chair", "Panel Chair"),
        ("process_engineer", "Process Engineer"),
        ("instrumentation_control", "Instrumentation & Control Engineer"),
        ("safety_consequence", "Safety Engineer"),
        ("maintenance_reliability", "Maintenance Engineer"),
        ("mechanical_engineer", "Mechanical Engineer"),
    ]
    futures = {}
    for agent_id, label in challengers:
        futures[EXECUTOR.submit(
            run_exhaustive_round_two_agent,
            agent_id,
            label,
            payload,
            node,
            draft_rows,
            review_rows,
        )] = (agent_id, label)
    outputs = []
    for future in as_completed(futures):
        agent_id, label = futures[future]
        try:
            parsed = future.result()
            outputs.append(parsed)
        except Exception as error:
            missing_reviews = [
                missing_round_two_review(row, ROUND_TWO_BATCH_RETRY_ATTEMPTS)
                for row in review_rows
            ]
            outputs.append({
                "agent_name": label,
                "round_2_position": f"No valid review was returned after three attempts: {error}",
                "scenario_reviews": missing_reviews,
                "review_complete": False,
                "reviewed_scenario_count": 0,
                "required_scenario_count": len(review_rows),
                "missing_review_count": len(review_rows),
                "missing_review_ids": [str(row.get("review_id", "")) for row in review_rows],
                "review_attempt_limit": ROUND_TWO_BATCH_RETRY_ATTEMPTS,
                "agreements": [],
                "safeguard_challenges": [],
                "scenario_challenges": [],
                "new_scenarios": [],
                "consensus_recommendations": [],
                "failed": True,
            })
    outputs.sort(key=lambda item: normalize_whitespace(item.get("agent_name", "")))
    missing_round_2a_count = sum(
        int(output.get("missing_review_count", 0)) for output in outputs
    )
    job_id = normalize_whitespace(payload.get("_job_id"))
    if job_id and job_id in JOBS:
        add_event(
            job_id,
            "round 2A",
            (
                f"{len(outputs)} agents were assigned all {len(review_rows)} unique "
                "reconciled Round 1 scenarios; "
                f"{missing_round_2a_count} assignments remained blank after three attempts "
                "and were recorded as incomplete coverage rather than disagreement."
            ),
        )

    contested_packets = build_round_two_contested_packets(review_rows, outputs)
    if job_id and job_id in JOBS:
        add_event(
            job_id,
            "round 2B",
            (
                f"Cross-agent discussion started for {len(contested_packets)} existing "
                "scenario(s) requiring refinement after Round 2A."
            ),
        )
    if contested_packets:
        output_by_agent = {
            normalize_whitespace(output.get("agent_name")): output for output in outputs
        }
        refinement_futures = {}
        for agent_id, label in challengers:
            refinement_futures[EXECUTOR.submit(
                run_round_two_refinement_agent,
                agent_id,
                label,
                payload,
                node,
                output_by_agent[label],
                contested_packets,
            )] = label
        refinements_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for refinement_future in as_completed(refinement_futures):
            label = refinement_futures[refinement_future]
            try:
                refinements_by_agent[label] = refinement_future.result()
            except Exception as error:
                initial_output = output_by_agent[label]
                initial_by_id = {
                    str(item.get("review_id", "")): item
                    for item in initial_output.get("scenario_reviews", []) or []
                    if isinstance(item, dict)
                }
                refinements_by_agent[label] = [{
                    "review_id": str(packet.get("review_id", "")),
                    "initial_decision": normalized_round_two_decision(
                        initial_by_id.get(str(packet.get("review_id", "")), {}).get("decision")
                    ),
                    "final_decision": "",
                    "responds_to_agents": [],
                    "technical_response": "",
                    "evidence_basis": "",
                    "position_changed": False,
                    "change_reason": "",
                    "proposed_resolution": "",
                    "review_status": "missing_after_3_attempts",
                    "review_attempts": ROUND_TWO_BATCH_RETRY_ATTEMPTS,
                    "review_error": str(error),
                } for packet in contested_packets]
        outputs = [
            apply_round_two_refinements(
                output,
                refinements_by_agent[normalize_whitespace(output.get("agent_name"))],
            )
            for output in outputs
        ]
    else:
        outputs = [apply_round_two_refinements(output, []) for output in outputs]

    proposals = build_round_two_new_scenario_proposals(node, outputs)
    outputs = [
        {
            **output,
            "new_scenarios": [
                proposal for proposal in proposals
                if normalize_whitespace(output.get("agent_name"))
                in proposal.get("source_agents", [])
            ],
        }
        for output in outputs
    ]
    if job_id and job_id in JOBS:
        add_event(
            job_id,
            "round 2B",
            f"Full-panel cross-review started for {len(proposals)} proposed new scenarios.",
        )
    if proposals:
        proposal_futures = {}
        for agent_id, label in challengers:
            proposal_futures[EXECUTOR.submit(
                run_round_two_new_scenario_review_agent,
                agent_id,
                label,
                payload,
                node,
                proposals,
                review_rows,
            )] = label
        proposal_reviews_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for proposal_future in as_completed(proposal_futures):
            label = proposal_futures[proposal_future]
            try:
                proposal_reviews_by_agent[label] = proposal_future.result()
            except Exception as error:
                proposal_reviews_by_agent[label] = [{
                    **missing_round_two_review(
                        proposal,
                        ROUND_TWO_BATCH_RETRY_ATTEMPTS,
                        id_field="proposal_id",
                    ),
                    "review_error": str(error),
                } for proposal in proposals]
        outputs = [
            {
                **output,
                "new_scenario_reviews": proposal_reviews_by_agent[
                    normalize_whitespace(output.get("agent_name"))
                ],
                "reviewed_new_scenario_count": sum(
                    item.get("review_status") != "missing_after_3_attempts"
                    and bool(normalized_round_two_decision(item.get("decision")))
                    for item in proposal_reviews_by_agent[
                        normalize_whitespace(output.get("agent_name"))
                    ]
                ),
                "required_new_scenario_review_count": len(proposals),
                "missing_new_scenario_review_count": sum(
                    item.get("review_status") == "missing_after_3_attempts"
                    for item in proposal_reviews_by_agent[
                        normalize_whitespace(output.get("agent_name"))
                    ]
                ),
            }
            for output in outputs
        ]
    else:
        outputs = [
            {
                **output,
                "new_scenario_reviews": [],
                "reviewed_new_scenario_count": 0,
                "required_new_scenario_review_count": 0,
                "missing_new_scenario_review_count": 0,
            }
            for output in outputs
        ]
    new_scenario_deliberations = aggregate_round_two_new_scenario_decisions(
        proposals, outputs
    )
    summary = build_round_two_deliberation_summary(outputs, review_rows)
    summary.update({
        "new_scenario_proposal_count": len(proposals),
        "new_scenario_required_review_count": len(proposals) * len(outputs),
        "new_scenario_missing_review_count": sum(
            int(output.get("missing_new_scenario_review_count", 0))
            for output in outputs
        ),
        "panel_supported_new_scenario_count": sum(
            bool(item.get("panel_supported")) for item in new_scenario_deliberations
        ),
        "contested_new_scenario_count": sum(
            bool(item.get("panel_dissent_present")) for item in new_scenario_deliberations
        ),
        "rejected_new_scenario_count": sum(
            item.get("panel_outcome") == "disagree"
            for item in new_scenario_deliberations
        ),
    })
    if job_id and job_id in JOBS:
        add_event(
            job_id,
            "round 2B",
            (
                f"Discussion completed: {summary['agent_decision_change_count']} existing-scenario "
                f"decisions changed; {summary['panel_supported_new_scenario_count']} of "
                f"{summary['new_scenario_proposal_count']} proposed additions received panel support."
            ),
        )
    return outputs, summary, new_scenario_deliberations


def round_three_row_has_complete_rubric(row: Any) -> bool:
    """Validate one evidence-grounded four-check Chair decision."""
    if not isinstance(row, dict) or not normalize_whitespace(row.get("row_id")):
        return False
    criteria = normalize_round_two_criterion_assessment(
        row.get("chair_criterion_assessment") or row.get("criterion_assessment")
    )
    decision = derive_round_two_decision(criteria)
    if not decision:
        return False
    if not normalize_whitespace(row.get("chair_resolution")):
        return False
    if not sanitize_list(row.get("chair_evidence_basis"), 20):
        return False
    if decision == "partially_agree" and not normalize_whitespace(
        row.get("chair_revision_summary")
    ):
        return False
    return True


def round_three_rows_have_complete_rubric(rows: Any) -> bool:
    """Require unique IDs and a complete rubric for every returned candidate."""
    if not isinstance(rows, list) or not rows:
        return False
    candidate_keys = [
        normalize_whitespace(row.get("row_id"))
        if isinstance(row, dict) else ""
        for row in rows
    ]
    return bool(
        all(candidate_keys)
        and len(set(candidate_keys)) == len(candidate_keys)
        and all(round_three_row_has_complete_rubric(row) for row in rows)
    )


def round_three_candidate_universe(
    node: Dict[str, Any],
    reviewed_rows: List[Dict[str, Any]],
    new_scenario_deliberations: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build the exact, stable-ID candidate set that the Chair must adjudicate.

    Round 3 operates on the unique cause-level scenarios reviewed in Round 2,
    plus every panel-reviewed new-scenario proposal. It must not regress to the
    smaller pre-review Chair draft.
    """
    candidates: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for index, raw in enumerate(reviewed_rows or [], 1):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        candidate_id = normalize_whitespace(
            row.get("review_id") or row.get("row_id")
        ) or f"{node.get('node_id', 'N-UNKNOWN')}-R1-{index:03d}"
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        row["row_id"] = candidate_id
        row["review_id"] = normalize_whitespace(row.get("review_id")) or candidate_id
        row["source_review_ids"] = unique_strings(
            sanitize_list(row.get("source_review_ids"), 80) + [row["review_id"]]
        )
        row["source_proposal_ids"] = sanitize_list(
            row.get("source_proposal_ids"), 40
        )
        row["round_3_candidate_id"] = candidate_id
        row["round_3_candidate_origin"] = "round_1_reviewed_scenario"
        candidates.append(row)

    for item in new_scenario_deliberations or []:
        if not isinstance(item, dict):
            continue
        proposal = item.get("proposal")
        if not isinstance(proposal, dict):
            continue
        proposal_id = normalize_whitespace(
            item.get("proposal_id") or proposal.get("proposal_id")
        )
        if not proposal_id or proposal_id in seen:
            continue
        seen.add(proposal_id)
        proposal_with_id = dict(proposal)
        proposal_with_id["proposal_id"] = proposal_id
        row = scenario_to_row(
            proposal_with_id,
            node,
            len(candidates) + 1,
            ", ".join(sanitize_list(proposal.get("source_agents"), 20))
            or "Round 2 panel proposal",
        )
        row["row_id"] = proposal_id
        row["source_review_ids"] = []
        row["source_proposal_ids"] = [proposal_id]
        row["round_3_candidate_id"] = proposal_id
        row["round_3_candidate_origin"] = "round_2_new_scenario_proposal"
        candidates.append(row)
    return candidates


def round_three_digest_for_candidates(
    discussion_digest: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return only the Round 2 evidence linked to one Round 3 batch."""
    review_ids: Set[str] = set()
    proposal_ids: Set[str] = set()
    for row in candidates:
        review_ids.update(sanitize_list(row.get("source_review_ids"), 80))
        proposal_ids.update(sanitize_list(row.get("source_proposal_ids"), 40))
    return {
        "scenario_deliberations": [
            item for item in discussion_digest.get("scenario_deliberations", []) or []
            if isinstance(item, dict)
            and normalize_whitespace(item.get("review_id")) in review_ids
        ],
        "new_scenario_deliberations": [
            item for item in discussion_digest.get("new_scenario_deliberations", []) or []
            if isinstance(item, dict)
            and normalize_whitespace(item.get("proposal_id")) in proposal_ids
        ],
    }


def round_three_panel_basis(
    row: Dict[str, Any],
    discussion_digest: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve the Round 2B panel evidence linked to one Chair candidate."""
    if not isinstance(discussion_digest, dict):
        return {
            "classification": "not_supplied",
            "source_review_ids": [],
            "source_proposal_ids": [],
            "decision_counts": {},
            "criterion_summary": {},
        }
    review_ids = set(sanitize_list(row.get("source_review_ids"), 80))
    proposal_ids = set(sanitize_list(row.get("source_proposal_ids"), 40))
    linked_scenarios = [
        item for item in discussion_digest.get("scenario_deliberations", []) or []
        if isinstance(item, dict) and str(item.get("review_id", "")) in review_ids
    ]
    linked_proposals = [
        item for item in discussion_digest.get("new_scenario_deliberations", []) or []
        if isinstance(item, dict) and str(item.get("proposal_id", "")) in proposal_ids
    ]
    decisions: List[Any] = []
    criterion_assessments: List[Dict[str, Any]] = []
    for item in linked_scenarios:
        decisions.extend((item.get("round_2b_decisions") or {}).values())
        for argument in item.get("agent_arguments", []) or []:
            if isinstance(argument, dict):
                criterion_assessments.append(
                    argument.get("final_criterion_assessment", {})
                )
    for item in linked_proposals:
        for review in item.get("reviews", []) or []:
            if not isinstance(review, dict):
                continue
            decisions.append(review.get("decision"))
            criterion_assessments.append(review.get("criterion_assessment", {}))
    classification = panel_decision_classification(
        decisions,
        expected_review_count=len(decisions),
    )
    stored_criterion_summaries = [
        item.get("round_2b_criterion_summary", {})
        for item in linked_scenarios
        if isinstance(item.get("round_2b_criterion_summary"), dict)
    ]
    return {
        **classification,
        "source_review_ids": sorted(review_ids),
        "source_proposal_ids": sorted(proposal_ids),
        "linked_review_ids": [item.get("review_id", "") for item in linked_scenarios],
        "linked_proposal_ids": [item.get("proposal_id", "") for item in linked_proposals],
        "criterion_summary": (
            aggregate_panel_criteria(criterion_assessments)
            if criterion_assessments
            else (stored_criterion_summaries[0] if len(stored_criterion_summaries) == 1 else {})
        ),
        "linked_criterion_summaries": stored_criterion_summaries,
    }


def adjudicate_round_three_rows(
    node: Dict[str, Any],
    rows: List[Dict[str, Any]],
    attempt: int,
    discussion_digest: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Derive final worksheet treatment from the Chair's stored rubric.

    The Chair supplies criterion-level assessments and technical explanations;
    the backend applies the same deterministic category rule used in Round 2.
    Disagree candidates are recorded in the audit trace but cannot enter the
    credible worksheet. Matrix coverage is restored later as Not applicable.
    """
    retained: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    for index, raw_row in enumerate(rows, 1):
        if not isinstance(raw_row, dict):
            continue
        rubric = rubric_decision_record({
            "criterion_assessment": (
                raw_row.get("chair_criterion_assessment")
                or raw_row.get("criterion_assessment")
            ),
            "suggested_decision": (
                raw_row.get("chair_suggested_decision")
                or raw_row.get("suggested_decision")
            ),
        })
        decision = rubric["decision"]
        panel_basis = round_three_panel_basis(raw_row, discussion_digest)
        panel_classification = normalize_whitespace(
            panel_basis.get("panel_conclusion") or panel_basis.get("classification")
        )
        panel_review_complete = bool(panel_basis.get("review_complete", False))
        evidence_basis = (
            raw_row.get("chair_evidence_basis")
            or raw_row.get("evidence_basis")
            or []
        )
        documented_chair_basis = bool(
            sanitize_list(evidence_basis, 20)
            and normalize_whitespace(raw_row.get("chair_resolution"))
        )
        duplicate_of = ""
        matrix_key = normalize_whitespace(raw_row.get("matrix_key")).lower()
        cause = normalize_whitespace(raw_row.get("potential_cause"))
        for existing in retained:
            if matrix_key != normalize_whitespace(existing.get("matrix_key")).lower():
                continue
            if technical_statements_are_paraphrases(
                cause, existing.get("potential_cause"), "potential_cause"
            ):
                duplicate_of = normalize_whitespace(existing.get("row_id"))
                break

        if duplicate_of:
            treatment = "reject_duplicate"
        elif discussion_digest is not None and not panel_classification:
            treatment = "exclude_unlinked_from_round_2"
        elif decision == "disagree":
            treatment = "exclude_chair_core_criterion_failure"
        elif (
            panel_classification == "disagree"
            and decision in {"agree", "partially_agree"}
            and not documented_chair_basis
        ):
            treatment = "exclude_unsupported_panel_consensus_override"
        elif decision == "agree" and panel_classification == "disagree":
            treatment = "retain_after_documented_chair_panel_override"
        elif decision == "partially_agree" and panel_classification == "disagree":
            treatment = "retain_with_revision_after_documented_chair_panel_override"
        elif decision == "agree" and discussion_digest is not None and not panel_review_complete:
            treatment = (
                "retain_after_incomplete_panel_chair_adjudication"
            )
        elif decision == "partially_agree" and discussion_digest is not None and not panel_review_complete:
            treatment = (
                "retain_with_revision_after_incomplete_panel_chair_adjudication"
            )
        else:
            treatment = {
                "agree": "retain",
                "partially_agree": "retain_with_revision_or_open_action",
            }.get(decision, "incomplete")
        adjudication_id = f"{node.get('node_id', 'N-UNKNOWN')}-R3A-{index:03d}"
        source_review_ids = sanitize_list(raw_row.get("source_review_ids"), 80)
        source_proposal_ids = sanitize_list(raw_row.get("source_proposal_ids"), 40)
        final_status = (
            "Rejected" if not treatment.startswith("retain")
            else ("New" if source_proposal_ids and not source_review_ids else "Retained")
        )
        rejection_reason = ""
        if treatment == "reject_duplicate":
            rejection_reason = f"Semantic duplicate of {duplicate_of} within the same matrix item."
        elif not treatment.startswith("retain"):
            rejection_reason = normalize_whitespace(
                raw_row.get("rejection_reason") or raw_row.get("chair_resolution") or treatment
            )
        trace_record = {
            "adjudication_id": adjudication_id,
            "node_id": node.get("node_id", ""),
            "node_name": node.get("node_name", ""),
            "row_id": raw_row.get("row_id", ""),
            "matrix_key": raw_row.get("matrix_key", ""),
            "guideword": raw_row.get("guideword", ""),
            "parameter": raw_row.get("parameter", ""),
            "deviation": raw_row.get("deviation", ""),
            "potential_cause": raw_row.get("potential_cause", ""),
            "source_review_ids": source_review_ids,
            "source_proposal_ids": source_proposal_ids,
            "criterion_assessment": rubric["criterion_assessment"],
            "round_2_panel_basis": panel_basis,
            "chair_suggested_decision": rubric["agent_suggested_decision"],
            "chair_decision": decision,
            "decision_overridden_by_rubric": rubric["decision_overridden_by_rubric"],
            "decision_method": "deterministic_scenario_validity_rubric",
            "worksheet_treatment": treatment,
            "final_status": final_status,
            "rejection_reason": rejection_reason,
            "duplicate_of_row_id": duplicate_of,
            "chair_resolution": normalize_whitespace(raw_row.get("chair_resolution")),
            "chair_evidence_basis": sanitize_list(evidence_basis, 20),
            "chair_revision_summary": normalize_whitespace(
                raw_row.get("chair_revision_summary")
            ),
            "review_status": "completed" if decision else "incomplete_rubric",
            "review_attempt": attempt,
        }
        trace.append(trace_record)
        if duplicate_of:
            for existing in retained:
                if normalize_whitespace(existing.get("row_id")) != duplicate_of:
                    continue
                existing["source_review_ids"] = unique_strings(
                    sanitize_list(existing.get("source_review_ids"), 80)
                    + source_review_ids
                )
                existing["source_proposal_ids"] = unique_strings(
                    sanitize_list(existing.get("source_proposal_ids"), 40)
                    + source_proposal_ids
                )
                break
        if not treatment.startswith("retain"):
            continue
        row = dict(raw_row)
        row.update({
            "chair_adjudication_id": adjudication_id,
            "chair_criterion_assessment": rubric["criterion_assessment"],
            "round_2_panel_basis": panel_basis,
            "chair_suggested_decision": rubric["agent_suggested_decision"],
            "chair_decision": decision,
            "chair_decision_overridden_by_rubric": rubric[
                "decision_overridden_by_rubric"
            ],
            "chair_decision_method": "deterministic_scenario_validity_rubric",
            "chair_final_treatment": treatment,
            "final_status": final_status,
            "rejection_reason": "",
            "duplicate_of_row_id": "",
            "source_review_ids": source_review_ids,
            "source_proposal_ids": source_proposal_ids,
            "chair_evidence_basis": sanitize_list(evidence_basis, 20),
            "chair_revision_summary": normalize_whitespace(
                raw_row.get("chair_revision_summary")
            ),
            "chair_review_status": "completed",
            "chair_review_attempt": attempt,
            "chair_authorized": True,
        })
        retained.append(row)
    return retained, trace


def build_round_three_adjudication_summary(
    trace: List[Dict[str, Any]],
    expected_candidate_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize single-Chair finalization without misusing agreement metrics."""
    decision_distribution = {
        decision: 0 for decision in ("agree", "partially_agree", "disagree")
    }
    treatment_distribution: Dict[str, int] = {}
    criterion_distribution = {
        criterion: {rating: 0 for rating in ("pass", "revision_required", "fail")}
        for criterion in ROUND_TWO_REVIEW_CRITERIA
    }
    completed = 0
    overrides = 0
    panel_classification_distribution: Dict[str, int] = {}
    for item in trace:
        if not isinstance(item, dict):
            continue
        decision = normalized_round_two_decision(item.get("chair_decision"))
        if decision:
            decision_distribution[decision] += 1
            completed += 1
        treatment = normalize_whitespace(item.get("worksheet_treatment")) or "unspecified"
        treatment_distribution[treatment] = treatment_distribution.get(treatment, 0) + 1
        panel_classification = normalize_whitespace(
            (item.get("round_2_panel_basis") or {}).get("panel_conclusion")
            or (item.get("round_2_panel_basis") or {}).get("classification")
        ) or "not_supplied"
        panel_classification_distribution[panel_classification] = (
            panel_classification_distribution.get(panel_classification, 0) + 1
        )
        overrides += int(bool(item.get("decision_overridden_by_rubric", False)))
        criteria = item.get("criterion_assessment") or {}
        if isinstance(criteria, dict):
            for criterion in ROUND_TWO_REVIEW_CRITERIA:
                assessment = criteria.get(criterion) or {}
                rating = normalized_review_criterion_rating(assessment.get("rating"))
                if rating:
                    criterion_distribution[criterion][rating] += 1
    total = len(trace)
    expected = max(total, int(expected_candidate_count or total))
    return {
        "expected_candidate_count": expected,
        "adjudication_count": total,
        "completed_adjudication_count": completed,
        "unreturned_candidate_count": max(0, expected - total),
        "adjudication_coverage_rate": completed / expected if expected else 1.0,
        "decision_distribution": decision_distribution,
        "worksheet_treatment_distribution": treatment_distribution,
        "round_2_panel_classification_distribution": panel_classification_distribution,
        "criterion_rating_distribution": criterion_distribution,
        "rubric_override_count": overrides,
        "agreement_metrics_applicable": False,
        "agreement_metrics_note": (
            "Pairwise agreement and Fleiss' kappa are not defined for Round 3 because "
            "the Panel Chair is a single final adjudicator."
        ),
    }


def normalize_round_three_escalated_proposals(
    node: Dict[str, Any],
    proposals: Any,
) -> List[Dict[str, Any]]:
    """Normalize and de-duplicate Chair proposals before targeted panel review.

    Model retries may repeat the same proposal with a different proposal ID.
    Those retries are one technical candidate, not independent panel-review
    assignments.  Their IDs remain in ``source_proposal_ids`` for traceability.
    """
    output: List[Dict[str, Any]] = []
    output_by_fingerprint: Dict[str, Dict[str, Any]] = {}
    if not isinstance(proposals, list):
        return output
    for index, item in enumerate(proposals, 1):
        if not isinstance(item, dict):
            continue
        rubric = rubric_decision_record({
            "criterion_assessment": item.get("chair_criterion_assessment"),
            "suggested_decision": item.get("chair_suggested_decision"),
        })
        proposal_id = normalize_whitespace(item.get("proposal_id")) or (
            f"{node.get('node_id', 'N-UNKNOWN')}-R3P-{index:03d}"
        )
        normalized = {
            "proposal_id": proposal_id,
            "source_proposal_ids": [proposal_id],
            "node_id": node.get("node_id", ""),
            "node_name": node.get("node_name", ""),
            "guideword": normalized_guideword(item.get("guideword")),
            "parameter": normalize_whitespace(item.get("parameter")),
            "matrix_key": normalize_whitespace(item.get("matrix_key")),
            "deviation": normalize_whitespace(item.get("deviation")),
            "potential_cause": normalize_whitespace(item.get("potential_cause")),
            "consequence": sanitize_list(item.get("consequence"), 12),
            "safeguard": sanitize_list(item.get("safeguard"), 12),
            "necessary_action": sanitize_list(item.get("necessary_action"), 12),
            "criterion_assessment": rubric["criterion_assessment"],
            "chair_suggested_decision": rubric["agent_suggested_decision"],
            "chair_derived_decision": rubric["decision"],
            "decision_overridden_by_rubric": rubric[
                "decision_overridden_by_rubric"
            ],
            "chair_evidence_basis": sanitize_list(
                item.get("chair_evidence_basis"), 20
            ),
            "proposal_reason": normalize_whitespace(item.get("proposal_reason")),
            "required_next_step": "Full-panel review before final worksheet inclusion",
            "panel_review_status": "not_reviewed_by_panel",
            "worksheet_eligible": False,
        }
        fingerprint = "|".join([
            normalize_whitespace(normalized.get("matrix_key")).lower(),
            normalize_whitespace(normalized.get("guideword")).lower(),
            normalize_whitespace(normalized.get("parameter")).lower(),
            normalize_whitespace(normalized.get("deviation")).lower(),
            normalize_whitespace(normalized.get("potential_cause")).lower(),
            " ".join(sanitize_list(normalized.get("consequence"), 12)).lower(),
        ])
        existing = output_by_fingerprint.get(fingerprint)
        if existing is not None:
            existing["source_proposal_ids"] = unique_strings(
                sanitize_list(existing.get("source_proposal_ids"), 40)
                + [proposal_id]
            )
            existing["chair_evidence_basis"] = unique_strings(
                sanitize_list(existing.get("chair_evidence_basis"), 20)
                + sanitize_list(normalized.get("chair_evidence_basis"), 20)
            )
            continue
        output_by_fingerprint[fingerprint] = normalized
        output.append(normalized)
    return output


def review_round_three_escalated_proposals(
    payload: Dict[str, Any],
    node: Dict[str, Any],
    proposals: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run the required targeted full-panel review for Chair-originated proposals."""
    if not proposals:
        return []
    reviewers = [
        ("panel_chair", "Panel Chair"),
        ("process_engineer", "Process Engineer"),
        ("instrumentation_control", "Instrumentation & Control Engineer"),
        ("safety_consequence", "Safety Engineer"),
        ("maintenance_reliability", "Maintenance Engineer"),
        ("mechanical_engineer", "Mechanical Engineer"),
    ]
    futures = {
        EXECUTOR.submit(
            run_round_two_new_scenario_review_agent,
            agent_id,
            label,
            payload,
            node,
            proposals,
            review_rows,
        ): label
        for agent_id, label in reviewers
    }
    outputs: List[Dict[str, Any]] = []
    for future, label in futures.items():
        try:
            reviews = future.result()
        except Exception as error:
            reviews = [
                missing_round_two_review(
                    proposal,
                    ROUND_TWO_BATCH_RETRY_ATTEMPTS,
                    id_field="proposal_id",
                )
                for proposal in proposals
            ]
            outputs.append({
                "agent_name": label,
                "new_scenario_reviews": reviews,
                "review_error": str(error),
            })
            continue
        outputs.append({"agent_name": label, "new_scenario_reviews": reviews})
    deliberations = aggregate_round_two_new_scenario_decisions(proposals, outputs)
    for item in deliberations:
        item["review_stage"] = "round_3_targeted_panel_review"
    return deliberations


def refine_round_two(
    payload: Dict[str, Any], node: Dict[str, Any],
    draft_rows: List[Dict[str, Any]], challenges: List[Dict[str, Any]],
    review_rows: Optional[List[Dict[str, Any]]] = None,
    new_scenario_deliberations: Optional[List[Dict[str, Any]]] = None,
    discussion_digest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Round 3 as small, ID-locked Chair adjudication batches.

    A valid decision from one batch is never discarded because another batch
    omitted a candidate. Only missing/incomplete candidates are retried. Any
    candidate still incomplete after three attempts remains excluded with an
    explicit audit record, while completed Chair decisions can still produce
    the final worksheet.
    """
    discussion_digest = discussion_digest or build_round_two_chair_digest(
        challenges, new_scenario_deliberations
    )
    candidates = round_three_candidate_universe(
        node, draft_rows, new_scenario_deliberations
    )
    expected_candidate_count = len(candidates)
    if expected_candidate_count == 0:
        return {
            "round_3_chair_summary": "No Round 2 scenario candidates required Chair adjudication.",
            "round_3_chair_synthesis": [],
            "revised_rows": [],
            "round_3_adjudication_trace": [],
            "round_3_adjudication_summary": build_round_three_adjudication_summary([], 0),
            "round_3_escalated_proposals": [],
            "round_3_new_proposal_deliberations": [],
            "round_3_review_attempts": 0,
            "round_3_model_call_count": 0,
            "round_3_batch_count": 0,
            "round_3_missing_candidate_ids": [],
            "chair_finalization_failed": False,
            "chair_finalization_incomplete": False,
            "raw_text": "",
        }

    candidate_by_id = {
        normalize_whitespace(row.get("row_id")): row for row in candidates
    }
    completed_by_id: Dict[str, Dict[str, Any]] = {}
    attempt_by_id: Dict[str, int] = {}
    summary_items: List[str] = []
    synthesis_items: List[str] = []
    escalated_proposals: List[Dict[str, Any]] = []
    response_hashes: List[str] = []
    last_error_by_id: Dict[str, str] = {}
    model_call_count = 0
    batches = [
        candidates[index:index + ROUND_THREE_BATCH_SIZE]
        for index in range(0, len(candidates), ROUND_THREE_BATCH_SIZE)
    ]

    def returned_candidate_id(
        row: Dict[str, Any], requested_ids: Set[str]
    ) -> str:
        row_id = normalize_whitespace(row.get("row_id"))
        if row_id in requested_ids:
            return row_id
        linked_ids = (
            sanitize_list(row.get("source_review_ids"), 80)
            + sanitize_list(row.get("source_proposal_ids"), 40)
        )
        matches = [item for item in linked_ids if item in requested_ids]
        return matches[0] if len(set(matches)) == 1 else ""

    for batch_index, batch in enumerate(batches, 1):
        pending_ids = [normalize_whitespace(row.get("row_id")) for row in batch]
        for attempt in range(1, ROUND_TWO_BATCH_RETRY_ATTEMPTS + 1):
            pending_ids = [item for item in pending_ids if item not in completed_by_id]
            if not pending_ids:
                break
            requested_rows = [candidate_by_id[item] for item in pending_ids]
            requested_id_set = set(pending_ids)
            batch_digest = round_three_digest_for_candidates(
                discussion_digest, requested_rows
            )
            prompt = chair_refine_prompt(
                node,
                requested_rows,
                batch_digest,
                [],
                payload.get("documents", {}),
            ) + f"""

ROUND 3 BATCH CONTROL — MANDATORY:
- This is batch {batch_index} of {len(batches)}.
- Return exactly one revised_rows object for every ID below, in this order.
- Copy each row_id exactly. Do not renumber, omit, merge, or invent IDs.
- Complete all four criterion assessments, Chair evidence, Chair resolution, and any required revision summary for every returned row.
- Do not repeat candidates from earlier batches.
Required candidate IDs: {json.dumps(pending_ids, ensure_ascii=True)}
""".strip()
            try:
                raw = run_model(
                    "panel_chair",
                    payload["agents"]["panel_chair"],
                    payload["api_keys"],
                    system_prompt_for("panel_chair"),
                    prompt,
                    trace={
                        "job_id": payload.get("_job_id", ""),
                        "stage": "round_3_chair_finalization",
                        "node_id": node.get("node_id", ""),
                        "round": "Round 3",
                        "attempt_role": f"chair_batch_{batch_index}_attempt_{attempt}",
                        "coverage_attempt": attempt,
                        "batch_index": batch_index,
                        "batch_count": len(batches),
                        "required_candidate_count": len(pending_ids),
                    },
                )
                model_call_count += 1
                response_hashes.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
                parsed_batch = extract_json_object(raw)
                summary = normalize_whitespace(
                    parsed_batch.get("round_3_chair_summary")
                )
                if summary:
                    summary_items.append(summary)
                synthesis_items.extend(
                    sanitize_list(parsed_batch.get("round_3_chair_synthesis"), 20)
                )
                if isinstance(parsed_batch.get("round_3_escalated_proposals"), list):
                    escalated_proposals.extend(
                        item for item in parsed_batch["round_3_escalated_proposals"]
                        if isinstance(item, dict)
                    )
                returned_rows = (
                    parsed_batch.get("revised_rows")
                    if isinstance(parsed_batch.get("revised_rows"), list)
                    else []
                )
                for returned in returned_rows:
                    if not isinstance(returned, dict):
                        continue
                    candidate_id = returned_candidate_id(returned, requested_id_set)
                    if not candidate_id:
                        continue
                    merged = dict(candidate_by_id[candidate_id])
                    merged.update(returned)
                    merged["row_id"] = candidate_id
                    merged["source_review_ids"] = unique_strings(
                        sanitize_list(candidate_by_id[candidate_id].get("source_review_ids"), 80)
                        + sanitize_list(returned.get("source_review_ids"), 80)
                    )
                    merged["source_proposal_ids"] = unique_strings(
                        sanitize_list(candidate_by_id[candidate_id].get("source_proposal_ids"), 40)
                        + sanitize_list(returned.get("source_proposal_ids"), 40)
                    )
                    if not round_three_row_has_complete_rubric(merged):
                        last_error_by_id[candidate_id] = (
                            "Chair response lacked a complete four-check assessment, "
                            "evidence basis, resolution, or required revision summary."
                        )
                        continue
                    completed_by_id[candidate_id] = merged
                    attempt_by_id[candidate_id] = attempt
            except Exception as error:
                model_call_count += 1
                for candidate_id in pending_ids:
                    last_error_by_id[candidate_id] = str(error)

    ordered_completed = [
        completed_by_id[candidate_id]
        for candidate_id in candidate_by_id
        if candidate_id in completed_by_id
    ]
    missing_ids = [
        candidate_id for candidate_id in candidate_by_id
        if candidate_id not in completed_by_id
    ]
    adjudicated_rows, adjudication_trace = adjudicate_round_three_rows(
        node,
        ordered_completed,
        max(attempt_by_id.values(), default=0),
        discussion_digest,
    )
    for item in adjudication_trace:
        item["review_attempt"] = attempt_by_id.get(
            normalize_whitespace(item.get("row_id")), item.get("review_attempt", 0)
        )
    for index, candidate_id in enumerate(missing_ids, 1):
        source = candidate_by_id[candidate_id]
        adjudication_trace.append({
            "adjudication_id": f"{node.get('node_id', 'N-UNKNOWN')}-R3F-{index:03d}",
            "node_id": node.get("node_id", ""),
            "node_name": node.get("node_name", ""),
            "row_id": candidate_id,
            "matrix_key": source.get("matrix_key", ""),
            "guideword": source.get("guideword", ""),
            "parameter": source.get("parameter", ""),
            "deviation": source.get("deviation", ""),
            "potential_cause": source.get("potential_cause", ""),
            "source_review_ids": sanitize_list(source.get("source_review_ids"), 80),
            "source_proposal_ids": sanitize_list(source.get("source_proposal_ids"), 40),
            "criterion_assessment": {},
            "chair_decision": "",
            "worksheet_treatment": "exclude_unadjudicated_after_three_attempts",
            "final_status": "Rejected",
            "rejection_reason": last_error_by_id.get(
                candidate_id,
                "Chair omitted the candidate after three targeted attempts.",
            ),
            "duplicate_of_row_id": "",
            "review_status": "missing_after_3_attempts",
            "review_attempt": ROUND_TWO_BATCH_RETRY_ATTEMPTS,
            "failure_reason": last_error_by_id.get(
                candidate_id,
                "Chair omitted the candidate after three targeted attempts.",
            ),
        })

    chair_finalization_failed = bool(expected_candidate_count and not completed_by_id)
    chair_finalization_incomplete = bool(missing_ids)
    if chair_finalization_failed:
        chair_summary = (
            "The Chair did not complete an evidence-grounded four-check decision "
            "for any Round 3 candidate after three targeted attempts. No scenario "
            "was authorized for the final worksheet."
        )
    else:
        chair_summary = " ".join(unique_strings(summary_items))
        coverage_statement = (
            f"The Chair completed {len(completed_by_id)} of "
            f"{expected_candidate_count} candidate adjudications."
        )
        if missing_ids:
            coverage_statement += (
                f" {len(missing_ids)} candidate(s) remained incomplete after three "
                "targeted attempts and were excluded rather than treated as agreement."
            )
        chair_summary = normalize_whitespace(
            f"{coverage_statement} {chair_summary}"
        )

    normalized_escalated = normalize_round_three_escalated_proposals(
        node, escalated_proposals
    )
    round_three_proposal_deliberations = review_round_three_escalated_proposals(
        payload, node, normalized_escalated, candidates
    )
    for proposal_index, deliberation in enumerate(round_three_proposal_deliberations, 1):
        proposal = deliberation.get("proposal") or {}
        proposal_id = normalize_whitespace(deliberation.get("proposal_id"))
        source_proposal_ids = unique_strings(
            sanitize_list(proposal.get("source_proposal_ids"), 40)
            or [proposal_id]
        )
        panel_conclusion = normalized_round_two_decision(
            deliberation.get("panel_conclusion")
        )
        review_complete = bool(deliberation.get("panel_review_complete", False))
        chair_proposal_decision = normalized_round_two_decision(
            proposal.get("chair_derived_decision")
        )
        chair_proposal_valid = chair_proposal_decision in {
            "agree", "partially_agree"
        }
        duplicate_of = ""
        for existing in adjudicated_rows:
            if normalize_whitespace(existing.get("matrix_key")).lower() != normalize_whitespace(
                proposal.get("matrix_key")
            ).lower():
                continue
            if technical_statements_are_paraphrases(
                proposal.get("potential_cause"), existing.get("potential_cause"), "potential_cause"
            ):
                duplicate_of = normalize_whitespace(existing.get("row_id"))
                break
        include = (
            chair_proposal_valid
            and review_complete
            and panel_conclusion in {"agree", "partially_agree"}
            and not duplicate_of
        )
        treatment = (
            "include_new_after_targeted_panel_review" if include
            else ("reject_duplicate" if duplicate_of else "reject_new_after_targeted_panel_review")
        )
        rejection_reason = ""
        if duplicate_of:
            rejection_reason = f"Semantic duplicate of {duplicate_of} within the same matrix item."
        elif not chair_proposal_valid:
            rejection_reason = (
                "The Chair proposal did not contain a complete passing or revisable "
                "four-check scenario-validity assessment."
            )
        elif not include:
            rejection_reason = (
                "Targeted panel review was incomplete."
                if not review_complete
                else f"Targeted panel conclusion was {panel_conclusion or 'not returned'}."
            )
        adjudication_id = f"{node.get('node_id', 'N-UNKNOWN')}-R3P-{proposal_index:03d}"
        proposal_trace = {
            "adjudication_id": adjudication_id,
            "node_id": node.get("node_id", ""),
            "node_name": node.get("node_name", ""),
            "row_id": proposal_id,
            "matrix_key": proposal.get("matrix_key", ""),
            "guideword": proposal.get("guideword", ""),
            "parameter": proposal.get("parameter", ""),
            "deviation": proposal.get("deviation", ""),
            "potential_cause": proposal.get("potential_cause", ""),
            "source_review_ids": [],
            "source_proposal_ids": source_proposal_ids,
            "criterion_assessment": proposal.get("criterion_assessment", {}),
            "round_2_panel_basis": {
                "classification": panel_conclusion,
                "panel_conclusion": panel_conclusion,
                "conclusion_basis": deliberation.get("panel_conclusion_basis", ""),
                "decision_counts": deliberation.get("decision_counts", {}),
                "review_complete": review_complete,
            },
            "chair_decision": chair_proposal_decision,
            "decision_method": "chair_proposal_plus_targeted_full_panel_review",
            "worksheet_treatment": treatment,
            "final_status": "New" if include else "Rejected",
            "rejection_reason": rejection_reason,
            "duplicate_of_row_id": duplicate_of,
            "chair_resolution": proposal.get("proposal_reason", ""),
            "chair_evidence_basis": proposal.get("chair_evidence_basis", []),
            "review_status": "completed" if review_complete else "incomplete_panel_review",
            "review_attempt": ROUND_TWO_BATCH_RETRY_ATTEMPTS,
        }
        adjudication_trace.append(proposal_trace)
        proposal.update({
            "panel_review_status": "completed" if review_complete else "incomplete",
            "panel_conclusion": panel_conclusion,
            "panel_conclusion_basis": deliberation.get("panel_conclusion_basis", ""),
            "panel_decision_counts": deliberation.get("decision_counts", {}),
            "worksheet_eligible": include,
            "final_status": "New" if include else "Rejected",
            "rejection_reason": rejection_reason,
            "duplicate_of_row_id": duplicate_of,
        })
        if include:
            new_row = scenario_to_row(proposal, node, len(adjudicated_rows) + 1, "Panel Chair")
            new_row.update({
                "row_id": proposal_id,
                "source_review_ids": [],
                "source_proposal_ids": source_proposal_ids,
                "round_3_candidate_origin": "round_3_chair_proposal",
                "chair_adjudication_id": adjudication_id,
                "chair_authorized": True,
                "chair_decision": chair_proposal_decision,
                "chair_final_treatment": treatment,
                "final_status": "New",
                "rejection_reason": "",
                "duplicate_of_row_id": "",
            })
            adjudicated_rows.append(new_row)

    parsed: Dict[str, Any] = {
        "round_3_chair_summary": chair_summary,
        "round_3_chair_synthesis": unique_strings(synthesis_items),
        "revised_rows": [
            row for row in adjudicated_rows
            if isinstance(row, dict) and not row_is_not_applicable(row)
        ],
        "round_3_adjudication_trace": adjudication_trace,
        "round_3_review_attempts": max(attempt_by_id.values(), default=(3 if missing_ids else 0)),
        "round_3_model_call_count": model_call_count,
        "round_3_batch_count": len(batches),
        "round_3_missing_candidate_ids": missing_ids,
        "round_3_escalated_proposals": normalized_escalated,
        "round_3_new_proposal_deliberations": round_three_proposal_deliberations,
        "chair_finalization_failed": chair_finalization_failed,
        "chair_finalization_incomplete": chair_finalization_incomplete,
        "raw_response_hashes": response_hashes,
        "raw_text": "",
    }
    parsed["round_3_adjudication_summary"] = build_round_three_adjudication_summary(
        adjudication_trace, expected_candidate_count + len(normalized_escalated)
    )
    return parsed


def scribe_rows(
    payload: Dict[str, Any], node: Dict[str, Any],
    revised_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # The Chair owns technical content. Formatting is deterministic so a
    # Scribe model cannot add/remove scenarios or rewrite safeguards/actions.
    rows = [
        sanitize_row(row, node, index + 1)
        for index, row in enumerate(revised_rows)
        if isinstance(row, dict)
    ]
    rows = [
        row for row in rows
        if row["guideword"] and not row_is_not_applicable(row)
    ]
    return [
        sanitize_row(row, node, index + 1)
        for index, row in enumerate(rows)
        if isinstance(row, dict) and not row_is_not_applicable(row)
    ]


def consolidate_worksheet_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return one presentation row per node and guideword-matrix item.

    The analysis pipeline deliberately retains one internal row per distinct
    cause. Word, Excel, and web output should not repeat the guideword row,
    however. This projection combines the atomic rows into bullet-ready,
    de-duplicated cell values while retaining their identifiers in audit-only
    fields on the consolidated row.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    group_order: List[Tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        node_id = normalize_whitespace(row.get("node_id")) or "N-UNKNOWN"
        matrix_key = normalize_whitespace(row.get("matrix_key")).lower()
        if not matrix_key:
            matrix_key = normalize_whitespace(row.get("parent_row_id")).lower()
        if not matrix_key:
            matrix_key = "::".join([
                normalize_whitespace(row.get("guideword")).lower(),
                normalize_whitespace(row.get("parameter")).lower(),
                normalize_whitespace(row.get("deviation") or row.get("effect")).lower(),
            ])
        key = (node_id, matrix_key)
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(row)

    aliases = {
        "deviation": ("deviation", "effect"),
        "potential_cause": ("potential_cause", "cause"),
        "consequence": ("consequence",),
        "safeguard": ("safeguard", "safeguards"),
        "necessary_action": ("necessary_action", "action_required"),
        "safeguard_evidence": ("safeguard_evidence",),
    }

    def unique_cell(group: List[Dict[str, Any]], field_names: Tuple[str, ...]) -> str:
        """Collect semantically distinct, bullet-ready statements for one column."""
        values: List[str] = []
        for atomic_row in group:
            value: Any = ""
            for field_name in field_names:
                if atomic_row.get(field_name) not in (None, "", []):
                    value = atomic_row.get(field_name)
                    break
            cell_items = row_text_items(value, 30)
            for cell_item in cell_items or ["Not identified"]:
                # A semicolon may join two clauses within one technical
                # statement. Only explicit list/newline structure creates a
                # new worksheet bullet; punctuation must not fragment prose.
                normalized_item = normalize_whitespace(cell_item)
                if normalized_item:
                    values.append(normalized_item)
        field = field_names[0] if field_names else ""
        unique_values = semantic_dedupe_worksheet_items(values, field, 30)
        placeholder_values = {
            "not identified", "not applicable", "none", "n/a", "na",
            "action not identified", "no action required",
        }
        if any(value.lower().rstrip(".") not in placeholder_values for value in unique_values):
            unique_values = [
                value for value in unique_values
                if value.lower().rstrip(".") not in placeholder_values
            ]
        return "\n".join(unique_values)

    def combined_deviation(group: List[Dict[str, Any]]) -> str:
        values: List[str] = []
        for atomic_row in group:
            values.extend(row_text_items(atomic_row.get("deviation") or atomic_row.get("effect"), 12))
        return "\n".join(semantic_dedupe_worksheet_items(values, "deviation", 12))

    def combined_list(group: List[Dict[str, Any]], field_name: str) -> List[str]:
        values: List[str] = []
        for atomic_row in group:
            raw = atomic_row.get(field_name)
            if isinstance(raw, list):
                values.extend(normalize_whitespace(item) for item in raw)
            else:
                values.extend(row_text_items(raw, 50))
        return unique_strings(values)

    consolidated: List[Dict[str, Any]] = []
    for key in group_order:
        group = grouped[key]
        first = dict(group[0])
        group_row_id = (
            normalize_whitespace(first.get("parent_row_id"))
            or re.sub(r"-C\d+$", "", normalize_whitespace(first.get("row_id")), flags=re.IGNORECASE)
            or normalize_whitespace(first.get("row_id"))
        )
        first["row_id"] = group_row_id
        first["parent_row_id"] = group_row_id
        first["cause_id"] = ""
        first["cause_index"] = 1
        first["group_size"] = len(group)
        first["atomic_scenario_count"] = len(group)
        first["atomic_row_ids"] = unique_strings([
            normalize_whitespace(item.get("row_id")) for item in group
        ])
        first["cause_ids"] = unique_strings([
            normalize_whitespace(item.get("cause_id")) for item in group
        ])
        first["atomic_chains"] = [{
            "row_id": item.get("row_id", ""),
            "potential_cause": item.get("potential_cause") or item.get("cause") or "Not identified",
            "consequence": item.get("consequence") or "Not identified",
            "safeguard": item.get("safeguard") or item.get("safeguards") or "Not identified",
            "necessary_action": item.get("necessary_action") or item.get("action_required") or "Not identified",
        } for item in group]
        first["deviation"] = combined_deviation(group)
        for output_field, field_names in aliases.items():
            if output_field != "deviation":
                first[output_field] = unique_cell(group, field_names)
        first["cause"] = first["potential_cause"]
        first["effect"] = first["deviation"]
        first["safeguards"] = first["safeguard"]
        first["action_required"] = first["necessary_action"]
        for list_field in (
            "source_review_ids", "source_proposal_ids", "source_agents",
            "source_references", "source_round_2_decisions",
        ):
            first[list_field] = combined_list(group, list_field)
        consolidated.append(first)
    return consolidated


def build_action_register(worksheet: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    action_index: Dict[Tuple[str, str], int] = {}
    for row in worksheet:
        raw_actions = row_text_items(
            row.get("necessary_action") or row.get("action_required"), 20
        )
        for action in unique_strings(raw_actions):
            action = normalize_whitespace(action)
            if not action or action.lower().rstrip(".") in {
                "not identified", "action not identified", "not applicable", "n/a", "none",
                "not required", "no action required", "no additional action required",
            }:
                continue
            action_key = (
                normalize_whitespace(row.get("node_id")).lower(),
                action.lower().rstrip("."),
            )
            source_row_id = normalize_whitespace(row.get("row_id"))
            if action_key in action_index:
                existing = actions[action_index[action_key]]
                existing["source_row_ids"] = unique_strings(
                    list(existing.get("source_row_ids") or []) + [source_row_id]
                )
                existing["source_cause_ids"] = unique_strings(
                    list(existing.get("source_cause_ids") or [])
                    + [normalize_whitespace(row.get("cause_id"))]
                )
                continue
            actions.append({
                "action_id": f"A-{len(actions) + 1:03d}",
                "row_id": row.get("row_id", ""),
                "parent_row_id": row.get("parent_row_id", ""),
                "cause_id": row.get("cause_id", ""),
                "node_id": row.get("node_id", ""),
                "node_name": row.get("node_name", ""),
                "guideword": row.get("guideword", ""),
                "deviation": row.get("deviation", ""),
                "potential_cause": row.get("potential_cause", row.get("cause", "")),
                "consequence": row.get("consequence", ""),
                "safeguard": row.get("safeguard", row.get("safeguards", "")),
                "necessary_action": action,
                "source_references": row.get("source_references", []),
                "action_required": action,
                "responsible_person": row.get("responsible_person", "Discipline owner"),
                "target_date": row.get("target_date", "To be assigned"),
                "action_status": row.get("action_status", "Open"),
                "source_row_ids": [source_row_id] if source_row_id else [],
                "source_cause_ids": [normalize_whitespace(row.get("cause_id"))] if row.get("cause_id") else [],
            })
            action_index[action_key] = len(actions) - 1
    return actions


def build_final_report(payload: Dict[str, Any], nodes, worksheet, actions) -> Dict[str, Any]:
    try:
        raw = run_model(
            "scribe_report", payload["agents"]["scribe_report"], payload["api_keys"],
            system_prompt_for("scribe_report"), report_prompt(nodes, worksheet, actions),
            trace={
                "job_id": payload.get("_job_id", ""),
                "stage": "final_report",
                "round": "Report",
                "attempt_role": "report_summary",
            },
        )
        parsed = extract_json_object(raw)
    except Exception as error:
        parsed = {
            "executive_summary": f"HAZOP completed, but the final report step failed: {error}",
            "key_findings": [],
            "assumptions_and_limitations": [],
        }
    parsed["node_count"] = len(nodes)
    parsed["worksheet_row_count"] = len(worksheet)
    parsed["action_count"] = len(actions)
    parsed["key_findings"] = sanitize_list(parsed.get("key_findings"), 12)
    parsed["assumptions_and_limitations"] = sanitize_assumptions(parsed.get("assumptions_and_limitations"), 12)
    parsed["source_references"] = sanitize_source_references(parsed.get("source_references"), 20)
    if not normalize_whitespace(parsed.get("executive_summary", "")):
        parsed["executive_summary"] = (
            f"HAZOP completed for {len(nodes)} node(s), "
            f"producing {len(worksheet)} worksheet rows and {len(actions)} action items."
        )
    return parsed


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG image")
    index = 2
    while index < len(data) - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            break
        length = int.from_bytes(data[index:index + 2], "big")
        if length < 2:
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if index + 7 > len(data):
                break
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            if width > 0 and height > 0:
                return width, height
        index += length
    raise ValueError("JPEG dimensions not found")


def png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG image")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def office_images_from_job(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    images = (((job.get("result") or {}).get("report_inputs") or {}).get("pid_images") or [])
    output: List[Dict[str, Any]] = []
    for index, data_url in enumerate(images[:3], 1):
        if not isinstance(data_url, str):
            continue
        match = re.match(r"data:(image/[a-zA-Z0-9.+-]+);base64,(.+)", data_url, re.DOTALL)
        if not match:
            continue
        media_type = match.group(1).lower()
        try:
            data = base64.b64decode(match.group(2), validate=False)
            if media_type in {"image/jpeg", "image/jpg"}:
                width, height = jpeg_dimensions(data)
                ext = "jpg"
                content_type = "image/jpeg"
            elif media_type == "image/png":
                width, height = png_dimensions(data)
                ext = "png"
                content_type = "image/png"
            else:
                continue
        except Exception:
            continue
        output.append({
            "title": f"Submitted P&ID Image {index}",
            "data": data,
            "width": width,
            "height": height,
            "ext": ext,
            "content_type": content_type,
        })
    return output


def add_report_section(lines: List[str], title: str, body: Any, limit: int = 60000) -> None:
    lines.extend(["", title.upper(), "-" * min(70, len(title) + 8)])
    if isinstance(body, str):
        text = body
    else:
        text = json.dumps(json_safe(body), ensure_ascii=True, indent=2)
    for raw_line in clamp_text(text, limit).splitlines():
        raw_line = raw_line.rstrip()
        if not raw_line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=105, replace_whitespace=False) or [""])


def report_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(report_text(item) for item in value if report_text(item))
    if isinstance(value, dict):
        return "; ".join(f"{k}: {report_text(v)}" for k, v in value.items() if report_text(v))
    return normalize_whitespace(str(value or ""))


def report_bullets(value: Any, max_items: int = 8) -> List[str]:
    if isinstance(value, list):
        return [normalize_whitespace(report_text(item)) for item in value if normalize_whitespace(report_text(item))][:max_items]
    # Split on the raw string first — report_text() collapses newlines via
    # normalize_whitespace, so calling it before splitting destroys the line
    # breaks that separate bullet items and leaves everything as one item.
    text = report_text(value) if isinstance(value, dict) else str(value or "")
    if not normalize_whitespace(text):
        return []
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]*\|[ \t\f\v]*", "\n", text)
    text = re.sub(r"(?:^|\n)[ \t\f\v]*\d+[.)][ \t\f\v]+", "\n", text)
    items = [normalize_whitespace(item) for item in re.split(r"\n+", text) if normalize_whitespace(item)]
    return items[:max_items]


def add_wrapped_line(lines: List[str], text: str, indent: str = "") -> None:
    wrapped = textwrap.wrap(normalize_whitespace(text), width=max(40, 100 - len(indent))) or [""]
    for index, part in enumerate(wrapped):
        lines.append(f"{indent}{part}" if index == 0 else f"{indent}  {part}")


def add_report_bullet_block(lines: List[str], label: str, value: Any, max_items: int = 6) -> None:
    items = report_bullets(value, max_items)
    if not items:
        lines.append(f"  - {label}: Not identified")
        return
    lines.append(f"  - {label}:")
    for item in items:
        add_wrapped_line(lines, f"* {item}", "      ")


def short_report_text(value: Any, limit: int = 90) -> str:
    text = report_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def table_lines(headers: List[str], rows: List[List[Any]], widths: List[int], max_rows: int = 80) -> List[str]:
    def cell(value: Any, width: int) -> str:
        text = short_report_text(value, width)
        return text.ljust(width)[:width]

    selected = rows[:max_rows]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    output = [border]
    output.append("| " + " | ".join(cell(header, width) for header, width in zip(headers, widths)) + " |")
    output.append(border)
    for row in selected:
        padded = list(row) + [""] * max(0, len(headers) - len(row))
        output.append("| " + " | ".join(cell(value, width) for value, width in zip(padded, widths)) + " |")
    output.append(border)
    if len(rows) > len(selected):
        output.append(f"... {len(rows) - len(selected)} additional row(s) omitted from this table view.")
    return output


def add_table(lines: List[str], headers: List[str], rows: List[List[Any]], widths: List[int], max_rows: int = 80) -> None:
    if not rows:
        lines.append("- None listed.")
        return
    lines.extend(table_lines(headers, rows, widths, max_rows))


def add_subsection(lines: List[str], title: str) -> None:
    lines.extend(["", f"## {title}"])


def add_subsubsection(lines: List[str], title: str) -> None:
    lines.extend(["", f"### {title}"])


def pid_items(parse: Dict[str, Any], *keys: str) -> List[Dict[str, Any]]:
    for key in keys:
        value = parse.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def add_pid_parse_report(lines: List[str], result: Dict[str, Any]) -> None:
    parse = result.get("pid_parse") or {}
    if not parse:
        lines.append("- No P&ID image parse was generated for this job.")
        return
    lines.extend([
        f"- Vision parse succeeded: {bool(parse.get('vision_parse_succeeded'))}",
        f"- Equipment items: {len(pid_items(parse, 'equipment'))}",
        f"- Piping lines: {len(pid_items(parse, 'lines'))}",
        f"- Instruments: {len(pid_items(parse, 'instruments_and_controls', 'instruments'))}",
        f"- Valves/relief devices: {len(pid_items(parse, 'valves_and_relief_devices', 'valves'))}",
    ])

    add_subsubsection(lines, "Equipment Table")
    add_table(
        lines,
        ["Tag", "Type", "Function/Description", "Confidence"],
        [
            [
                item.get("tag") or item.get("name"),
                item.get("type"),
                item.get("function") or item.get("description") or item.get("service"),
                item.get("confidence"),
            ]
            for item in pid_items(parse, "equipment")
        ],
        [18, 16, 48, 12],
        80,
    )

    add_subsubsection(lines, "Piping / Process Line Table")
    line_rows = []
    for item in pid_items(parse, "lines"):
        line_rows.append([
            item.get("line_id") or item.get("id") or item.get("name"),
            item.get("from") or item.get("source"),
            item.get("to") or item.get("destination"),
            item.get("material_or_service") or item.get("material") or item.get("service"),
            item.get("flow_direction"),
        ])
    add_table(lines, ["Line/Stream", "From", "To", "Material/Service", "Direction"], line_rows, [16, 18, 18, 34, 12], 80)

    add_subsubsection(lines, "Instruments and Controls Table")
    add_table(
        lines,
        ["Tag", "Type", "Associated Item", "Description"],
        [
            [
                item.get("tag") or item.get("name"),
                item.get("type"),
                item.get("associated_equipment_or_line") or item.get("associated_equipment") or item.get("line") or item.get("location"),
                item.get("description") or item.get("function"),
            ]
            for item in pid_items(parse, "instruments_and_controls", "instruments")
        ],
        [16, 20, 28, 40],
        80,
    )

    add_subsubsection(lines, "Valves and Relief Devices Table")
    add_table(
        lines,
        ["Tag", "Type", "Location", "Description"],
        [
            [
                item.get("tag") or item.get("name"),
                item.get("type"),
                item.get("location") or item.get("associated_equipment_or_line") or item.get("line"),
                item.get("description") or item.get("function"),
            ]
            for item in pid_items(parse, "valves_and_relief_devices", "valves")
        ],
        [16, 20, 28, 40],
        80,
    )

    uncertainties = parse.get("uncertainties") if isinstance(parse.get("uncertainties"), list) else []
    if uncertainties:
        add_subsubsection(lines, "P&ID Parse Warnings and Uncertainties")
        for item in uncertainties[:30]:
            add_wrapped_line(lines, f"- {report_text(item)}")


def collect_reference_texts(*containers: Any, limit: int = 80) -> List[str]:
    references: List[str] = []

    def add(value: Any) -> None:
        text = report_text(value)
        if text:
            references.append(text)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"source_references", "references", "data_sources"}:
                    if isinstance(item, list):
                        for entry in item:
                            add(entry)
                    else:
                        add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for container in containers:
        walk(container)

    seen = set()
    unique: List[str] = []
    for reference in references:
        key = reference.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(reference)
        if len(unique) >= limit:
            break
    return unique


def job_report_lines(job: Dict[str, Any]) -> List[str]:
    result = job.get("result", {})
    report = result.get("report") or {}
    report_inputs = result.get("report_inputs") or {}
    nodes = result.get("nodes") or []
    worksheet = worksheet_rows_for_export(result)
    actions = result.get("action_register") or []
    ranked_actions = sorted(actions, key=lambda row: normalize_whitespace(row.get("action_id")))
    references = collect_reference_texts(report, worksheet, actions)

    lines = [
        "# Haznova HAZOP Report",
        f"Job ID: {job.get('id', '')}",
        f"Status: {job.get('status', '')}",
        f"Submitted: {job.get('submitted_at', '')}",
        f"Updated: {job.get('updated_at', '')}",
        "",
        "## Executive Summary",
    ]

    for item in report_bullets(report.get("executive_summary") or "HAZOP study completed.", 6):
        add_wrapped_line(lines, f"- {item}")
    lines.extend([
        "",
        "## Study Metrics",
        f"- Nodes studied: {len(nodes)}",
        f"- Worksheet rows: {len(worksheet)}",
        f"- Action items: {len(actions)}",
    ])

    key_findings = report.get("key_findings") or []
    lines.extend(["", "## Key Findings"])
    if key_findings:
        for item in key_findings[:10]:
            add_wrapped_line(lines, f"- {report_text(item)}")
    else:
        lines.append("- No key findings were generated.")

    lines.extend(["", "## Submitted Inputs"])
    if report_inputs.get("hazop_setup"):
        add_subsubsection(lines, "HAZOP Setup / Manual Nodes")
        for item in report_bullets(report_inputs.get("hazop_setup"), 12):
            add_wrapped_line(lines, f"- {item}")
    if report_inputs.get("process_description"):
        add_subsubsection(lines, "Process Description")
        for item in report_bullets(report_inputs.get("process_description"), 12):
            add_wrapped_line(lines, f"- {item}")
    else:
        lines.append("- No process description text was submitted.")
    if report_inputs.get("pid_text"):
        add_subsubsection(lines, "P&ID / Drawing Text Extract")
        for item in report_bullets(report_inputs.get("pid_text"), 8):
            add_wrapped_line(lines, f"- {item}")
    if report_inputs.get("safeguards"):
        add_subsubsection(lines, "Controls, Alarms, and Maintenance Procedures")
        for item in report_bullets(report_inputs.get("safeguards"), 8):
            add_wrapped_line(lines, f"- {item}")
    lines.extend(["", "## P&ID Parse"])
    add_pid_parse_report(lines, result)

    lines.extend(["", "## Nodes Reviewed"])
    add_table(
        lines,
        ["Node", "Name", "Boundary", "Design Intent"],
        [
            [
                node.get("node_id"),
                node.get("node_name"),
                node.get("boundaries"),
                node.get("design_intent"),
            ]
            for node in nodes
        ],
        [10, 24, 34, 48],
        120,
    )
    for node in nodes[:20]:
        params = ", ".join(sanitize_list(node.get("parameters"), 20))
        if params:
            add_wrapped_line(lines, f"- {node.get('node_id', '')} parameters assessed: {params}")

    lines.extend(["", "## Complete Discussion Rounds"])
    for node_result in result.get("node_results", []) or []:
        node = node_result.get("node", {})
        rounds = node_result.get("rounds", {})
        lines.extend(["", f"### {node.get('node_id', '')} - {node.get('node_name', '')}"])
        add_wrapped_line(lines, f"- Design intent: {node.get('design_intent', '')}")
        chair_opening = rounds.get("round_1_chair_opening") or []
        lines.append("Round 1 - Chair-led expert elicitation")
        if chair_opening:
            lines.append("  Chair opening and control of discussion:")
            for item in report_bullets(chair_opening, 5):
                add_wrapped_line(lines, f"* {item}", "      ")
        if rounds.get("round_1_summary"):
            lines.append("  Chair Round 1 consolidation:")
            for item in report_bullets(rounds.get("round_1_summary"), 5):
                add_wrapped_line(lines, f"* {item}", "      ")
        for agent in qualitative_agent_outputs(node_result.get("round_1")):
            lines.append(f"  {report_text(agent.get('agent_name') or 'Expert agent')}:")
            for item in report_bullets(agent.get("opening_statement") or agent.get("domain_focus"), 3):
                add_wrapped_line(lines, f"* {item}", "      ")
            for scenario in (agent.get("scenarios") or [])[:8]:
                add_wrapped_line(lines, f"* {scenario.get('guideword', '')} {scenario.get('parameter', '')} - {scenario.get('deviation', '')}", "      ")
                add_report_bullet_block(lines, "Potential cause", scenario.get("potential_causes") or scenario.get("potential_cause") or scenario.get("cause"), 3)
                add_report_bullet_block(lines, "Consequence", scenario.get("consequence") or scenario.get("consequences"), 3)
                add_report_bullet_block(lines, "Safeguard", scenario.get("safeguard") or scenario.get("safeguards"), 3)

        lines.append("Round 2 - Cross-review and cross-agent discussion")
        if rounds.get("round_2_summary"):
            for item in report_bullets(rounds.get("round_2_summary"), 5):
                add_wrapped_line(lines, f"* {item}", "      ")
        deliberation = node_result.get("round_2_deliberation_summary") or {}
        if deliberation:
            add_wrapped_line(
                lines,
                (
                    "* Deliberation metrics: pairwise agreement "
                    f"{float(deliberation.get('round_2a_pairwise_agreement_rate', 0)):.1%} "
                    f"(Round 2A) to {float(deliberation.get('round_2b_pairwise_agreement_rate', 0)):.1%} "
                    f"(Round 2B); Fleiss' kappa "
                    f"{float(deliberation.get('round_2a_fleiss_kappa', 0)):.3f} to "
                    f"{float(deliberation.get('round_2b_fleiss_kappa', 0)):.3f}; unanimous scenarios "
                    f"{deliberation.get('round_2a_unanimous_count', 0)} to "
                    f"{deliberation.get('round_2b_unanimous_count', 0)}; agent decisions changed "
                    f"{deliberation.get('agent_decision_change_count', 0)}. Review coverage was "
                    f"{float(deliberation.get('round_2a_review_coverage_rate', 1)):.1%} in Round 2A "
                    f"and {float(deliberation.get('round_2b_review_coverage_rate', 1)):.1%} in Round 2B."
                ),
                "      ",
            )
        for agent in qualitative_agent_outputs(node_result.get("round_2")):
            lines.append(f"  {report_text(agent.get('agent_name') or 'Expert agent')}:")
            if agent.get("required_scenario_count") is not None:
                add_wrapped_line(
                    lines,
                    (
                        f"* Review coverage: {agent.get('reviewed_scenario_count', 0)} of "
                        f"{agent.get('required_scenario_count', 0)} Round 1 scenarios explicitly reviewed; "
                        f"{agent.get('missing_review_count', 0)} blank after three attempts."
                    ),
                    "      ",
                )
            for item in report_bullets(agent.get("round_2_position") or agent.get("challenge_summary"), 4):
                add_wrapped_line(lines, f"* {item}", "      ")
            for item in (agent.get("discussion_refinements") or [])[:5]:
                if not isinstance(item, dict):
                    continue
                add_wrapped_line(
                    lines,
                    (
                        f"* Round 2B {item.get('review_id', 'scenario')}: "
                        f"{item.get('initial_decision', '')} to {item.get('final_decision', '')}. "
                        f"{report_text(item.get('change_reason'))}"
                    ),
                    "      ",
                )
            for item in (agent.get("agreements") or [])[:5]:
                if not isinstance(item, dict):
                    continue
                add_wrapped_line(lines, f"* Agreement with {item.get('agrees_with_agent', 'panel')}: {report_text(item.get('reason'))}", "      ")
            for item in (agent.get("safeguard_challenges") or [])[:5]:
                if not isinstance(item, dict):
                    continue
                add_wrapped_line(lines, f"* Safeguard challenge on {item.get('row_id', 'row')}: {report_text(item.get('issue'))}", "      ")
                add_wrapped_line(lines, f"  Proposed revision: {report_text(item.get('proposed_revision'))}", "      ")
            for item in ((agent.get("scenario_challenges") or []) + (agent.get("challenges") or []))[:5]:
                if not isinstance(item, dict):
                    continue
                add_wrapped_line(lines, f"* Scenario correction on {item.get('row_id', 'row')}: {report_text(item.get('issue'))}", "      ")
                add_wrapped_line(lines, f"  Proposed revision: {report_text(item.get('proposed_revision'))}", "      ")
            for item in (agent.get("new_scenarios") or agent.get("missing_scenarios") or [])[:5]:
                if not isinstance(item, dict):
                    continue
                add_wrapped_line(lines, f"* Missing scenario: {item.get('guideword', '')} {item.get('parameter', '')} - {item.get('deviation', '')}", "      ")
                add_report_bullet_block(lines, "Potential cause", item.get("potential_cause") or item.get("cause"), 2)
                add_report_bullet_block(lines, "Consequence", item.get("consequence"), 2)

        new_deliberations = node_result.get("round_2_new_scenario_deliberations") or []
        if new_deliberations:
            lines.append("  Round 2B new-scenario decisions:")
            for item in new_deliberations:
                if not isinstance(item, dict):
                    continue
                proposal = item.get("proposal") or {}
                counts = item.get("decision_counts") or {}
                add_wrapped_line(
                    lines,
                    (
                        f"* {item.get('proposal_id', '')} - {proposal.get('deviation', '')}; "
                        f"outcome={item.get('panel_outcome', '')}; Agree={counts.get('agree', 0)}, "
                        f"Partially agree={counts.get('partially_agree', 0)}, "
                        f"Disagree={counts.get('disagree', 0)}."
                    ),
                    "      ",
                )
                for review in item.get("reviews", []) or []:
                    if not isinstance(review, dict):
                        continue
                    add_wrapped_line(
                        lines,
                        (
                            f"  {review.get('agent_name', '')}: {review.get('decision', '')} - "
                            f"{report_text(review.get('reason'))}"
                        ),
                        "      ",
                    )

        lines.append("Round 3 - Chair finalization")
        if rounds.get("round_3_chair_summary"):
            for item in report_bullets(rounds.get("round_3_chair_summary"), 5):
                add_wrapped_line(lines, f"* {item}", "      ")
        if rounds.get("round_3_chair_synthesis"):
            lines.append("  Final Chair synthesis:")
            for item in report_bullets(rounds.get("round_3_chair_synthesis"), 8):
                add_wrapped_line(lines, f"* {item}", "      ")
        chair_rows = (
            node_result.get("final_display_rows")
            or consolidate_worksheet_rows(node_result.get("final_rows") or [])
        )
        add_table(
            lines,
            ["Guideword", "Deviation / Effect", "Potential cause", "Consequence", "Safeguard", "Necessary action"],
            [
                [
                    row.get("guideword"),
                    row.get("deviation"),
                    row.get("potential_cause") or row.get("cause"),
                    row.get("consequence"),
                    row.get("safeguard"),
                    row.get("necessary_action"),
                ]
                for row in chair_rows
            ],
            [10, 18, 30, 30, 26, 30],
            80,
        )
    lines.extend(["", "## Source References"])
    if references:
        for index, reference in enumerate(references[:30], 1):
            add_wrapped_line(lines, f"{index}. {reference}")
    else:
        lines.append("- No source references were recorded.")

    lines.extend(["", "## HAZOP Worksheet"])
    for node_id in sorted({row.get("node_id", "") for row in worksheet}):
        node_rows = [row for row in worksheet if row.get("node_id", "") == node_id]
        node_name = node_rows[0].get("node_name", "") if node_rows else ""
        lines.extend(["", f"### {node_id} - {node_name}"])
        add_table(
            lines,
            ["Guideword", "Deviation / Effect", "Potential cause", "Consequence", "Safeguard", "Necessary action"],
            [
                [
                    row.get("guideword"),
                    row.get("deviation"),
                    row.get("potential_cause") or row.get("cause"),
                    row.get("consequence"),
                    row.get("safeguard"),
                    row.get("necessary_action"),
                ]
                for row in node_rows
            ],
            [10, 20, 32, 32, 26, 40],
            160,
        )

    lines.extend(["", "## Action Register"])
    if not ranked_actions:
        lines.append("- No action items were generated.")
    add_table(
        lines,
        ["Rank", "Action ID", "Node", "Deviation / Effect", "Action", "Status"],
        [
            [
                index,
                row.get("action_id"),
                row.get("node_id"),
                row.get("deviation"),
                row.get("necessary_action"),
                row.get("action_status"),
            ]
            for index, row in enumerate(ranked_actions, 1)
        ],
        [6, 10, 10, 24, 50, 10],
        120,
    )

    lines.extend(["", "## Study Limitations"])
    limitations = report.get("assumptions_and_limitations") or []
    if limitations:
        for item in limitations[:15]:
            add_wrapped_line(lines, f"- {report_text(item)}")
    else:
        lines.append("- No additional study limitations were generated.")

    if job.get("error"):
        add_report_section(lines, "Error Trace", job.get("error"), 40000)
    return lines


def xml_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def docx_paragraph(text: Any = "", style: str = "", bullet: bool = False) -> str:
    text = report_text(text)
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{xml_escape(style)}"/>'
    if bullet:
        ppr += '<w:ind w:left="360" w:hanging="180"/>'
    ppr_xml = f"<w:pPr>{ppr}</w:pPr>" if ppr else ""
    prefix = "• " if bullet else ""
    return f"<w:p>{ppr_xml}<w:r><w:t xml:space=\"preserve\">{xml_escape(prefix + text)}</w:t></w:r></w:p>"


def docx_bullets(items: Any, max_items: int = 8) -> str:
    bullets = report_bullets(items, max_items)
    if not bullets:
        return docx_paragraph("None stated.", bullet=True)
    return "".join(docx_paragraph(item, bullet=True) for item in bullets)


def docx_cell(value: Any, width: int = 2200, force_bullets: bool = False) -> str:
    items = worksheet_slot_items(value, 30) if force_bullets else report_bullets(value, 8)
    if not items:
        items = [report_text(value)]
    paragraphs = "".join(docx_paragraph(item, bullet=(force_bullets or len(items) > 1)) for item in items if normalize_whitespace(item))
    if not paragraphs:
        paragraphs = docx_paragraph("")
    return f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>{paragraphs}</w:tc>'


def docx_table(
    headers: List[str],
    rows: List[List[Any]],
    widths: Optional[List[int]] = None,
    max_rows: int = 120,
    bullet_columns: Optional[List[int]] = None,
) -> str:
    widths = widths or [2200 for _ in headers]
    bullet_column_set = set(bullet_columns or [])
    table_pr = (
        '<w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="9CB3C9"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="9CB3C9"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="9CB3C9"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="9CB3C9"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D8E2EB"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="D8E2EB"/>'
        '</w:tblBorders></w:tblPr>'
    )
    header_row = "<w:tr>" + "".join(
        f'<w:tc><w:tcPr><w:tcW w:w="{widths[i] if i < len(widths) else 2200}" w:type="dxa"/><w:shd w:fill="DDEAF3"/></w:tcPr>'
        f'{docx_paragraph(header, "TableHeader")}</w:tc>'
        for i, header in enumerate(headers)
    ) + "</w:tr>"
    body_rows = []
    for row in rows[:max_rows]:
        padded = list(row) + [""] * max(0, len(headers) - len(row))
        body_rows.append("<w:tr>" + "".join(
            docx_cell(value, widths[i] if i < len(widths) else 2200, i in bullet_column_set)
            for i, value in enumerate(padded[:len(headers)])
        ) + "</w:tr>")
    if len(rows) > max_rows:
        body_rows.append("<w:tr>" + docx_cell(f"{len(rows) - max_rows} additional row(s) omitted from this Word table. Use the Excel workbook for complete data.", sum(widths)) + "</w:tr>")
    return "<w:tbl>" + table_pr + header_row + "".join(body_rows) + "</w:tbl>"


def docx_image_xml(relationship_id: str, image_index: int, title: str, width_emu: int, height_emu: int) -> str:
    safe_title = xml_escape(title)
    return (
        '<w:p><w:r><w:drawing>'
        '<wp:inline distT="0" distB="0" distL="0" distR="0" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
        f'<wp:docPr id="{image_index}" name="{safe_title}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:nvPicPr>'
        f'<pic:cNvPr id="{image_index}" name="{safe_title}"/>'
        '<pic:cNvPicPr/>'
        '</pic:nvPicPr>'
        '<pic:blipFill>'
        f'<a:blip r:embed="{relationship_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
        '<a:stretch><a:fillRect/></a:stretch>'
        '</pic:blipFill>'
        '<pic:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '</pic:spPr>'
        '</pic:pic>'
        '</a:graphicData>'
        '</a:graphic>'
        '</wp:inline>'
        '</w:drawing></w:r></w:p>'
    )


def docx_image(image: Dict[str, Any], relationship_id: str, image_index: int) -> str:
    width_px = max(1, int(image.get("width") or 1))
    height_px = max(1, int(image.get("height") or 1))
    width_emu = int(width_px / 96 * 914400)
    height_emu = int(height_px / 96 * 914400)
    max_width_emu = int(9.8 * 914400)
    max_height_emu = int(4.6 * 914400)
    scale = min(1.0, max_width_emu / max(1, width_emu), max_height_emu / max(1, height_emu))
    return docx_image_xml(
        relationship_id,
        image_index,
        image.get("title") or f"P&ID Image {image_index}",
        int(width_emu * scale),
        int(height_emu * scale),
    )


def docx_fixed_image(relationship_id: str, image_index: int, title: str, width_inches: float = 9.8, height_inches: float = 4.6) -> str:
    return docx_image_xml(
        relationship_id,
        image_index,
        title,
        int(width_inches * 914400),
        int(height_inches * 914400),
    )


def docx_report_bytes(job: Dict[str, Any]) -> bytes:
    result = job.get("result", {})
    report = result.get("report") or {}
    report_inputs = result.get("report_inputs") or {}
    nodes = result.get("nodes") or []
    node_results = result.get("node_results") or []
    worksheet = worksheet_rows_for_export(result)
    actions = result.get("action_register") or []
    parse = result.get("pid_parse") or {}
    images = office_images_from_job(job)
    references = collect_reference_texts(report, worksheet, actions)

    body: List[str] = [
        docx_paragraph("Haznova HAZOP Report", "Title"),
        docx_paragraph(f"Job ID: {job.get('id', '')}"),
        docx_paragraph(f"Status: {job.get('status', '')}"),
        docx_paragraph(f"Generated: {utc_now()}"),
        docx_paragraph("Executive Summary", "Heading1"),
        docx_bullets(report.get("executive_summary") or "HAZOP study completed.", 8),
        docx_paragraph("Study Metrics", "Heading1"),
        docx_bullets([
            f"Nodes studied: {len(nodes)}",
            f"Worksheet rows: {len(worksheet)}",
            f"Action items: {len(actions)}",
        ]),
        docx_paragraph("Key Findings", "Heading1"),
        docx_bullets(report.get("key_findings") or ["No key findings were generated."], 12),
        docx_paragraph("Submitted Inputs", "Heading1"),
        docx_paragraph("HAZOP Setup / Manual Nodes", "Heading2"),
        docx_bullets(report_inputs.get("hazop_setup") or ["No manual HAZOP setup text was submitted."], 12),
        docx_paragraph("Process Description", "Heading2"),
        docx_bullets(report_inputs.get("process_description") or ["No process description text was submitted."], 12),
        docx_paragraph("P&ID / Drawing Text", "Heading2"),
        docx_bullets(report_inputs.get("pid_text") or ["No drawing text extract was submitted."], 8),
        docx_paragraph("Controls, Alarms, and Maintenance Procedures", "Heading2"),
        docx_bullets(report_inputs.get("safeguards") or ["No controls, alarms, or maintenance procedure notes were submitted."], 8),
        docx_paragraph("Submitted P&ID Images", "Heading1"),
    ]
    if images:
        for index, image in enumerate(images, 1):
            body.append(docx_paragraph(image.get("title") or f"Submitted P&ID Image {index}", "Heading2"))
            body.append(docx_image(image, f"rIdImage{index}", index))
    else:
        body.append(docx_bullets(["No supported PNG or JPEG P&ID image was available for Word embedding."]))
    body.extend([
        docx_paragraph("P&ID Parse", "Heading1"),
        docx_table(
            ["Metric", "Value"],
            [
                ["Vision parse succeeded", bool(parse.get("vision_parse_succeeded"))],
                ["Equipment items", len(pid_items(parse, "equipment"))],
                ["Piping lines", len(pid_items(parse, "lines"))],
                ["Instruments", len(pid_items(parse, "instruments_and_controls", "instruments"))],
                ["Valves/relief devices", len(pid_items(parse, "valves_and_relief_devices", "valves"))],
            ],
            [4200, 4200],
        ),
        docx_paragraph("Nodes Reviewed", "Heading1"),
        docx_table(
            ["Node", "Name", "Boundary", "Design Intent"],
            [[node.get("node_id"), node.get("node_name"), node.get("boundaries"), node.get("design_intent")] for node in nodes],
            [1200, 2600, 4200, 5200],
        ),
    ])
    body.append(docx_paragraph("Discussion Rounds", "Heading1"))
    for node_result in node_results:
        if not isinstance(node_result, dict):
            continue
        node = node_result.get("node") or {}
        rounds = node_result.get("rounds") or {}
        body.append(docx_paragraph(f"{node.get('node_id', '')} - {node.get('node_name', '')}", "Heading2"))
        body.append(docx_bullets([f"Design intent: {node.get('design_intent', '')}"], 1))
        body.append(docx_paragraph("Round 1 - Chair-led Expert Elicitation", "Heading2"))
        body.append(docx_bullets(rounds.get("round_1_chair_opening") or rounds.get("round_1_summary") or ["No Round 1 chair summary was recorded."], 8))
        body.append(docx_table(
            ["Agent", "Position", "Representative Scenarios"],
            [[
                agent.get("agent_name"),
                agent.get("opening_statement") or agent.get("domain_focus"),
                [
                    f"{scenario.get('guideword', '')} {scenario.get('parameter', '')}: {scenario.get('deviation', '')}"
                    for scenario in (agent.get("scenarios") or [])[:6]
                    if isinstance(scenario, dict)
                ],
            ] for agent in qualitative_agent_outputs(node_result.get("round_1"))],
            [2600, 5200, 6200],
            40,
        ))
        body.append(docx_paragraph("Round 2 - Independent Review and Cross-Agent Discussion", "Heading2"))
        methodology = node_result.get("round_2_review_methodology") or ROUND_TWO_DECISION_METHOD
        body.append(docx_bullets([
            (
                "Official review labels were derived deterministically from four technical checks. "
                "Disagree required failure of at least one core credibility criterion; Partially agree "
                "required no core failure but at least one correctable criterion; Agree required all "
                "criteria to pass. Agent-proposed labels were retained only as audit information."
            ),
            (
                "Method basis: " + "; ".join(
                    methodology.get("hazop_basis", []) + methodology.get("consensus_basis", [])
                )
            ),
        ], 2))
        deliberation = node_result.get("round_2_deliberation_summary") or {}
        if deliberation:
            body.append(docx_bullets([(
                "Round 2 deliberation: pairwise agreement changed from "
                f"{float(deliberation.get('round_2a_pairwise_agreement_rate', 0)):.1%} to "
                f"{float(deliberation.get('round_2b_pairwise_agreement_rate', 0)):.1%}; "
                f"Fleiss' kappa changed from {float(deliberation.get('round_2a_fleiss_kappa', 0)):.3f} "
                f"to {float(deliberation.get('round_2b_fleiss_kappa', 0)):.3f}; "
                f"unanimous scenarios changed from {deliberation.get('round_2a_unanimous_count', 0)} "
                f"to {deliberation.get('round_2b_unanimous_count', 0)}; "
                f"{deliberation.get('agent_decision_change_count', 0)} agent decisions changed. "
                f"Review coverage was {float(deliberation.get('round_2a_review_coverage_rate', 1)):.1%} "
                f"in Round 2A and {float(deliberation.get('round_2b_review_coverage_rate', 1)):.1%} "
                "in Round 2B; omitted responses remained blank after three attempts."
            )], 1))
        body.append(docx_table(
            ["Agent", "Review Coverage / Position", "Agreements / Challenges / Missing Scenarios"],
            [[
                agent.get("agent_name"),
                (
                    f"{agent.get('reviewed_scenario_count', 0)} of {agent.get('required_scenario_count', 0)} scenarios reviewed; "
                    f"{agent.get('missing_review_count', 0)} blank after three attempts. "
                    if agent.get("required_scenario_count") is not None else ""
                ) + report_text(agent.get("round_2_position") or agent.get("challenge_summary")),
                (agent.get("agreements") or []) + (agent.get("safeguard_challenges") or [])
                + (agent.get("scenario_challenges") or agent.get("challenges") or [])
                + (agent.get("new_scenarios") or agent.get("missing_scenarios") or []),
            ] for agent in qualitative_agent_outputs(node_result.get("round_2"))],
            [2600, 5200, 6200],
            40,
        ))
        decision_audit_rows = []
        for agent in qualitative_agent_outputs(node_result.get("round_2")):
            initial_by_id = {
                str(review.get("review_id", "")): review
                for review in agent.get("initial_scenario_reviews", []) or []
                if isinstance(review, dict)
            }
            for review in agent.get("scenario_reviews", []) or []:
                if not isinstance(review, dict):
                    continue
                review_id = str(review.get("review_id", ""))
                initial = initial_by_id.get(review_id, review)
                initial_decision = (
                    "No response (3 attempts)"
                    if initial.get("review_status") == "missing_after_3_attempts"
                    else initial.get("decision")
                )
                final_decision: Any
                if review.get("round_2b_review_status") == "missing_after_3_attempts":
                    final_decision = "No response (3 attempts)"
                elif "round_2a_decision" in review:
                    final_decision = review.get("decision")
                else:
                    final_decision = "Not required"
                decision_audit_rows.append([
                    agent.get("agent_name"),
                    review_id,
                    initial_decision,
                    final_decision,
                    "Yes" if review.get("round_2b_position_changed") else "No",
                    criterion_rating_summary(
                        review.get("criterion_assessment")
                        or initial.get("criterion_assessment")
                    ),
                    (
                        f"Suggested={review.get('agent_suggested_decision', '') or initial.get('agent_suggested_decision', '')}; "
                        f"rubric override={'Yes' if review.get('decision_overridden_by_rubric', False) or initial.get('decision_overridden_by_rubric', False) else 'No'}"
                    ),
                    (
                        "Review assignment remained blank after three attempts."
                        if review.get("round_2b_review_status") == "missing_after_3_attempts"
                        or initial.get("review_status") == "missing_after_3_attempts"
                        else review.get("reason")
                    ),
                    review.get("evidence_basis"),
                ])
        if decision_audit_rows:
            body.append(docx_paragraph("Round 2 Existing-Scenario Decision Audit", "Heading2"))
            body.append(docx_table(
                ["Agent", "Review ID", "Round 2A", "Round 2B", "Changed", "Final Criterion Ratings", "Suggestion Audit", "Final Reason", "Evidence Basis"],
                decision_audit_rows,
                [1900, 1700, 1100, 1100, 800, 3900, 2200, 4300, 3800],
                2000,
                bullet_columns=[5],
            ))
        new_deliberations = node_result.get("round_2_new_scenario_deliberations") or []
        if new_deliberations:
            body.append(docx_paragraph("Round 2B New-Scenario Deliberation", "Heading2"))
            body.append(docx_table(
                ["Proposal", "Source", "Scenario", "Panel Decisions", "Outcome", "Review Reasons"],
                [[
                    item.get("proposal_id"),
                    (item.get("proposal") or {}).get("source_agents"),
                    (
                        f"{(item.get('proposal') or {}).get('guideword', '')} / "
                        f"{(item.get('proposal') or {}).get('parameter', '')}: "
                        f"{(item.get('proposal') or {}).get('deviation', '')}; cause: "
                        f"{(item.get('proposal') or {}).get('potential_cause', '')}"
                    ),
                    item.get("decision_counts"),
                    item.get("panel_outcome"),
                    [
                        (
                            f"{review.get('agent_name', '')}: No response after three attempts"
                            if review.get("review_status") == "missing_after_3_attempts"
                            else f"{review.get('agent_name', '')}: {review.get('decision', '')} - {review.get('reason', '')}"
                        )
                        for review in item.get("reviews", []) or []
                    ],
                ] for item in new_deliberations],
                [1300, 2200, 4200, 2200, 1900, 5200],
                120,
                bullet_columns=[1, 5],
            ))
        body.append(docx_paragraph("Round 3 - Chair Finalization", "Heading2"))
        body.append(docx_bullets(rounds.get("round_3_chair_summary") or ["Chair finalization was completed."], 8))
        round_three_summary = node_result.get("round_3_adjudication_summary") or {}
        if round_three_summary:
            decisions = round_three_summary.get("decision_distribution") or {}
            panel_classes = round_three_summary.get(
                "round_2_panel_classification_distribution"
            ) or {}
            body.append(docx_bullets([(
                f"Chair adjudication coverage was {float(round_three_summary.get('adjudication_coverage_rate', 0)):.1%}; "
                f"Round 2B panel conclusions were Agree={panel_classes.get('agree', 0)}, "
                f"Partially agree={panel_classes.get('partially_agree', 0)}, "
                f"Disagree={panel_classes.get('disagree', 0)}, "
                f"and not linked/returned={panel_classes.get('not_supplied', 0) + panel_classes.get('unlinked', 0)}; "
                f"Agree={decisions.get('agree', 0)}, Partially agree={decisions.get('partially_agree', 0)}, "
                f"Disagree={decisions.get('disagree', 0)}. Pairwise agreement and Fleiss' kappa are not "
                "applicable because Round 3 has one final adjudicator."
            )], 1))
        round_three_trace = node_result.get("round_3_adjudication_trace") or []
        if round_three_trace:
            body.append(docx_table(
                ["Adjudication", "Candidate", "Round 2B Panel Basis", "Official Decision", "Worksheet Treatment", "Criterion Ratings", "Chair Resolution / Evidence"],
                [[
                    item.get("adjudication_id", ""),
                    (
                        f"{item.get('guideword', '')} / {item.get('parameter', '')}: "
                        f"{item.get('deviation', '')}; cause: {item.get('potential_cause', '')}"
                    ),
                    (
                        f"{(item.get('round_2_panel_basis') or {}).get('classification', '')}; "
                        f"counts={(item.get('round_2_panel_basis') or {}).get('decision_counts', {})}"
                    ),
                    item.get("chair_decision", "") or item.get("review_status", ""),
                    item.get("worksheet_treatment", ""),
                    criterion_rating_summary(item.get("criterion_assessment", {})),
                    [
                        item.get("chair_resolution", ""),
                        *sanitize_list(item.get("chair_evidence_basis"), 20),
                        item.get("chair_revision_summary", ""),
                    ],
                ] for item in round_three_trace if isinstance(item, dict)],
                [1500, 3600, 2800, 1500, 2200, 3700, 4600],
                500,
                bullet_columns=[4, 5],
            ))
        round_three_proposals = node_result.get("round_3_escalated_proposals") or []
        if round_three_proposals:
            body.append(docx_paragraph("Round 3 Chair Proposals Requiring Panel Review", "Heading2"))
            body.append(docx_table(
                ["Proposal", "Scenario", "Chair Rubric Decision", "Evidence / Reason", "Status"],
                [[
                    item.get("proposal_id", ""),
                    (
                        f"{item.get('guideword', '')} / {item.get('parameter', '')}: "
                        f"{item.get('deviation', '')}; cause: {item.get('potential_cause', '')}"
                    ),
                    item.get("chair_derived_decision", ""),
                    [
                        *sanitize_list(item.get("chair_evidence_basis"), 20),
                        item.get("proposal_reason", ""),
                    ],
                    item.get("required_next_step", ""),
                ] for item in round_three_proposals if isinstance(item, dict)],
                [1800, 4700, 2000, 5000, 3500],
                200,
                bullet_columns=[3],
            ))
        round_three_rows = (
            node_result.get("final_display_rows")
            or consolidate_worksheet_rows(node_result.get("final_rows") or [])
        )
        round_three_rows = [
            row for row in round_three_rows
            if isinstance(row, dict) and not row_is_not_applicable(row)
        ]
        body.append(docx_table(
            ["Row", "Guideword", "Deviation / Effect", "Potential Cause", "Consequence", "Safeguard", "Necessary Action"],
            [[
                row.get("row_id", ""),
                row.get("guideword"),
                row.get("deviation"),
                row.get("potential_cause") or row.get("cause"),
                row.get("consequence"),
                row.get("safeguard"),
                row.get("necessary_action"),
            ] for row in round_three_rows if isinstance(row, dict)],
            [700, 1300, 2100, 2600, 2600, 2400, 2500],
            80,
            bullet_columns=[3, 4, 5, 6],
        ))
    body.extend([
        docx_paragraph("HAZOP Worksheet", "Heading1"),
    ])
    for node_id in sorted({row.get("node_id", "") for row in worksheet}):
        node_rows = [row for row in worksheet if row.get("node_id") == node_id]
        node_name = node_rows[0].get("node_name", "") if node_rows else ""
        body.append(docx_paragraph(f"{node_id} - {node_name}", "Heading2"))
        body.append(docx_table(
            ["Row", "Guideword", "Deviation / Effect", "Potential Cause", "Consequence", "Safeguard", "Necessary Action"],
            [[
                row.get("row_id", ""),
                row.get("guideword"),
                row.get("deviation"),
                row.get("potential_cause") or row.get("cause"),
                row.get("consequence"),
                row.get("safeguard"),
                row.get("necessary_action"),
            ] for row in node_rows],
            [700, 1300, 2100, 2600, 2600, 2400, 2500],
            bullet_columns=[3, 4, 5, 6],
        ))
    body.extend([
        docx_paragraph("Action Register", "Heading1"),
        docx_table(
            ["Action ID", "Node", "Deviation / Effect", "Necessary Action", "Status"],
            [[row.get("action_id"), row.get("node_id"), row.get("deviation"), row.get("necessary_action"), row.get("action_status")] for row in sorted(actions, key=lambda row: normalize_whitespace(row.get("action_id")))],
            [1100, 1000, 2800, 7600, 1200],
        ),
        docx_paragraph("Source References", "Heading1"),
        docx_bullets(references or ["No source references were recorded."], 30),
        docx_paragraph("Study Limitations", "Heading1"),
        docx_bullets(report.get("assumptions_and_limitations") or ["No additional study limitations were generated."], 20),
        '<w:sectPr><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/></w:sectPr>',
    ])
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<w:body>' + "".join(body) + '</w:body></w:document>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:docDefaults><w:rPrDefault><w:rPr><w:sz w:val="20"/></w:rPr></w:rPrDefault>'
        '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
        '<w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/>'
        '<w:pPr><w:spacing w:after="120"/></w:pPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:pPr><w:spacing w:before="0" w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/><w:color w:val="0A3761"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
        '<w:pPr><w:spacing w:before="360" w:after="160"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="0A3761"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>'
        '<w:pPr><w:spacing w:before="200" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="20"/><w:color w:val="007C89"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="TableHeader"><w:name w:val="Table Header"/>'
        '<w:pPr><w:spacing w:after="40"/></w:pPr><w:rPr><w:b/><w:color w:val="0A3761"/></w:rPr></w:style>'
        '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/></w:style>'
        '</w:styles>'
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as docx:
        image_defaults = ""
        if any(image.get("ext") == "png" for image in images):
            image_defaults += '<Default Extension="png" ContentType="image/png"/>'
        if any(image.get("ext") == "jpg" for image in images):
            image_defaults += '<Default Extension="jpg" ContentType="image/jpeg"/>'
        docx.writestr("[Content_Types].xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>{image_defaults}<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        docx.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        image_rels = "".join(
            f'<Relationship Id="rIdImage{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{index}.{image.get("ext", "png")}"/>'
            for index, image in enumerate(images, 1)
        )
        docx.writestr("word/_rels/document.xml.rels", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>{image_rels}</Relationships>')
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        for index, image in enumerate(images, 1):
            docx.writestr(f'word/media/image{index}.{image.get("ext", "png")}', image.get("data", b""))
    return output.getvalue()


def xlsx_col(index: int) -> str:
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def worksheet_bullet_text(value: Any, max_items: int = 12, default: str = "") -> str:
    items = worksheet_slot_items(value, max_items)
    if not items and normalize_whitespace(default):
        items = [normalize_whitespace(default)]
    if not items:
        return ""
    return "\n".join(f"• {item}" for item in items)


def worksheet_slot_items(value: Any, max_items: int = 30) -> List[str]:
    """Split worksheet bullets without removing repeated aligned slots."""
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"\r?\n+", str(value or ""))
    items: List[str] = []
    for item in raw_items:
        text = normalize_whitespace(item).lstrip("•*- ").strip()
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def xlsx_cell(row: int, col: int, value: Any, header: bool = False) -> str:
    ref = f"{xlsx_col(col)}{row}"
    style = ' s="1"' if header else ""
    number = numeric_value(value, None)
    if number is not None and normalize_whitespace(value) not in {"", "Not quantified"}:
        return f'<c r="{ref}"{style}><v>{number:.12g}</v></c>'
    raw_text = str(value if value is not None else "")
    text = raw_text if ("\n" in raw_text or raw_text.lstrip().startswith("•")) else report_text(value)
    return f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">{xml_escape(text)}</t></is></c>'


def xlsx_sheet_xml(headers: List[str], rows: List[List[Any]], column_widths: Optional[List[int]] = None) -> str:
    sheet_rows = []
    sheet_rows.append('<row r="1">' + "".join(xlsx_cell(1, col, header, True) for col, header in enumerate(headers, 1)) + "</row>")
    for row_index, row in enumerate(rows, 2):
        padded = list(row) + [""] * max(0, len(headers) - len(row))
        sheet_rows.append(f'<row r="{row_index}">' + "".join(xlsx_cell(row_index, col, value) for col, value in enumerate(padded[:len(headers)], 1)) + "</row>")
    widths = column_widths or [min(70, max(12, len(header) + 8)) for header in headers]
    cols = "".join(
        f'<col min="{idx}" max="{idx}" width="{widths[idx - 1] if idx - 1 < len(widths) else 20}" customWidth="1"/>'
        for idx in range(1, len(headers) + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<cols>{cols}</cols><sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )


def worksheet_rows_for_export(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    display_rows = [
        row for row in (result.get("worksheet_display") or [])
        if isinstance(row, dict) and not row_is_not_applicable(row)
    ]
    if display_rows:
        return display_rows
    worksheet = [
        row for row in (result.get("worksheet") or [])
        if isinstance(row, dict) and not row_is_not_applicable(row)
    ]
    nodes = [node for node in (result.get("nodes") or []) if isinstance(node, dict)]
    node_by_id = {
        normalize_whitespace(node.get("node_id")): node
        for node in nodes
        if normalize_whitespace(node.get("node_id"))
    }
    rows_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for row in worksheet:
        node_id = normalize_whitespace(row.get("node_id")) or "N-UNKNOWN"
        rows_by_node.setdefault(node_id, []).append(row)

    export_rows: List[Dict[str, Any]] = []
    for node_id in sorted(rows_by_node):
        node_rows = rows_by_node[node_id]
        parameters = sorted({
            normalize_whitespace(row.get("parameter")).lower()
            for row in node_rows
            if normalize_whitespace(row.get("parameter"))
        }) or list(HAZOP_PARAMETERS)
        node = node_by_id.get(node_id) or {
            "node_id": node_id,
            "node_name": normalize_whitespace(first_non_empty(*(row.get("node_name") for row in node_rows))) or node_id,
            "node_description": normalize_whitespace(first_non_empty(*(row.get("node_description") for row in node_rows))),
            "design_intent": normalize_whitespace(first_non_empty(*(row.get("design_intent") for row in node_rows))),
            "parameters": parameters,
        }
        if not node.get("parameters"):
            node = dict(node)
            node["parameters"] = parameters
        export_rows.extend(
            row for row in node_rows
            if isinstance(row, dict) and not row_is_not_applicable(row)
        )
    return consolidate_worksheet_rows(export_rows)


def xlsx_workbook_bytes(job: Dict[str, Any]) -> bytes:
    result = job.get("result", {})
    worksheet = worksheet_rows_for_export(result)
    sheets = [
        ("HAZOP Worksheet", [
            "Row ID", "Node", "Node Name", "Guideword", "Parameter",
            "Deviation / Effect", "Potential Cause", "Consequence", "Safeguard", "Safeguard Evidence",
            "Necessary Action", "Source Review IDs", "Source Proposal IDs"
        ], [[
            row.get("row_id"), row.get("node_id"), row.get("node_name"),
            row.get("guideword"), row.get("parameter"),
            worksheet_bullet_text(row.get("deviation"), default=f"{row.get('guideword', '')} {row.get('parameter', '')}"),
            worksheet_bullet_text(row.get("potential_cause") or row.get("cause"), default="Not identified"),
            worksheet_bullet_text(row.get("consequence"), default="Not identified"),
            worksheet_bullet_text(row.get("safeguard"), default="Not identified"),
            worksheet_bullet_text(row.get("safeguard_evidence")),
            worksheet_bullet_text(row.get("necessary_action"), default="Not identified"),
            worksheet_bullet_text(row.get("source_review_ids")),
            worksheet_bullet_text(row.get("source_proposal_ids")),
        ] for row in worksheet], [18, 10, 18, 14, 12, 34, 42, 42, 38, 36, 38, 28, 24]),
    ]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as xlsx:
        overrides = "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets) + 1))
        xlsx.writestr("[Content_Types].xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>{overrides}</Types>')
        xlsx.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        sheet_defs = "".join(f'<sheet name="{xml_escape(name[:31])}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _, _, _) in enumerate(sheets, 1))
        xlsx.writestr("xl/workbook.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{sheet_defs}</sheets></workbook>')
        rels = "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1))
        rels += f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        xlsx.writestr("xl/_rels/workbook.xml.rels", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>')
        xlsx.writestr("xl/styles.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FF0A3761"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFDDEAF3"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf><xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs></styleSheet>')
        for i, (_, headers, rows, widths) in enumerate(sheets, 1):
            xlsx.writestr(f"xl/worksheets/sheet{i}.xml", xlsx_sheet_xml(headers, rows, widths))
    return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main HAZOP workflow
# ─────────────────────────────────────────────────────────────────────────────

def finalize_node_rows(rows: List[Dict[str, Any]], node: Dict[str, Any], documents: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        row for row in rows
        if isinstance(row, dict) and not row_is_not_applicable(row)
    ]
    rows = remove_cross_guideword_communication_items(rows, node)
    rows = [
        sanitize_row(row, node, index + 1)
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    ]
    rows = clean_cross_guideword_content(rows, node)
    rows = clean_node_safeguards(rows, documents, node)
    rows = clean_node_actions(rows, node)
    rows = remove_cross_guideword_communication_items(rows, node)
    rows = clean_cross_guideword_content(rows, node)
    return [
        complete_worksheet_row(row, node)
        for row in rows
        if isinstance(row, dict) and not row_is_not_applicable(row)
    ]


def save_performance_snapshot_event(job_id: str, reason: str, saved_message: str, failed_message: str) -> None:
    try:
        snapshot_path = save_job_performance_snapshot(job_id, reason)
        if snapshot_path:
            add_event(job_id, "performance", f"{saved_message} {snapshot_path}.")
    except Exception as snapshot_error:
        add_event(job_id, "performance", f"{failed_message}: {snapshot_error}")


def run_hazop_job(job_id: str, payload: Dict[str, Any]) -> None:
    try:
        payload["_job_id"] = job_id
        set_status(job_id, "running")
        set_result(job_id, ["performance", "experiment_metadata"], {
            "job_id": job_id,
            "run_started_at": utc_now(),
            "backend_source_hash": backend_source_hash(),
            "prompt_template_version": "haznova-agent-prompts-2026-11-01-batched-round3",
            "settings": json_safe(payload.get("settings", {})),
            "agents": json_safe(payload.get("agents", {})),
            "guidewords": [{"guideword": guideword, "meaning": meaning} for guideword, meaning in HAZOP_GUIDEWORDS],
            "default_parameters": list(HAZOP_PARAMETERS),
            "round_two_review_methodology": json_safe(ROUND_TWO_DECISION_METHOD),
            "round_three_chair_methodology": json_safe(ROUND_THREE_DECISION_METHOD),
            "document_lengths": {
                key: (len(value) if not isinstance(value, list) else len(value))
                for key, value in payload.get("documents", {}).items()
            },
            "evaluation_note": (
                "Use this metadata with model_calls, node_results, worksheet row_origin flags, "
                "and action register for later LLM/agent performance evaluation. "
                "Precision, recall, F1, and repeatability require expert-reference data and repeated runs."
            ),
        })
        add_event(job_id, "start", "HAZOP workflow started.")
        ensure_not_stopped(job_id)
        pid_parse: Dict[str, Any] = {}
        if payload["documents"].get("pid_images"):
            set_progress(job_id, "pid_image_parse", "P&ID Image Parser is reading diagram images.", 1)
            pid_parse = parse_pid_images(payload)
            set_result(job_id, ["pid_parse"], pid_parse)
            complete_latest_step(job_id, "pid_image_parse", "P&ID Image Parser is reading diagram images.")
            succeeded = (pid_parse or {}).get("vision_parse_succeeded", False)
            add_event(job_id, "P&ID parser", f"P&ID image parser {'completed successfully' if succeeded else 'used fallback (text context only)'}.")
        ensure_not_stopped(job_id)
        set_progress(job_id, "process_understanding", "Panel Chair is building shared process and P&ID understanding.", 2)
        process_understanding = build_process_understanding(payload)
        payload["documents"]["process_understanding"] = compact_json(process_understanding, 90000)
        set_result(job_id, ["process_understanding"], process_understanding)
        complete_latest_step(job_id, "process_understanding", "Panel Chair is building shared process and P&ID understanding.")
        add_event(job_id, "process understanding", "Panel Chair completed the shared factual basis for process flow, procedure, node scope, and guidewords.")
        ensure_not_stopped(job_id)
        set_progress(job_id, "planning", "Panel Chair is defining HAZOP nodes.", 4)
        planning = plan_nodes(payload)
        nodes = attach_evidence_safeguards_to_nodes(
            planning["nodes"],
            process_understanding,
            pid_parse,
        )
        planning["nodes"] = nodes
        set_result(job_id, ["planning"], planning)
        set_result(job_id, ["nodes"], nodes)
        complete_latest_step(job_id, "planning", "Panel Chair is defining HAZOP nodes.")
        add_event(job_id, "planning", f"Panel Chair defined {len(nodes)} node(s).")
        ensure_not_stopped(job_id)
        worksheet: List[Dict[str, Any]] = []
        review_decision_trace: List[Dict[str, Any]] = []
        round_3_adjudication_trace: List[Dict[str, Any]] = []
        round_3_expected_candidate_count = 0
        round_3_escalated_proposals: List[Dict[str, Any]] = []
        round_3_new_proposal_deliberations: List[Dict[str, Any]] = []
        total_nodes = max(1, len(nodes))
        for node_index, node in enumerate(nodes, 1):
            ensure_not_stopped(job_id)
            base = 8 + ((node_index - 1) / total_nodes) * 82
            node_label = f"{node['node_id']} - {node['node_name']}"
            add_event(job_id, "node", f"Starting {node_label}.")
            set_progress(job_id, f"{node['node_id']}-chair-effects", f"{node_label}: Chair defining guideword effects.", base, node["node_id"], "Chair setup")
            chair_effects = chair_define_effects_for_node(payload, node)
            node = chair_effects.get("node", node)
            nodes[node_index - 1] = node
            set_result(job_id, ["nodes"], nodes)
            complete_latest_step(job_id, f"{node['node_id']}-chair-effects", f"{node_label}: Chair defining guideword effects.")
            add_event(job_id, "chair setup", f"{node_label}: Chair defined {len(guideword_parameter_matrix(node))} guideword/effect item(s).")
            ensure_not_stopped(job_id)
            set_progress(job_id, f"{node['node_id']}-round1", f"{node_label}: Round 1 expert elicitation.", base + 8 / total_nodes, node["node_id"], "Round 1")
            round_one = round_one_for_node(payload, node)
            merged = merge_round_one(payload, node, round_one)
            draft_rows = merged.get("draft_rows", [])
            raw_round_one_candidates = round_one_atomic_candidates(node, round_one)
            review_rows = round_one_review_universe(
                node,
                round_one,
                draft_rows,
                raw_round_one_candidates,
            )
            complete_latest_step(job_id, f"{node['node_id']}-round1", f"{node_label}: Round 1 expert elicitation.")
            add_event(
                job_id,
                "round 1",
                (
                    f"{node_label}: generated {len(raw_round_one_candidates)} cause-level expert "
                    f"candidate(s); the Chair consolidated them into {len(review_rows)} unique "
                    "scenario(s) for panel review."
                ),
            )
            ensure_not_stopped(job_id)
            set_progress(job_id, f"{node['node_id']}-round2", f"{node_label}: Round 2A exhaustive review of {len(review_rows)} unique reconciled Round 1 scenarios, followed by Round 2B refinement of all non-unanimous-Agree findings.", base + 27 / total_nodes, node["node_id"], "Round 2")
            challenges, deliberation_summary, new_scenario_deliberations = round_two_for_node(
                payload, node, draft_rows, round_one, review_rows
            )
            node_review_trace = build_round_two_review_trace(
                node,
                review_rows,
                challenges,
                new_scenario_deliberations,
            )
            round_two_scenario_digest = build_round_two_chair_digest(
                challenges,
                new_scenario_deliberations,
            )
            review_decision_trace.extend(node_review_trace)
            set_result(job_id, ["review_decision_trace"], review_decision_trace)
            complete_latest_step(job_id, f"{node['node_id']}-round2", f"{node_label}: Round 2A review and Round 2B discussion completed.")
            add_event(
                job_id,
                "round 2",
                (
                    f"{node_label}: {len(challenges)} agents each reviewed all {len(review_rows)} "
                    "unique reconciled Round 1 scenarios; "
                    f"{deliberation_summary['round_2a_missing_review_count']} assignments were incomplete, "
                    f"and {deliberation_summary['round_2b_discussed_scenario_count']} scenarios requiring refinement "
                    "entered Round 2B. Pairwise decision agreement changed from "
                    f"{deliberation_summary['round_2a_pairwise_agreement_rate']:.1%} to "
                    f"{deliberation_summary['round_2b_pairwise_agreement_rate']:.1%}; Fleiss' kappa "
                    f"changed from {deliberation_summary['round_2a_fleiss_kappa']:.3f} to "
                    f"{deliberation_summary['round_2b_fleiss_kappa']:.3f}. The panel reviewed "
                    f"{deliberation_summary['new_scenario_proposal_count']} proposed additions and "
                    f"supported {deliberation_summary['panel_supported_new_scenario_count']}."
                ),
            )
            ensure_not_stopped(job_id)
            set_progress(job_id, f"{node['node_id']}-round3", f"{node_label}: Round 3 chair finalization.", base + 54 / total_nodes, node["node_id"], "Round 3")
            refined = refine_round_two(
                payload,
                node,
                review_rows,
                challenges,
                review_rows,
                new_scenario_deliberations,
                round_two_scenario_digest,
            )
            revised_rows = refined.get("revised_rows", [])
            node_round_3_trace = refined.get("round_3_adjudication_trace", [])
            node_round_3_proposals = refined.get("round_3_escalated_proposals", [])
            node_round_3_proposal_deliberations = refined.get(
                "round_3_new_proposal_deliberations", []
            )
            round_3_escalated_proposals.extend(node_round_3_proposals)
            round_3_new_proposal_deliberations.extend(
                node_round_3_proposal_deliberations
            )
            round_3_expected_candidate_count += int(
                (refined.get("round_3_adjudication_summary") or {}).get(
                    "expected_candidate_count", len(node_round_3_trace)
                )
            )
            round_3_adjudication_trace.extend(node_round_3_trace)
            set_result(
                job_id,
                ["round_3_adjudication_trace"],
                round_3_adjudication_trace,
            )
            set_result(
                job_id,
                ["round_3_escalated_proposals"],
                round_3_escalated_proposals,
            )
            set_result(
                job_id,
                ["round_3_new_proposal_deliberations"],
                round_3_new_proposal_deliberations,
            )
            set_result(
                job_id,
                ["round_3_adjudication_summary"],
                build_round_three_adjudication_summary(
                    round_3_adjudication_trace,
                    round_3_expected_candidate_count,
                ),
            )
            final_rows = scribe_rows(payload, node, revised_rows)
            final_rows = repair_not_applicable_rows_from_fallbacks(
                final_rows,
                node,
                revised_rows,
                draft_rows,
                rows_from_round_one(node, round_one),
                rows_from_new_scenario_deliberations(node, new_scenario_deliberations),
            )
            final_rows = finalize_node_rows(final_rows, node, payload["documents"])
            if refined.get("chair_finalization_failed"):
                add_event(
                    job_id,
                    "worksheet fallback",
                    (
                        f"{node_label}: no unadjudicated candidate was promoted after the Chair "
                        "failed to complete the evidence-grounded four-check rubric; no final worksheet rows were authorized."
                    ),
                )
            elif refined.get("chair_finalization_incomplete"):
                missing_count = len(refined.get("round_3_missing_candidate_ids", []) or [])
                add_event(
                    job_id,
                    "worksheet partial",
                    (
                        f"{node_label}: completed Chair decisions were preserved and "
                        f"{missing_count} candidate(s) still incomplete after three targeted "
                        "attempts were excluded with an audit record."
                    ),
                )
            final_rows, row_lineage = attach_final_row_lineage(
                node,
                final_rows,
                review_rows,
                challenges,
                new_scenario_deliberations,
            )
            for lineage_entry in row_lineage:
                append_result(job_id, ["performance", "row_lineage"], lineage_entry)
            worksheet.extend(final_rows)
            final_display_rows = consolidate_worksheet_rows(final_rows)
            worksheet_display = consolidate_worksheet_rows(worksheet)
            node_result = {
                "node": node,
                "rounds": {
                    "chair_effect_definition_summary": chair_effects.get("chair_effect_definition_summary", []),
                    "round_1_chair_opening": merged.get("round_1_chair_opening", []),
                    "round_1_summary": normalize_whitespace(merged.get("round_1_summary", "")),
                    "round_2_summary": (
                        f"Round 2A exhaustive review completed: {len(challenges)} agents each reviewed "
                        f"{len(review_rows)} unique reconciled, cause-level scenarios derived from "
                        f"{len(raw_round_one_candidates)} raw expert candidates. Round 2B reconsidered "
                        f"{deliberation_summary['round_2b_discussed_scenario_count']} scenarios requiring "
                        "post-review refinement; "
                        f"pairwise decision agreement changed from "
                        f"{deliberation_summary['round_2a_pairwise_agreement_rate']:.1%} to "
                        f"{deliberation_summary['round_2b_pairwise_agreement_rate']:.1%}; Fleiss' "
                        f"kappa changed from {deliberation_summary['round_2a_fleiss_kappa']:.3f} to "
                        f"{deliberation_summary['round_2b_fleiss_kappa']:.3f}. The full panel reviewed "
                        f"{deliberation_summary['new_scenario_proposal_count']} proposed new scenarios; "
                        f"{deliberation_summary['panel_supported_new_scenario_count']} were panel-supported, "
                        f"{deliberation_summary['contested_new_scenario_count']} retained documented dissent, and "
                        f"{deliberation_summary['rejected_new_scenario_count']} concluded Disagree."
                    ),
                    "round_3_chair_summary": normalize_whitespace(refined.get("round_3_chair_summary", "")),
                    "round_3_chair_synthesis": refined.get("round_3_chair_synthesis", []),
                },
                "chair_effect_definitions": chair_effects.get("effect_definitions", []),
                "round_1": round_one,
                "round_1_raw_candidate_count": len(raw_round_one_candidates),
                "round_1_reconciled_review_count": len(review_rows),
                "round_1_review_universe": review_rows,
                "round_2": challenges,
                "round_2_deliberation_summary": deliberation_summary,
                "round_2_new_scenario_deliberations": new_scenario_deliberations,
                "round_2_scenario_deliberations": round_two_scenario_digest.get(
                    "scenario_deliberations", []
                ),
                "round_2_review_trace": node_review_trace,
                "round_2_review_methodology": json_safe(ROUND_TWO_DECISION_METHOD),
                "round_3_adjudication_trace": node_round_3_trace,
                "round_3_adjudication_summary": refined.get(
                    "round_3_adjudication_summary", {}
                ),
                "round_3_escalated_proposals": node_round_3_proposals,
                "round_3_new_proposal_deliberations": node_round_3_proposal_deliberations,
                "round_3_chair_methodology": json_safe(ROUND_THREE_DECISION_METHOD),
                "round_3_evidence_manifest": round_three_evidence_manifest(
                    node, payload.get("documents", {})
                ),
                "round_3_review_attempts": refined.get("round_3_review_attempts", 0),
                "round_3_model_call_count": refined.get("round_3_model_call_count", 0),
                "round_3_batch_count": refined.get("round_3_batch_count", 0),
                "round_3_missing_candidate_ids": refined.get(
                    "round_3_missing_candidate_ids", []
                ),
                "chair_finalization_failed": bool(
                    refined.get("chair_finalization_failed", False)
                ),
                "chair_finalization_incomplete": bool(
                    refined.get("chair_finalization_incomplete", False)
                ),
                "chair_finalized_rows": revised_rows,
                "final_rows": final_rows,
                "final_display_rows": final_display_rows,
            }
            append_result(job_id, ["node_results"], node_result)
            # ``worksheet`` is the non-repetitive human-facing table.
            # ``worksheet_atomic`` retains one record per cause for metrics,
            # Round 2 auditability, and exact chain-level traceability.
            set_result(job_id, ["worksheet"], worksheet_display)
            set_result(job_id, ["worksheet_display"], worksheet_display)
            set_result(job_id, ["worksheet_atomic"], worksheet)
            complete_latest_step(job_id, f"{node['node_id']}-round3", f"{node_label}: Round 3 chair finalization.")
            add_event(
                job_id,
                "round 3",
                (
                    f"{node_label}: closed with {len(final_rows)} atomic scenario row(s), "
                    f"consolidated into {len(final_display_rows)} final worksheet row(s)."
                ),
            )
        ensure_not_stopped(job_id)
        set_progress(
            job_id,
            "synthesis",
            "Deterministic formatting is compiling the worksheet and action register; the Scribe is summarizing the final report.",
            94,
        )
        actions = build_action_register(worksheet)
        worksheet_display = consolidate_worksheet_rows(worksheet)
        report = build_final_report(payload, nodes, worksheet_display, actions)
        set_result(job_id, ["worksheet"], worksheet_display)
        set_result(job_id, ["worksheet_display"], worksheet_display)
        set_result(job_id, ["worksheet_atomic"], worksheet)
        set_result(job_id, ["review_decision_trace"], review_decision_trace)
        set_result(job_id, ["round_3_adjudication_trace"], round_3_adjudication_trace)
        set_result(job_id, ["round_3_escalated_proposals"], round_3_escalated_proposals)
        set_result(
            job_id,
            ["round_3_new_proposal_deliberations"],
            round_3_new_proposal_deliberations,
        )
        set_result(
            job_id,
            ["round_3_adjudication_summary"],
            build_round_three_adjudication_summary(
                round_3_adjudication_trace,
                round_3_expected_candidate_count,
            ),
        )
        set_result(job_id, ["action_register"], actions)
        set_result(job_id, ["report"], report)
        complete_latest_step(
            job_id,
            "synthesis",
            "Deterministic formatting compiled the worksheet and action register; the Scribe summarized the final report.",
        )
        set_progress(job_id, "completed", "HAZOP completed.", 100)
        complete_latest_step(job_id, "completed", "HAZOP completed.")
        add_event(
            job_id,
            "completed",
            (
                f"Completed {len(nodes)} node(s), {len(worksheet)} atomic scenario row(s), "
                f"{len(worksheet_display)} final worksheet row(s), and {len(actions)} action item(s)."
            ),
        )
        set_status(job_id, "completed")
        save_performance_snapshot_event(
            job_id,
            "completed",
            "Saved job performance data to",
            "Could not save job performance data",
        )
    except JobStopped:
        with JOBS_LOCK:
            result = JOBS[job_id].get("result", {})
            completed_nodes = len(result.get("node_results", []) or [])
            worksheet_rows = len(result.get("worksheet", []) or [])
            atomic_rows = len(result.get("worksheet_atomic", []) or [])
        mark_job_stopped(
            job_id,
            (
                f"HAZOP stopped after {completed_nodes} completed node(s), {worksheet_rows} final "
                f"worksheet row(s), and {atomic_rows} atomic scenario row(s). Partial results are available."
            ),
        )
        save_performance_snapshot_event(
            job_id,
            "stopped",
            "Saved partial job performance data to",
            "Could not save partial job performance data",
        )
    except Exception as error:
        trace = traceback.format_exc(limit=8)
        add_event(job_id, "failed", f"Workflow failed: {error}")
        set_status(job_id, "failed", f"{error}\n\n{trace}")
        set_progress(job_id, "failed", "Workflow failed. See report tab for debug trace.", 100)
        def fail_running(job):
            for step in job["steps"]:
                if step["status"] == "running":
                    step["status"] = "failed"
        update_job(job_id, fail_running)
        save_performance_snapshot_event(
            job_id,
            "failed",
            "Saved failed job performance data to",
            "Could not save failed job performance data",
        )


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────────────

def frontend_agents() -> List[Dict[str, str]]:
    return [
        {"id": agent_id, "label": label, "description": description}
        for agent_id, label, description in AGENTS
    ]


class HAZOPHandler(SimpleHTTPRequestHandler):
    server_version = "HAZOPAgentPanel/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", "time": utc_now()})
            return
        if parsed.path == "/api/models":
            self._send_json(HTTPStatus.OK, {
                "models": AVAILABLE_MODELS,
                "default_models": DEFAULT_MODELS,
                "default_provider": "openai",
                "agents": frontend_agents(),
                "thinking_levels": ["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
                "temperature_range": {
                    "min": 0,
                    "max": 2,
                    "default": DEFAULT_TEMPERATURE,
                    "vision_default": DEFAULT_VISION_TEMPERATURE,
                },
                "hazop_guidewords": [{"guideword": word, "meaning": meaning} for word, meaning in HAZOP_GUIDEWORDS],
                "hazop_parameters": HAZOP_PARAMETERS,
                "worksheet_columns": [
                    "guideword", "deviation", "potential_cause", "consequence", "safeguard",
                    "necessary_action",
                ],
            })
            return
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/report.docx"):
            parts = parsed.path.strip("/").split("/")
            job_id = parts[2] if len(parts) >= 4 else ""
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                snapshot = json_safe(job) if job else None
            if not snapshot:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                return
            docx = docx_report_bytes(snapshot)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            self.send_header("Content-Disposition", f'attachment; filename="haznova_hazop_{job_id[:8]}.docx"')
            self.send_header("Content-Length", str(len(docx)))
            self.end_headers()
            self.wfile.write(docx)
            return
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/worksheet.xlsx"):
            parts = parsed.path.strip("/").split("/")
            job_id = parts[2] if len(parts) >= 4 else ""
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                snapshot = json_safe(job) if job else None
            if not snapshot:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                return
            xlsx = xlsx_workbook_bytes(snapshot)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", f'attachment; filename="haznova_hazop_{job_id[:8]}.xlsx"')
            self.send_header("Content-Length", str(len(xlsx)))
            self.end_headers()
            self.wfile.write(xlsx)
            return
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/status"):
            parts = parsed.path.strip("/").split("/")
            job_id = parts[2] if len(parts) >= 4 else ""
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                snapshot = job_status_snapshot(job) if job else None
            if not snapshot:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                return
            self._send_json(HTTPStatus.OK, snapshot)
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.split("/")[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                snapshot = json_safe(job) if job else None
            if not snapshot:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                return
            self._send_json(HTTPStatus.OK, snapshot)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/stop"):
            parts = parsed.path.strip("/").split("/")
            job_id = parts[2] if len(parts) >= 4 else ""
            with JOBS_LOCK:
                exists = job_id in JOBS
            if not exists:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                return
            job = request_stop_job(job_id)
            self._send_json(HTTPStatus.OK, {"job": job})
            return
        if parsed.path != "/api/hazop":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8"))
            normalized = validate_payload(payload)
            job = create_job(normalized)
            initial_snapshot = json_safe(job)
            with JOBS_LOCK:
                JOBS[job["id"]] = job
            worker = threading.Thread(target=run_hazop_job, args=(job["id"], normalized), daemon=True)
            worker.start()
            self._send_json(
                HTTPStatus.ACCEPTED,
                {"job_id": job["id"], "job": initial_snapshot},
            )
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body"})
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def end_headers(self):
        self._send_cors_headers()
        super().end_headers()

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        response = json_dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        return


def find_free_port(host: str, preferred_port: int, attempts: int = 250) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found from {preferred_port} to {preferred_port + attempts - 1}")


def run_server() -> None:
    preferred_port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    if not STATIC_DIR.exists():
        raise SystemExit(f"Static directory not found: {STATIC_DIR}")
    port = find_free_port(HOST, preferred_port)
    url = f"http://{HOST}:{port}"
    try:
        server = ThreadingHTTPServer((HOST, port), HAZOPHandler)
    except OSError as error:
        raise SystemExit(f"Could not start Haznova backend on {url}: {error}") from error
    print(f"Haznova backend running at {url}")
    if port != preferred_port:
        print(f"Port {preferred_port} was unavailable; selected {port}.")
    print("Keep this terminal window open while using the website.")
    if os.environ.get("HAZNOVA_OPEN_BROWSER", "1").lower() not in {"0", "false", "no"}:
        def open_browser_quietly() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Timer(0.6, open_browser_quietly).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
