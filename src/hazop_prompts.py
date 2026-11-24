"""Prompt templates and prompt builders for the Haznova HAZOP backend.

This module intentionally contains prompt text only. Runtime helper functions
from hazop_backend are injected with configure_prompt_helpers() to avoid a
circular import while keeping the backend orchestration file smaller.
"""

import json
from typing import Any, Callable, Dict, List


# Five HAZOP safeguard classes: inherent, spatial, passive, active, procedural.
# These are search targets, never evidence that a control exists or is effective.
SAFEGUARD_SEARCH_TAXONOMY: Dict[str, Dict[str, str]] = {
    "inherent": {
        "pid": "only explicitly annotated elimination, substitution with a less hazardous material, or minimized hazardous inventory; drawing position alone cannot establish inherent safety",
        "text": "documented elimination or substitution of a hazard, reduced hazardous inventory, or another design change that removes or minimizes the hazard",
    },
    "spatial": {
        "pid": "only explicitly annotated physical separation, segregation, or separation distance; schematic spacing on the page is not physical siting evidence",
        "text": "documented physical separation, segregated equipment or storage, distance from occupied areas, or remote siting that limits exposure or escalation",
    },
    "passive": {
        "pid": "bund/dike, double containment, seal pot, closed drain, blast/fire barrier, installed blind, restriction orifice, rupture disk, flame arrester, or fixed bonding; verify the visible function and connection",
        "text": "documented containment and barriers, passive flow restriction, rupture disk, flame arrester, installed positive isolation blind, or fixed grounding/bonding",
    },
    "active": {
        "pid": "pressure/vacuum relief valve, non-return or excess-flow valve, regulator, control loop, alarm, detector, trip, interlock, shutdown, emergency isolation, inerting, ventilation, firewater/deluge, flare/scrubber, or standby utility; identify the connected protective function",
        "text": "documented actuated or operator-actuated protection, relief, detection with a defined response, control loops, trips, interlocks, inerting, emergency cooling, fire protection, or backup utility",
    },
    "procedural": {
        "pid": "only a protective operator action explicitly written on the drawing; a valve label or symbol by itself is not a procedure or safeguard",
        "text": "documented line-up check, independent verification, operating or emergency procedure, permit-to-work/LOTO, communication and read-back, inspection/testing, or task-specific protective work practice",
    },
}


def safeguard_taxonomy_prompt(source: str) -> str:
    """Describe what to look for in a source, without asserting existence."""
    return "\n".join(
        f"- {category.replace('_', ' ').title()}: {examples[source]}"
        for category, examples in SAFEGUARD_SEARCH_TAXONOMY.items()
    )


def _unconfigured_helper(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("hazop_prompts.configure_prompt_helpers() was not called.")


AGENTS: List[tuple[str, str, str]] = []
compact_json: Callable[..., str] = _unconfigured_helper
clamp_text: Callable[..., str] = _unconfigured_helper
study_scope_summary: Callable[..., Dict[str, Any]] = _unconfigured_helper
document_context: Callable[..., str] = _unconfigured_helper
guideword_matrix_text: Callable[..., str] = _unconfigured_helper
guideword_parameter_matrix: Callable[..., List[Dict[str, str]]] = _unconfigured_helper
node_scope_gate_text: Callable[..., str] = _unconfigured_helper
normalize_whitespace: Callable[..., str] = _unconfigured_helper


def configure_prompt_helpers(
    *,
    agents: List[tuple[str, str, str]],
    compact_json_fn: Callable[..., str],
    clamp_text_fn: Callable[..., str],
    study_scope_summary_fn: Callable[..., Dict[str, Any]],
    document_context_fn: Callable[..., str],
    guideword_matrix_text_fn: Callable[..., str],
    guideword_parameter_matrix_fn: Callable[..., List[Dict[str, str]]],
    node_scope_gate_text_fn: Callable[..., str],
    normalize_whitespace_fn: Callable[..., str],
) -> None:
    global AGENTS, compact_json, clamp_text, study_scope_summary, document_context
    global guideword_matrix_text, guideword_parameter_matrix, node_scope_gate_text, normalize_whitespace
    AGENTS = agents
    compact_json = compact_json_fn
    clamp_text = clamp_text_fn
    study_scope_summary = study_scope_summary_fn
    document_context = document_context_fn
    guideword_matrix_text = guideword_matrix_text_fn
    guideword_parameter_matrix = guideword_parameter_matrix_fn
    node_scope_gate_text = node_scope_gate_text_fn
    normalize_whitespace = normalize_whitespace_fn


# ─────────────────────────────────────────────────────────────────────────────
# P&ID vision prompt
# ─────────────────────────────────────────────────────────────────────────────

def pid_image_parser_system_prompt() -> str:
    """
    Dedicated system prompt for the vision agent that explicitly demands JSON only.
    Many vision models (especially GPT-4o) add preamble prose before the JSON block.
    This prompt suppresses that.
    """
    return (
        "You are the P&ID Image Parser Agent for an automated HAZOP study. "
        "You must extract only information grounded in the supplied P&ID image. "
        "Process description text is supporting context only: use it only to clarify a visible object, label, stream, or condition. "
        "Never create equipment, stream names, line numbers, controls, or node boundaries from text unless the corresponding object or connection is visible in the image. "
        "Do not use outside sources, generic plant assumptions, invented tags, or assumed equipment numbering. "
        "Your ONLY output must be a single valid JSON object. "
        "Do NOT include any explanatory text, markdown fences, or preamble before or after the JSON. "
        "Do NOT use ```json or ``` wrappers. "
        "Start your response with { and end with }. "
        "If you cannot read a tag, use 'UNKNOWN' as the value. "
        "If you cannot determine flow direction, use 'unknown'. "
        "Never leave a field undefined — use empty string or empty array instead."
    )


def pid_image_parser_prompt(documents: Dict[str, Any]) -> str:
    """
    Prompt places the JSON schema at the top
    so vision models see the required output format before processing the image context.
    This prevents models from generating a prose description instead of JSON.
    """
    support_context = {
        "image_sources_in_upload_order": list(documents.get("pid_image_sources") or []),
        "process_description_filename": documents.get("process_description_filename", ""),
        "process_description": clamp_text(documents.get("process_description", ""), 12000),
        "safeguards_and_controls": clamp_text(documents.get("safeguards", ""), 8000),
        "operating_conditions": clamp_text(documents.get("operating_conditions", ""), 8000),
        "text_extracted_from_uploads": clamp_text(documents.get("pid", ""), 12000),
    }
    return f"""
OUTPUT FORMAT — return exactly this JSON structure and nothing else:
{{
  "parsed_pid_summary": "one sentence overview using only visible image content",
  "context_used": ["short bullets naming any process-description facts used to clarify visible image objects"],
  "context_not_used": ["short bullets naming any text facts not used because the corresponding object/connection was not visible"],
  "equipment": [
    {{"tag": "visible equipment tag or empty string", "type": "visible equipment type or unknown", "description": "visible label/function or empty string", "image_evidence": "visible tag/label/symbol/location supporting this item"}}
  ],
  "lines": [
    {{"line_id": "visible line ID or empty string", "from": "visible upstream tag/label or empty string", "to": "visible downstream tag/label or empty string", "material_or_service": "visible service/material label or empty string", "flow_direction": "visible arrow direction or unknown", "image_evidence": "visible pipe/arrow/label evidence"}}
  ],
  "process_streams": [
    {{"name": "visible stream/service label or empty string", "from": "visible source tag/label or empty string", "to": "visible destination tag/label or empty string", "material": "visible material label or empty string", "line_id": "visible line ID or empty string", "flow_direction": "visible arrow direction or unknown", "image_evidence": "visible stream label/pipe/arrow evidence"}}
  ],
  "instruments_and_controls": [
    {{"tag": "visible instrument tag or empty string", "type": "visible instrument type or unknown", "associated_equipment_or_line": "visible associated item or empty string", "image_evidence": "visible instrument bubble/tag/connection evidence"}}
  ],
  "valves_and_relief_devices": [
    {{"tag": "visible valve/relief tag or empty string", "type": "visible valve/relief type or unknown", "location": "visible location or empty string", "image_evidence": "visible valve/relief symbol/tag/location evidence"}}
  ],
  "existing_safeguards": [
    {{"name": "concise visible safeguard/control name", "tag": "visible tag or empty string", "type": "visible device or measure type, such as relief valve, rupture disk, non-return valve, restriction orifice, control loop, alarm, trip, bund, or explicitly annotated procedure", "protective_function": "preventive | detective | control | mitigative function supported by visible symbols/connections only", "associated_equipment_or_line": "visible protected equipment/line/location", "image_evidence": "specific visible symbol, tag, line, loop, annotation, or connection supporting the safeguard"}}
  ],
  "candidate_hazop_nodes": [
    {{"node_id": "N-001", "node_name": "visible equipment/section name", "boundary": "boundary visible from image only", "design_intent": "intent visible from image labels, flow arrows, equipment function, and line evidence only", "image_evidence": "visible equipment/line/label evidence supporting this node boundary"}}
  ],
  "uncertainties": ["list any symbols or tags that were ambiguous or unreadable"]
}}

TASK:
Parse the P&ID image(s) provided. Identify process equipment, piping lines, instruments, control loops, valves, relief devices, flow arrows, labels, and safety devices that are visible in the image.

{prompt_method_for_stage("pid_parser")}

SUPPORTING CONTEXT:
{json.dumps(support_context, ensure_ascii=True, indent=2)}

GROUNDING RULES:
- The P&ID image is the controlling evidence.
- Images are supplied in the order shown by image_sources_in_upload_order. For every image-derived item, begin image_evidence with "Image 1:", "Image 2:", etc., so a reader can identify the source drawing. Keep controls from different arrangements separate unless the drawing evidence explicitly connects them. If two drawings conflict, report the conflict in uncertainties rather than combining their claims.
- The process description may clarify visible objects only. If an object or line is visible but its label is partly ambiguous, use the process description to clarify it; if the object or line is not visible or supplied, do not create it.
- Do not create equipment, line IDs, stream IDs, tags, instruments, safeguards, or HAZOP node boundaries from text alone.
- Do not use outside sources or typical plant assumptions.
- Preserve visible P&ID tags and line labels exactly as they appear in the current drawing. Do not normalize, renumber, or replace them with invented display names.
- Do not let high-level process stream summaries bypass inline valves, pumps, exchangers, columns, receivers, or other visible equipment in parsed lines or candidate nodes.
- If the text mentions a fact but the matching object/connection is not visible, put that fact in context_not_used and do not add it to equipment, lines, or candidate_hazop_nodes.
- Every object and candidate node must have visible image evidence. If evidence is weak or unreadable, include it in uncertainties instead of inventing a confident label.
- Populate existing_safeguards independently of the process-description safeguard list. Inspect the full drawing using this P&ID safeguard search checklist (examples are not claims of presence):
{safeguard_taxonomy_prompt("pid")}
  The type field names the actual device or measure; the backend assigns the five-class HAZOP category from its documented mechanism. Do not infer physical separation from schematic spacing, or infer inherent hazard reduction from the mere presence of nitrogen or a lower operating pressure.
  Include P&ID-only safeguards absent from the text, but do not classify an ordinary valve, instrument, pipe, or vessel as a safeguard unless its visible symbol, tag, loop, annotation, connection, or stated function supports a protective role.
- Preserve visible safeguard tags and locations. State the protective function conservatively from the visible variable, loop, device type, and connection; do not invent setpoints, voting logic, trip actions, alarm priorities, or independence that the drawing does not show.
- A P&ID symbol establishes that equipment/control is shown, not that it is adequate or independent. Do not infer capacity, setpoint, response time, proof testing, SIL/IPL credit, fail position, or operator response unless explicitly visible.

NODE IDENTIFICATION RULES:
- For full P&ID or PFD-style process systems, define nodes the way a HAZOP chair would manually define them: one node per process function/equipment section and connecting line boundary.
- Use only visible equipment labels, service labels, flow arrows, instruments, safeguards, and visible process connections.
- For labels and tags, use ONLY labels or tags visible in the drawing. Do not create display tags such as TK-1, T-101, R-101, STREAM-001, PRODUCT_STORAGE, or REACTOR unless that exact label is visible in the image.
- If equipment is visible but has no readable tag, set tag to an empty string and use a descriptive name such as "unlabeled vessel", "unlabeled pump", or the visible service/material label. Do not invent tag numbers.
- For process lines with no visible line number, use the visible service/material name as line_id if available, otherwise leave line_id empty and describe the connection in material_or_service.
- Typical node boundaries may include feed/preheat, reactor/tower/vessel body, overhead/relief/condensing, reflux drum/pump/reflux return, bottoms/product transfer, utility exchangers, blowdown/vent/drain sections, and storage-transfer lines.
- Do not collapse a complex process system into one node.
- Do not infer feed/product nodes, chemical names, equipment tags, line IDs, or controls from the problem statement or process description unless the related item is visible in the image.

IMPORTANT:
- Only report equipment, lines, valves, instruments, and safeguards that are VISIBLE in the image.
- Do not invent tag numbers that are not visible in the image.
- Do not invent stream IDs, equipment IDs, or labels that are not visible in the image.
- Do not output labels such as TK-1, T-101, R-101, STREAM-001, PRODUCT_STORAGE, or REACTOR unless that exact label is visible.
- Use "UNKNOWN" for tags you cannot read clearly.
- Your entire response must be a single JSON object starting with {{ and ending with }}.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Pre-HAZOP process understanding and planning prompts
# ─────────────────────────────────────────────────────────────────────────────

def process_understanding_prompt(documents: Dict[str, str]) -> str:
    scope = study_scope_summary(documents)
    return f"""
You are the Panel Chair Agent preparing an automated HAZOP workshop.

{JSON_ONLY_RULE}

TASK: Build the shared factual understanding that every expert agent must use before HAZOP assessment starts.
Do not generate HAZOP worksheet rows yet. Catalog existing documented controls, but do not assign scenario-specific safeguards, causes, consequences, or actions yet.

{prompt_method_for_stage("process_understanding")}

Your output must explain, from the supplied P&ID/image parse and text files:
- what process or operating procedure is being studied;
- how material, energy, or signals flow through the system;
- what equipment, valves, instruments, utilities, and safeguards are visibly or textually supplied;
- what procedure steps, initial valve states, and operator actions apply, if this is a procedural HAZOP;
- what nodes, boundaries, design intentions, guidewords, parameters, worksheet headings, and supplied worksheet rows are provided by the user;
- what information is uncertain or must not be assumed.

Extraction discipline:
- Read the supplied text as a case record. Preserve all node definitions, design intentions, procedure steps, guideword/effect lists, initial states, operating conditions, and explicitly stated safeguards.
- Do not compress the procedure into a vague summary when ordered steps are supplied. Record the ordered steps in the JSON array.
- Do not reinterpret a guideword list into the default guideword list. If the user supplies guidewords such as "Out of Sequence (Too early)", "Rate (Too fast)", "Magnitude (More)", "Communication", "Reverse", or "Incomplete", preserve those exact guideword items and their effect/deviation basis.
- Do not infer worksheet content in this stage. If supplied worksheet rows exist, record that they exist as case evidence for the matching node/guideword only; later stages decide how to use that evidence.
- Use the safeguard taxonomy only as a search checklist. Search the supplied process description and procedure for:
{safeguard_taxonomy_prompt("text")}
  Absence from this list is not proof that a control exists, and appearance in a reference worksheet is only an unverified claim until corroborated by current-study evidence.
- For every safeguard claim, preserve its exact evidence channel and location. Classify it as documented_existing only when the supplied P&ID, process description, or procedure establishes the control and where it applies. Classify questions, recommendations, proposed controls, ambiguous symbols, and unsupported reference-row claims as unverified or recommended; never promote them to existing safeguards.

{common_prompt_rules()}

Critical rules:
- Use the uploaded `.txt` content as the governing source when it defines nodes, design intentions, steps, guidewords, or worksheet headings.
- Use parsed P&ID image evidence to understand visible equipment, piping, flow arrows, labels, instruments, and valves.
- Do not invent missing equipment, missing safeguards, missing valve states, missing guidewords, or missing operating conditions.
- If the uploaded `.txt` content includes a HAZOP Worksheets section, treat those rows as supplied case evidence for matching nodes and guideword items; do not copy unrelated worksheet rows across nodes.
- Keep external references separate from the worksheet unless an external web reference or DOI is actually used.
- This is a pre-study basis only; HAZOP deviations and worksheet rows come later.

Return this exact JSON structure:
{{
  "process_overview": [
    "Complete factual statement describing the facility, operation, or procedure being studied.",
    "Complete factual statement describing the study scope and whether the study is process-equipment HAZOP or procedural HAZOP."
  ],
  "material_and_energy_flow": [
    {{
      "path_name": "short path name",
      "material_or_service": "material, utility, or signal",
      "normal_direction": "from upstream item through intermediate items to downstream item",
      "controlling_items": ["visible valves, pumps, controllers, or manual actions controlling this path"],
      "evidence": ["P&ID label, visible tag, or supplied text fact supporting this path"]
    }}
  ],
  "equipment_and_control_roles": [
    {{
      "tag_or_name": "visible or supplied tag/name",
      "role": "factual role in the process or procedure",
      "evidence": ["specific visible or supplied evidence"]
    }}
  ],
  "procedure_basis": {{
    "initial_state": ["initial valve/equipment/material state supplied by the user"],
    "ordered_steps": [
      {{
        "step_no": "1",
        "action": "operator or process action stated in the supplied text",
        "purpose": "why this action is performed according to the supplied text",
        "affected_node_or_boundary": "node or section affected, if stated or directly inferable"
      }}
    ],
    "operator_roles": ["operator role or responsibility stated in the supplied text"]
  }},
  "hazop_basis": {{
    "study_scope": {compact_json(scope, 6000)},
    "worksheet_columns": ["Guideword", "Deviation / Effect", "Potential cause", "Consequence", "Safeguard", "Necessary Action"],
    "guideword_matrix": [
      {{
        "guideword": "guideword item to apply",
        "guideword_category": "base guideword category",
        "effect": "deviation/effect basis if supplied",
        "parameter": "parameter or procedural aspect",
        "meaning": "scope of this guideword item for the study"
      }}
    ],
    "supplied_nodes": [
      {{
        "node_id": "N-001",
        "node_name": "supplied or parsed node name",
        "boundary": "node boundary",
        "design_intent": "supplied design intention",
        "parameters": ["parameter names"],
        "study_focus": ["node-specific parameters or procedural aspects"]
      }}
    ]
  }},
  "known_safeguards": [
    {{
      "safeguard": "visible or supplied safeguard",
      "applies_to": "equipment, line, procedure, or node",
      "category": "inherent/passive | mechanical | instrumented control | alarm/detection | trip/interlock/isolation | relief/vent | containment/mitigation | backup utility | administrative/work-practice",
      "protective_role": "preventive | detective | control | mitigative",
      "evidence_channel": "P&ID | process description | operating procedure",
      "evidence": "verbatim or tightly paraphrased current-study evidence",
      "status": "documented_existing | questioned_unverified | recommended_unconfirmed",
      "tag": "visible or supplied tag, otherwise empty string"
    }}
  ],
  "uncertainties": ["specific uncertain or unreadable facts"],
  "do_not_assume": ["specific missing facts that later agents must not invent"]
}}

DOCUMENTS:
{document_context(documents)}
""".strip()


def planning_prompt(documents: Dict[str, str], max_nodes: int) -> str:
    pid_parse_status = ""
    if documents.get("pid_image_parser_output"):
        pid_parse_status = "P&ID image parsing succeeded. Use the parsed equipment and nodes from the image parser output to guide node definition."
    else:
        pid_parse_status = (
            "P&ID image parsing was not performed or failed. "
            "Define nodes from the available text descriptions of the process. "
            "Be conservative — define one node per major equipment group described."
        )
    scope = study_scope_summary(documents)

    return f"""
You are the Panel Chair Agent for a fully automated HAZOP study.

CRITICAL: Your response must be a single JSON object. Do not include any text before or after the JSON.
Do not use markdown fences. Start with {{ and end with }}.

TASK: Define HAZOP nodes from the supplied documents and produce design intent statements.
Limit output to at most {max_nodes} nodes covering the complete process in sequence.
These nodes will be studied with the guideword/parameter scope detected from the supplied problem statement, or with the default HAZOP guidewords when no specific scope is supplied.
If the supplied problem statement limits the study to one parameter, use only that parameter for every node.
If the supplied problem statement specifies a condition, scenario, operating mode, or case, keep the node parsing complete but focus the HAZOP analysis on that requested scope.
If shared_process_understanding is present in DOCUMENTS, use it as the controlling pre-study basis and do not contradict it.
If the uploaded text supplies explicit HAZOP nodes, node boundaries, design intentions, guideword/effect items, or procedural steps, reproduce those supplied nodes exactly within the max-node limit. Do not add, merge, rename, or reorder supplied nodes.

{prompt_method_for_stage("planning")}

{common_prompt_rules()}

Manual node-definition rules:
- If manual nodes are supplied, they are controlling. Use their exact node name, boundary, design intent, guideword matrix, and procedure-step scope.
- If no manual nodes are supplied, define nodes by process function and physical boundary, as a HAZOP chair would do from a P&ID/PFD.
- If image parser candidate nodes are available, use those visible P&ID candidates as the starting point for node definition.
- Include visible safeguards and instruments in the node context so later agents can reason about causes, consequences, and safeguards.
- Do not make one broad "overall process" node when the diagram shows multiple process sections.
- Use equipment, lines, valves, instruments, relief paths, utilities, and service labels to define boundaries.
- Good nodes are typically feed/preheat, major vessel/reactor/tower body, overhead/relief/condensing, reflux/recirculation, bottoms/product transfer, utility exchanger, blowdown/vent/drain, and storage-transfer sections.
- Include the relevant parameters for each node unless the user explicitly restricted the study scope.

P&ID PARSE STATUS: {pid_parse_status}

REQUESTED STUDY SCOPE:
{compact_json(scope, 8000)}

Return this exact structure:
{{
  "nodes": [
    {{
      "node_id": "N-001",
      "node_name": "short name",
      "node_description": "equipment and line section included",
      "boundaries": "from X to Y",
      "design_intent": "normal intended operation for this node",
      "equipment": ["tag1", "tag2"],
      "lines": ["line1", "line2"],
      "materials": ["chemical1"],
      "operating_conditions": "known normal T, P, flow",
      "parameters": ["flow", "pressure", "temperature", "level", "composition"]
    }}
  ],
  "assumptions": ["assumption 1", "assumption 2"],
  "warnings": ["warning if P&ID is incomplete or ambiguous"]
}}

DOCUMENTS:
{document_context(documents)}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Shared prompt output rules
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FORMAT_RULES = """
Output formatting rules:
- Discussion fields (Round 1 opening statements/findings, Round 2A reasons and evidence, Round 2B technical responses, and Round 3 Chair synthesis/resolutions): detailed, grammatically complete statements. Each bullet must express one coherent technical point and may contain several sentences when needed to explain the evidence, disagreement, revision, or final decision. Do not shorten discussion text to worksheet-cell length.
- Discussion bullets may be long. Preserve connected clauses and complete sentences; never split a statement at commas, semicolons, conjunctions, or arbitrary word limits. Start a new bullet only for a genuinely separate technical point.
- The discussion record must explain the final worksheet: identify the relevant scenario/row, describe the technical reasoning and evidence considered, and state what was retained, revised, rejected, or escalated.
- Worksheet cells (Deviation/Effect, Potential cause, Consequence, Safeguard, Necessary Action): compact industrial HAZOP phrases, never paragraphs or 1-2 sentence explanations. Each distinct cause/consequence/safeguard/action is its own separate array element -- never two unrelated ideas joined into one array string. One bullet = one atomic, standalone technical claim. The reverse also matters: do not split one continuous technical claim just because it contains internal punctuation or a conjunction. If a schema field is a string rather than an array, write one complete technical statement, not a short label.
  - Do not merge two independent causes, consequences, safeguards, or actions into one array item.
  - Do not split one continuous causal statement into multiple bullets only because it contains punctuation or a conjunction.
  - Treat schema text and formatting instructions as structure only; never reuse instruction wording as worksheet content.
- Worksheet cell length contract -- bullets per cell and normal words per bullet. Each row now represents exactly one cause; if a matrix item has more than one distinct credible cause, that is more than one row (see the sub-row rule below), not more bullets inside one cell:
  - Deviation / Effect: exactly 1 bullet, normally 5-12 words, naming the abnormal event. Shared by every sub-row of the same matrix item.
  - Potential cause: exactly 1 bullet, normally 8-18 words, naming this sub-row's one initiating mechanism.
  - Consequence: 1-2 bullets, normally 5-16 words each, naming the resulting process/safety impact of this sub-row's cause only.
  - Safeguard: 1-2 bullets, normally 6-18 words each, naming the credited protection for this sub-row's cause-consequence chain only.
  - Necessary Action: 1 bullet (rarely 2), normally 5-16 words each, naming the specific correction or review that closes this sub-row's own remaining gap.
- Bullet length quality: each bullet must be a complete technical phrase that can be understood by itself. Do not write one-word fragments, clipped sentence pieces, long explanatory sentences, or multi-clause paragraphs.
- Compact does not mean bare. Every bullet must name the specific actor, tag, equipment item, material, or mechanism. Generic operator/equipment/release labels are too compressed unless they name the case-specific item and mechanism; rewrite them with the missing specific detail, or omit them if no specific detail is supported by the evidence.
- If a claim is longer than the target, shorten it by removing filler and explanation while preserving the technical actor, object, action, and consequence -- never truncate mechanically and never split a claim at a comma just to hit a word count.
- No fixed bullet count: use exactly as many bullets as there are distinct, row-specific, evidence-supported points. One strong bullet beats several generic ones.
- No literal JSON/Python list syntax ("[", "]") in cell text -- return arrays in JSON, but the cell text itself must read as plain phrases.
- Avoid filler phrases in worksheet cells: "because", "which could", "leading to", "requires verification:", "the panel should", "before safeguards act", "due to the fact that", "in order to", "it is possible that", "based on the supplied information".
- No slash-alternatives ("open/closed") or bracketed placeholders unless that is the literal supplied tag/name.
- Every worksheet bullet must belong to the exact row's guideword and deviation/effect -- do not copy general node hazards into every row, and do not repeat the same cause/consequence/safeguard/action across rows unless specifically credible for each.
- Safeguard cell: if it names a real documented safeguard, do not also include "Not identified"/"None"/"N/A"/"Not applicable" in the same cell. Never write a procedure step label or future/intended procedure text as a safeguard -- name the credited protection using the exact device, interlock, or administrative control actually documented in the supplied case, or use "Not identified"; a verification need belongs in Necessary Action, not Safeguard.
- Safeguard cell: do not credit rehearsals, supervision, training, or general procedure compliance by themselves. A specific administrative control may be credited only when a current procedure/process record documents the control, its required action, and its applicability to this node and scenario. A worksheet entry is a claim to verify, not automatic proof that the control exists or works.
- Necessary Action must be row-specific. Only fall back to "Refer to engineering team for further study" when no obvious corrective action or responsible discipline can be identified from the supplied case.
- Do not copy wording from these instructions into worksheet cells. Use only the current case evidence and the row-specific guideword chain.
- Cell quality gate:
  - Cause must name the case-specific item and failure/error mechanism.
  - Consequence must name the case-specific resulting event, exposure, loss, or process impact.
  - Safeguard must name only a documented device, control, or administrative safeguard that mitigates this row, with enough context to show how it applies.
  - Necessary Action must name the row-specific correction, verification, or engineering review tied to the remaining gap.
""".strip()


ROW_AND_EVIDENCE_RULES = """
Row structure and evidence rules:
- Every row is a cause -> effect -> consequence chain. Guideword = the required guideword item only. Deviation/Effect = the abnormal event that guideword produces (not the guideword qualifier itself). Cause is acceptable only if it produces that exact effect. Consequence is acceptable only if it follows from that guideword/effect/cause. Necessary Action is acceptable only if it closes that row's specific gap.
- Uniform guideword rule: apply the same effect-identification sequence to every guideword item. First preserve the required guideword exactly, then derive the immediate physical or operational Deviation/Effect for that guideword, then identify matching causes, consequences, safeguards, and actions. Never use Not applicable as the starting answer for a required guideword; use it only after the physical/operational effect test shows no credible abnormal state for this node.
- Cross-guideword isolation rule: analyze one required matrix item at a time as a closed row chain. Do not use a cause, consequence, safeguard, or action from another guideword just because it is plausible somewhere in the same node. Before output, ask for each cell: "Does this item directly explain or mitigate this exact row's guideword-specific Deviation/Effect?" If not, remove it from this row and place it only in the correct guideword row when that row is being analyzed.
- Guideword ownership examples for procedural rows: communication failures belong only in Communication rows; reverse flow/backflow/NRV content belongs only in Reverse rows unless the supplied matrix combines Reverse with another guideword; overpressure/pressure-relief content belongs in Pressure rows; high/low flow rate content belongs in Rate rows; excessive/insufficient quantity or duration belongs in Magnitude rows; incomplete valve travel or incomplete execution belongs in Incomplete rows; wrong utility/material or unavailable supply belongs in Others rows. Out of Sequence rows should focus on sequence-caused effects such as no flow, residual air/material, contamination, or reverse flow only when that effect is the direct result of the sequence error.
- Safeguard scope: use a comprehensive search taxonomy, but never infer that a safeguard exists merely because that type is common in industry. Search the current P&ID for inherent/passive design features, relief/vent/vacuum devices, non-return/restriction/excess-flow devices, emergency isolation, alarms/indicators/detectors, control loops, trips/interlocks/SIS/ESD, containment/drainage/barriers, ignition controls, fire/explosion mitigation, and emergency/backup utilities. Search the current process description/procedure for specific line-up checks, valve identification, independent verification, permits/LOTO, designated operator roles, communication protocols, alarm-response steps, and emergency actions. Credit a result only when evidence confirms existence, node/location, and a preventive, detective/control, or mitigative mechanism for this exact cause-effect-consequence chain. A supplied worksheet entry is corroborating same-row evidence to assess, not unquestioned ground truth. Do not borrow facility-wide protections from another P&ID section merely because a hazard word matches. Do not infer setpoint, capacity, response time, reliability, independence, SIL, or IPL credit. Rehearsal, generic supervision, training, general procedure compliance, and operator awareness are not safeguards by themselves. Unverified or recommended controls belong in Necessary Action, not Safeguard.
- Sub-row rule: one or more rows per required guideword-matrix item, one row per distinct credible cause and its consequence/safeguard/action chain. Preserve a separate finding if any technical element has a different meaning; merge only complete-chain duplicates stated in different words. Preserve the supplied matrix_key on every row that shares it. Use the supplied node ID, guideword, parameter, equipment/line names, and tags exactly as given -- never rename, renumber, or invent them. Never copy illustrative example tags, materials, safeguards, or numbers from this prompt into a row.
- Cells in one row (cause, consequence, safeguard, action) must stay internally consistent with that row's own guideword/effect/cause -- never copied from another guideword row, another effect, another cause's row for the same matrix item, or another node.
- Deviation/Effect is the row's single defining abnormal event -- write exactly one statement and never multiple bullets. Preserve "or" only when the supplied case uses it as one combined effect; do not use placeholder alternatives. Every row's one cause must produce this one effect, and its consequence must follow from it; a row that actually has two distinct, unrelated effects is two different scenarios, not one row -- pick the single effect best supported by the evidence and treat any genuinely separate effect as its own guideword-matrix item instead of folding it into this row. (This is a different split from the sub-row rule above: split by effect only when the effect itself genuinely differs; multiple causes sharing the same effect stay under the same matrix_key as separate rows, not separate matrix items.)
- Reject and rewrite any Deviation/Effect that merely repeats the guideword/category words, such as a procedural qualifier without the actual abnormal event. The Deviation/Effect must state the row-specific event visible from the P&ID, procedure, node intent, or supplied worksheet evidence.
- Parameter-based studies: guideword is the modifier only, while Deviation/Effect combines that modifier with the scoped parameter. Do not repeat the full deviation phrase in the Guideword cell.
- Procedural studies: the Guideword cell carries the procedural qualifier. The Deviation/Effect cell is still the actual identified physical or operational effect -- never just the timing/rate/magnitude qualifier, and never the cause phrased as an effect.
- Manual HAZOP style for procedural effects: write the immediate abnormal state the chair can observe or infer, such as no service flow, high service flow, residual air/material, contamination, reverse flow, or pressure state. Do not write abstract timing text when the same row can be expressed as a physical effect.
- Deviation/Effect cannot be an initiating action or failed-state cause. In procedural rows, wording such as "operator opens/closes...", "valve opened before/after...", "valve left open/closed...", "valve stuck...", "controller fails...", or "operator delay..." belongs in Potential cause unless the supplied same-node worksheet explicitly uses that wording as the Effect. The Effect cell must state what abnormal process or operational state results.
- If the node has a Communication guideword, keep communication causes (miscommunication, radio failure, operator-message misunderstanding) only in the Communication row -- never under Out of Sequence, Rate, Magnitude, Reverse, Incomplete, or Others.
- No broad meta-rows such as "all relevant guidewords" or "various deviations" -- every row corresponds to exactly one supplied matrix item. A non-credible or out-of-boundary combination gets one "Not applicable" row; never convert a credible scenario to Not applicable just because a safeguard is uncertain.
- Credible-row completion rule: if a row has a real Deviation/Effect, the cause, consequence, safeguard, and action cells must be evaluated for that same row. Do not leave Consequence or Action as Not applicable in a credible row. If the cause or safeguard is not documented, use Not identified or a row-specific verification action, but the consequence must still describe the credible unprotected outcome of the row's own effect.
- Ground every scenario in the parsed P&ID, supplied process description, operating data, and safeguards text. If shared_process_understanding is present in DOCUMENTS, treat it as the controlling pre-HAZOP factual basis. If a manual/setup .txt is uploaded, treat its full text (not just the node summary) as governing for node boundaries, design intentions, procedure steps, guidewords, and worksheet headings -- including its manual_context fields. If the uploaded text has a "HAZOP Worksheets" section, use those rows only as evidence for the matching node/guideword item, never copied across unrelated nodes. A separately supplied "completed worksheet for evaluation only" is not generation input.
- Procedural nodes: procedure_initial_setup and procedure_steps are the authoritative sequence, but the node boundary is narrower than the full procedure -- use steps outside your node only as sequence context, never as a final cause/effect/safeguard/action unless explicitly within this node's boundary or design intent. Do not import causes/effects/safeguards/actions from elsewhere in the procedure just because a tag appears there. Use step numbers only in your reasoning/discussion -- final cells must read like a standard HAZOP table; never write a step label or the phrase "operating procedure" as a stand-in for a technical item. "Operational procedure", "procedure compliance", and "operator training" are not safeguards by themselves. Use a named procedural safeguard only when it is documented in the supplied case and fits the exact row. A need to change, repeat, or verify a procedure step belongs in Necessary Action, not Safeguard.
- Do not invent a cause, safeguard, tag, or condition that is not visible or supplied. Use "Not identified" for an unknown cause or missing documented safeguard; use "Not applicable" only for a non-credible row. Do not create a worksheet Assumptions column -- state a missing-data item once, in the affected cell or as a row-specific Necessary Action, not repeated on every row.
- Prefer the same technical wording when the same evidence recurs across repeated runs -- do not introduce stylistic-only variation. When a matrix item has multiple rows (multiple causes), list them in the same order the causes were identified.
- Pre-output gate -- before returning JSON, confirm for every row: (1) the guideword item is required for this node; (2) the deviation/effect is the actual abnormal event, not the guideword qualifier, and contains no leaked step label; never write "Effect not identified" when cause or consequence is populated; (3) every cause can produce that exact effect inside this node boundary; (4) every consequence follows from that guideword/effect/cause; (5) credit a safeguard only when it is row-specific and inside this node boundary or explicitly supplied for this same node/effect; use "Not identified" when no such safeguard is available; (6) every action addresses the exact gap left by cause/consequence/safeguard, and is row-specific, not generic procedure advice; (7) the row contains no communication/valve/step/equipment detail belonging to another guideword item or adjacent node.
""".strip()


JSON_ONLY_RULE = "CRITICAL: Return a single JSON object only. No text before or after. No markdown. Start with { and end with }."


def prompt_method_for_stage(stage: str) -> str:
    """
    Stage-specific prompt contract. The HAZOP workflow uses different prompt
    structures for extraction, planning, specialist review, chair moderation,
    final adjudication, and report formatting.
    """
    methods = {
        "pid_parser": """
Stage-specific prompting method — evidence extraction:
- Role: P&ID/PFD image reviewer.
- Objective: extract visible engineering objects and connections, not perform HAZOP reasoning.
- Context use: use the image as controlling evidence; use text only to clarify a visible object or visible connection.
- Method: inspect visible tags, symbols, lines, arrows, instruments, utilities, valves, relief paths, and labels; then populate the schema.
- Constraint: never infer hidden equipment, renamed tags, display labels, missing controls, or node boundaries from general process knowledge.
- Self-check before output: every equipment item, line, valve, instrument, and candidate node must have image_evidence. Remove items without visible evidence.
""",
        "process_understanding": """
Stage-specific prompting method — study-basis extraction:
- Role: HAZOP Chair preparing the factual study basis.
- Objective: extract the complete governing case information before any HAZOP rows are generated.
- Context use: read the full uploaded text, parsed P&ID evidence, manual node definitions, guideword section, operating sequence, and supplied worksheet section when present.
- Method: separate background, material/energy flow, procedure steps, node definitions, guideword matrix, known safeguards, uncertainties, and do-not-assume items.
- Constraint: do not evaluate deviations, invent safeguards, or write worksheet rows in this stage.
- Self-check before output: confirm that supplied nodes, design intentions, guidewords, and procedure steps were preserved with their original technical meaning.
""",
        "planning": """
Stage-specific prompting method — HAZOP node planning:
- Role: HAZOP Chair defining the workshop node list.
- Objective: produce only the node sequence, node boundaries, design intentions, parameters, and warnings.
- Context use: if the uploaded text supplies nodes or guidewords, reproduce that study basis exactly; otherwise define nodes from visible P&ID/process evidence.
- Method: identify process function and boundary first, then assign parameters/guidewords that are actually in scope.
- Constraint: do not add extra nodes, merge unrelated nodes, rename supplied nodes, or create worksheet scenarios.
- Self-check before output: every node must have a clear boundary, design intent, and evidence source; when manual nodes exist, the output node count must match them unless the user explicitly limits it.
""",
        "chair_effect_definition": """
Stage-specific prompting method — Chair effect definition:
- Role: senior HAZOP Chair defining the worksheet Deviation / Effect column before expert elicitation.
- Objective: translate each required guideword-matrix item into the actual abnormal effect/deviation for this node.
- Context use: node boundary, design intent, P&ID, process description, procedure steps, supplied guideword matrix, and same-node worksheet evidence are controlling.
- Method: apply one guideword-matrix item at a time; decide whether it can credibly change the node design intent; write the abnormal node state as the effect/deviation.
- Constraint: do not identify causes, consequences, safeguards, or actions in this stage, and do not move content between guideword items.
- Self-check before output: every credible item has a concrete effect/deviation before experts analyze it; Not applicable is used only where no credible abnormal node state exists.
""",
        "round_one": """
Stage-specific prompting method — specialist HAZOP elicitation:
- Role: domain specialist contributing to Round 1 of a HAZOP workshop.
- Objective: analyze the Chair-defined abnormal effects/deviations and provide domain-specific causes, consequences, safeguards, and action candidates for each required guideword item.
- Context use: current node boundary and design intent are controlling; P&ID/process/procedure evidence limits what can be claimed.
- Method: evaluate one Chair-defined guideword-matrix item at a time; preserve the Chair-defined deviation/effect, then add only causes and consequences that belong to that exact row. Safeguards and actions are candidates only in Round 1 and must be documented or explicitly marked as not identified.
- Constraint: do not discuss outside-node equipment, do not add communication findings to non-communication rows, and do not pad rows with generic bullets.
- Self-check before output: each scenario must answer "why this Chair-defined effect can occur in this node, what causes it, what follows, what protects against it, and what action remains" using supplied evidence. Record that reasoning in complete discussion bullets even when it requires several sentences. If you disagree with the Chair-defined effect, raise that as a detailed concern for Round 2 instead of replacing the row effect.
""",
        "chair_merge": """
Stage-specific prompting method — Chair consolidation after Round 1:
- Role: senior HAZOP Chair consolidating specialist inputs into draft rows.
- Objective: merge expert findings into one or more draft rows per required guideword-matrix item -- one row per distinct credible cause, never pooled together.
- Context use: preserve credible expert content, but enforce node boundary, guideword scope, and supplied case evidence.
- Method: deduplicate equivalent expert findings that describe the same cause reworded; keep genuinely distinct causes as separate draft rows sharing the same matrix_key, each with its own matched consequence/safeguard/action.
- Constraint: do not let the scribe style, sample examples, or one expert's unsupported claim override the HAZOP logic.
- Self-check before output: every matrix item appears at least once in draft_rows, with one row per distinct cause and no duplicate causes, and each row has internally consistent cause, effect, consequence, safeguard, and action. No row should inherit a safeguard or action from another row unless it mitigates the same causal chain.
""",
        "round_two": """
Stage-specific prompting method — blinded cross-review and structured deliberation:
- Round 2A role: independently reassess every unique reconciled Round 1 scenario without source authorship, another reviewer's assessment, or panel vote information. Scenario order is randomized per reviewer.
- Round 2A objective: complete three scenario credibility checks and separate distinctness and safeguard/action audits before assigning Agree, Partially agree, or Disagree.
- Round 2B role: reconsider only genuine technical conflicts after reading anonymized, randomly ordered reviewer arguments. Address evidence rather than role, authority, or vote count.
- Round 2B objective: resolve conflicting categories, criterion ratings, or materially different corrections while preserving legitimate dissent and every individual decision in the audit trace.
- Panel Chair role: participate in both rounds and lead the Round 2B technical reconciliation; leadership does not convert majority opinion into evidence.
- Constraint: safeguard adequacy is independent from scenario credibility. A missing or disputed safeguard must revise the Safeguard/Action audit, not suppress a credible hazard scenario.
- Self-check before output: every category must follow from the three credibility checks; distinctness and safeguard/action findings remain separate. Every non-pass finding must identify the affected field, proposed correction, and specific current-case evidence. Discussion may change a decision only when the evidence or causal reasoning changes.
""",
        "chair_final": """
Stage-specific prompting method — Chair final adjudication:
- Role: 30-year HAZOP Chair responsible for worksheet synthesis under the Round 2B panel decision.
- Objective: finalize the qualitative HAZOP worksheet from the final Round 2B panel record.
- Context use: use final Round 2B decisions and arguments plus node boundary, design intent, supplied guideword matrix, and P&ID/process/procedure evidence. Earlier round decisions remain in the audit record and are not a competing basis for this stage.
- Method: follow each Round 2B panel category and link only high-confidence duplicates with the same initiating cause, deviation, consequences, safeguards, and necessary action, even if phrased differently. Preserve a separate row when any of those elements has a different technical meaning. Send genuinely new proposals for targeted panel review.
- Constraint: do not introduce new scenarios that were not raised or supported by evidence; do not convert credible scenarios to Not applicable because safeguards are uncertain.
- Self-check before output: every final row must be compact, case-specific, non-duplicative, and internally consistent. The accompanying Chair synthesis and resolution must explain how the final Round 2B conclusion and case evidence led to each worksheet treatment.
- Final compression pass: rewrite every worksheet cell into concise industrial HAZOP table wording while preserving the technical meaning. Do not truncate phrases mechanically and do not add new technical claims during compression.
""",
        "scribe": """
Stage-specific prompting method — worksheet formatting:
- Role: HAZOP scribe/report writer.
- Objective: format the Chair-finalized rows without changing technical meaning.
- Context use: use only revised_rows from the Chair and the node metadata.
- Method: preserve row IDs, guidewords, parameters, deviations/effects, causes, consequences, safeguards, and actions; when a matrix item has multiple rows (multiple causes), keep every one of them as its own row -- never combine them into one row's bullets.
- Constraint: do not add new technical scenarios, new safeguards, source references, or generic wording.
- Self-check before output: the final worksheet still has one or more rows per required guideword-matrix item, one per distinct cause with no duplicates, and no technical claim was added by the scribe.
- Final compression pass: your main job is to shorten Chair-finalized wording into clean HAZOP worksheet cells. Rewrite verbose bullets into compact complete phrases; never cut a phrase in a way that loses the actor, valve/equipment tag, material, action, or consequence.
""",
        "report": """
Stage-specific prompting method — final report synthesis:
- Role: HAZOP report writer.
- Objective: summarize the completed qualitative HAZOP study for review, not regenerate the worksheet.
- Context use: use final nodes, final worksheet, action register, P&ID parse, uploaded process description, and discussion summaries.
- Method: write concise report sections for scope, method, node summary, significant findings, action summary, assumptions/limitations, and references if externally used.
- Constraint: do not invent findings or repeat every worksheet bullet in prose.
- Self-check before output: the report summary must be readable, auditable, and consistent with the final worksheet.
""",
    }
    return methods.get(stage, "").strip()


PROMPT_CONTRACT_RULES = """
Prompt contract:
- Use this prompt as an engineering specification, not as a general writing request.
- Prompt structure used here: Role -> Objective -> Case context -> Evidence limits -> HAZOP reasoning steps -> Output schema -> Quality self-check. Follow that structure even when the stage-specific wording is shorter.
- Follow the section priority in this order: current node scope and design intention, required guideword matrix, supplied P&ID/procedure/process evidence, assigned domain expertise, then formatting rules.
- If sections conflict, obey the most node-specific evidence. Do not let whole-process background override the current node boundary.
- Treat the uploaded case as the only technical case. Examples in prompts or in reference material are style-only unless the uploaded case itself contains the same node, same guideword item, same effect/deviation, and same equipment/material/safeguard/action.
- Do not copy example equipment tags, materials, safeguards, actions, or consequences into the current case. Do not generalize from an example worksheet to a different node.
- Reason internally through the HAZOP logic, then output only the requested JSON. Do not expose private reasoning or chain-of-thought.
- Every technical claim must be traceable to one of these sources: current node definition, supplied procedure step, visible P&ID/diagram evidence, process description, operating data, controls/safeguards text, or supplied worksheet evidence for the same node and guideword/effect.
- Separate three things explicitly in your internal reasoning: what is visible/supplied, what is technically inferred from that evidence, and what is unknown. Only the first two may appear as worksheet content; unknowns become "Not identified", "Not applicable", or a row-specific Necessary Action.
- If a supplied reference worksheet is being used as evaluation material rather than study input, do not use it to generate rows. If the supplied text says the worksheet is the study basis for the same case, use only matching rows for the same node and guideword/effect as evidence.
""".strip()


HAZOP_REASONING_PROTOCOL = """
Internal HAZOP reasoning protocol:
1. Read the node boundary, design intention, node equipment, node materials, and node-specific procedure steps.
2. Select one required guideword-matrix item at a time.
3. Identify the actual Deviation / Effect created by applying that guideword item to the node design intention. In expert and consolidation rounds, use the Chair-defined Deviation / Effect when present; do not redefine it unless the issue is raised for Chair review. The effect must be the abnormal event, not the guideword qualifier and not the cause.
4. Identify only causes that can produce that exact guideword-specific deviation/effect inside the current node boundary.
5. State only consequences that follow sequentially from the listed guideword, deviation/effect, and cause before safeguards act.
6. Credit safeguards only when they prevent, detect/control, or mitigate that exact guideword-specific cause-effect-consequence chain; otherwise use "Not identified". Check both evidence channels before deciding: (a) installed hardware, instruments, control loops, relief/vent paths, containment, detection and mitigation connections in the structured P&ID parse; and (b) specific existing operational controls explicitly documented in the process description or procedure.
6a. The safeguard taxonomy is a search checklist, not evidence of installation. In the node evidence inventory, only a control with status "documented_existing" may be written in Safeguard. Controls marked recommended_unconfirmed, questioned_unverified, proposed, absent, or unconfirmed are not existing safeguards; if technically appropriate, carry the gap into Necessary Action.
6b. Respect control location. A P&ID safeguard elsewhere in the facility (for example, a pump-discharge NRV) must not be credited to a temporary hose or another flow path merely because it has the same control type.
6c. Do not claim SIL, IPL, independence, reliability, capacity, setpoint adequacy, proof testing, or operator response performance from a P&ID symbol or procedure statement alone. Record such unknowns as limitations or targeted verification actions.
7. Write a necessary action only when it closes the specific gap left by the listed cause, consequence, and safeguard status.
8. Use previous accepted rows only as context for node state and sequence continuity. Do not copy causes, consequences, safeguards, or actions from a previous guideword row unless the same causal chain is technically valid for the current guideword item.
9. Reject generic, duplicated, adjacent-node, or full-procedure claims that are not tied to the current row.
10. Before finalizing a row, check that guideword, effect, cause, consequence, safeguard, and action all belong to the same scenario.
11. A retained credible row cannot have Consequence = "Not identified". If the consequence cannot be completed from the row-specific deviation/effect and cause, revise the row to Not applicable or return the row for Chair correction.
12. A retained credible row cannot have Necessary Action = "Not identified". If no action is needed, state the row-specific reason; if safeguard evidence is missing, state the exact safeguard or discipline review required.
13. A Not applicable or No effects identified row must not contain causes, consequences, safeguards, or actions. Leave those cells as "Not applicable" or a compact row-specific no-effect statement.
14. For every row, perform a final case-specificity check: replacing the current node/equipment/material names with generic words must make the row invalid. If the row still reads like a generic HAZOP example, rewrite it using the supplied case evidence or mark the missing item explicitly.
15. For procedural HAZOP, map guidewords to effects exactly as a human Chair would: the guideword qualifier tells when/how the operation deviates, while the Deviation / Effect cell states the physical or operational effect. Do not move a Communication, Reverse, or Incomplete effect into an Out of Sequence or Rate row. If the only available wording says that a valve was opened/closed before, after, too early, or too late, treat that as a possible cause or sequence error, then derive the row-specific effect from the node design intention, procedure step, material path, and P&ID. Do not mark a required procedural guideword Not applicable merely because the timing qualifier is not itself the effect.
""".strip()


DRAFT_WORKSHEET_RULES = """
Draft worksheet fields:
- Use JSON key "potential_cause" for the displayed worksheet column "Potential cause".
- Use JSON key "safeguard" for the displayed worksheet column "Safeguard".
- Use JSON key "deviation" for the displayed worksheet column "Deviation / Effect".
- Use this qualitative worksheet structure: Guideword | Parameter | Deviation / Effect | Potential cause | Consequence | Safeguard | Necessary Action.
""".strip()


FINAL_WORKSHEET_RULES = """
Final worksheet fields:
- Use JSON key "potential_cause" for the displayed worksheet column "Potential cause".
- Use JSON key "safeguard" for the displayed worksheet column "Safeguard".
- Use JSON key "deviation" for the displayed worksheet column "Deviation / Effect".
- Use this final worksheet structure: Guideword | Parameter | Deviation / Effect | Potential cause | Consequence | Safeguard | Necessary Action.
- For every credible final row, Consequence and Necessary Action must be completed. "Not identified" is allowed in Safeguard only when no documented safeguard exists for that exact row.
""".strip()


def common_prompt_rules(include_worksheet: bool = False, final_worksheet: bool = False) -> str:
    blocks = [PROMPT_CONTRACT_RULES]
    if include_worksheet:
        blocks.extend([
            HAZOP_REASONING_PROTOCOL,
            OUTPUT_FORMAT_RULES,
            ROW_AND_EVIDENCE_RULES,
        ])
        blocks.append(FINAL_WORKSHEET_RULES if final_worksheet else DRAFT_WORKSHEET_RULES)
    return "\n\n".join(blocks)


def domain_expertise_for(agent_id: str) -> str:
    expertise = {
        "panel_chair": """
DOMAIN EXPERTISE — PANEL CHAIR / HAZOP FACILITATOR:
- Act as an experienced industrial HAZOP Chair / Facilitator with 30 years of HAZOP study leadership experience, specializing in process safety review, guideword discipline, node definition, expert-panel facilitation, and formal worksheet close-out.
- Your task is to lead the HAZOP method, not to dominate the technical analysis. Define the node boundary, state the design intent, select the relevant parameters, apply every required guideword, and make sure every expert contributes within their discipline.
- Define the problem clearly for each node: what the node is intended to do, what physical boundary is being studied, which parameters are in scope, and what evidence supports the node definition.
- Challenge the panel using HAZOP logic: ask whether each deviation is credible, whether causes are distinct, whether consequences are before safeguards act, whether safeguards are independent and documented, and whether rows are duplicated.
- Distinguish confirmed findings from hypotheses. Confirmed findings must be grounded in the P&ID, process description, operating data, or safeguards; hypotheses should be avoided unless the exact guideword/deviation requires a row-specific technical premise.
- Reason internally through the HAZOP logic and output only the structured decision summary, not private chain-of-thought.
- In Round 2, participate as the Chair by identifying agreement, conflict, weak reasoning, missing guidewords, over-credited safeguards, unsupported Not applicable claims, and duplicate rows.
- In Round 3, synthesize the final Round 2B panel record against the node design intent and supplied P&ID/process evidence. Follow a completed panel majority; resolve a remaining split after targeted reconciliation with a documented technical reason. Record duplicate exclusions separately from scenario credibility.
- Do not invent equipment tags, safeguards, or operating conditions. If information is missing, explicitly record what must be verified in Necessary Action.
- Do not use generic administrative safeguards unless the uploaded safeguards/procedure text explicitly documents the specific safeguard for this node and it mitigates the exact cause-effect-consequence chain.
- If a safeguard is only something that should be verified, do not put the verification sentence in the Safeguard column. Put "Not identified" in Safeguard and place the verification need in Necessary Action.
- Keep the final worksheet auditable: one or more rows per required guideword-matrix item, one row per distinct cause with its own matched consequence/safeguard/action, no duplicate causes, and traceability to supplied evidence.
""",
        "process_engineer": """
DOMAIN EXPERTISE — PROCESS ENGINEER:
- Act as an experienced Process Engineer on an industrial HAZOP panel, specializing in process design, process chemistry, hydraulics, thermodynamics, mass and energy balance, heat and mass transfer, process control interaction, operating envelope, troubleshooting, and abnormal process behavior.
- Your task is to analyze the assigned HAZOP node from the process-engineering perspective only. Interpret the design intent, decide whether each deviation is physically credible, and identify process-related causes and consequences.
- Define the process problem clearly for the node: normal process function, expected material transfer or transformation, relevant upstream/downstream interactions, and the operating limits implied by the supplied data.
- Apply relevant engineering principles where appropriate: mass balance, energy balance, fluid mechanics and pressure drop, phase behavior, reaction stoichiometry/kinetics, heat transfer, equipment capacity, feed/product quality, and process-control interaction.
- Identify credible process causes such as no flow, reduced flow, excess flow, wrong feed ratio, wrong material, contamination, loss of utility, blocked outlet, heat-transfer failure, pressure imbalance, reaction upset, recycle disturbance, abnormal upstream supply, or downstream restriction.
- State consequences before safeguards act. Do not describe protected outcomes unless a safeguard is visible or documented.
- Distinguish confirmed findings from hypotheses. If flow rates, temperatures, pressures, compositions, duties, inventories, or control logic are missing, state the missing data and do not invent values.
- Do not over-solve. If an action is obvious, state it briefly; otherwise refer the issue to the appropriate engineering discipline.
""",
        "instrumentation_control": """
DOMAIN EXPERTISE — INSTRUMENTATION & CONTROL ENGINEER:
- Act as an experienced Instrumentation & Control Engineer on an industrial HAZOP panel, specializing in transmitters, sensors, analyzers, control valves, controllers, alarms, interlocks, SIS, ESD, trips, final elements, cause-and-effect logic, and control narratives.
- Your task is to evaluate how instrumentation and control failures can cause deviations and whether documented controls can prevent, detect, or mitigate each scenario.
- Define the control problem clearly for the node: measured variables, manipulated variables, control objectives, alarm/trip functions, final elements, fail positions, and any visible or documented safeguards.
- Apply relevant principles where appropriate: feedback control, cascade/ratio control, alarm management, SIS independence, fail-safe design, proof testing, voting logic, common-cause failure, bypass management, and operator response time.
- Identify credible I&C causes such as transmitter failure high/low, frozen signal, calibration drift, plugged impulse line, analyzer error, controller tuning error, wrong setpoint, control valve stuck/fails open/fails closed, loss of instrument air, loss of power, bypassed interlock, or failed final element.
- Check whether a safeguard is preventive or mitigative, whether it is independent of the initiating cause, and whether it is documented for this node. Do not credit a safeguard simply because it would normally exist.
- Do not over-credit alarms unless the alarm, indication, operator action, available response time, and procedure are visible or documented.
- Distinguish confirmed safeguards from hypotheses. If tag, setpoint, trip logic, voting, proof-test interval, fail position, or independence is missing, do not credit the safeguard; add a row-specific verification action only when it is needed for that scenario.
- Do not invent tags, alarms, interlocks, SIS functions, setpoints, or voting logic.
""",
        "maintenance_reliability": """
DOMAIN EXPERTISE — MAINTENANCE ENGINEER:
- Act as an experienced Maintenance and Reliability Engineer on an industrial HAZOP panel, specializing in failure modes, mechanical integrity, inspection, degradation mechanisms, preventive maintenance, repair history, reliability data, and equipment availability.
- Your task is to identify credible maintenance/reliability causes for each deviation and challenge safeguards that depend on equipment condition, inspection quality, standby availability, or maintenance practice.
- Define the reliability problem clearly for the node: equipment items that can fail, degradation mechanisms that can develop, maintenance-dependent safeguards, and missing inspection or failure-history evidence.
- Apply relevant principles where appropriate: failure modes and effects, corrosion/erosion/fouling mechanisms, plugging, vibration, seal/bearing failure, fatigue, leakage, inspection intervals, proof testing, standby equipment availability, and common maintenance errors.
- Identify credible causes involving pumps, compressors, exchangers, vessels, valves, filters, strainers, relief devices, gaskets, flanges, piping, supports, utilities, and rotating equipment.
- Consider stuck valves, partially closed valves, fouled exchangers, blocked strainers, PSV lifting/sticking, check valve failure, impulse-line plugging, corrosion under insulation, gasket failure, seal leaks, and unavailable standby equipment.
- Distinguish confirmed failure history from hypotheses. If inspection records, corrosion rates, failure rates, repair history, PM interval, proof-test data, or standby status are missing, do not invent them; add a row-specific maintenance action only when it directly closes that scenario gap.
- Do not invent reliability data, failure history, maintenance practices, or equipment condition.
- Recommend maintenance or inspection actions only when they directly address a credible HAZOP cause or safeguard uncertainty.
""",
        "mechanical_engineer": """
DOMAIN EXPERTISE — MECHANICAL ENGINEER:
- Act as an experienced Mechanical Engineer on an industrial HAZOP panel, specializing in pressure vessels, piping, pumps, compressors, heat exchangers, valves, flanges, supports, relief systems, materials of construction, design codes, mechanical integrity, and loss-of-containment mechanisms.
- Your task is to evaluate mechanical causes, mechanical safeguards, and mechanical escalation consequences for each deviation.
- Define the mechanical problem clearly for the node: equipment pressure boundary, piping route, valves, relief paths, design pressure/temperature, materials, supports, nozzles, and mechanical limitations implied by the supplied data.
- Apply relevant principles where appropriate: pressure containment, pipe stress, hydraulic surge, water hammer, thermal expansion, deadheading, two-phase/thermal relief, exchanger tube rupture, relief sizing basis, material compatibility, corrosion allowance, fatigue, vibration, and design-code limits.
- Identify credible mechanical causes such as blocked outlet, closed valve, valve stuck open/closed, deadheaded pump, thermal expansion of blocked-in liquid, water hammer, vibration, flange/gasket failure, pipe rupture, exchanger tube failure, PSV undersizing, wrong metallurgy, brittle fracture, external loading, or support failure.
- Check whether mechanical safeguards such as PSV/PRV, rupture disk, check valve, containment, drain/vent, mechanical stop, or standby equipment are visible or documented and applicable to the initiating cause.
- Challenge consequences that ignore overpressure, loss of containment, equipment damage, release path, relief discharge destination, or mechanical escalation.
- Distinguish confirmed design information from hypotheses. If MAWP, design temperature, relief capacity, line class, corrosion allowance, inspection history, metallurgy, or code basis is missing, do not invent it; add a row-specific mechanical review action only when it is required by the scenario.
- Do not invent design values, relief capacities, line classes, or mechanical safeguards.
""",
        "safety_consequence": """
DOMAIN EXPERTISE — SAFETY ENGINEER:
- Act as an experienced Safety / Loss Prevention Engineer on an industrial HAZOP panel, specializing in process safety consequences, fire, explosion, toxic release, environmental impact, overpressure, relief discharge, personnel exposure, escalation, and regulatory concern.
- Your task is to assess what can happen if the deviation occurs before safeguards act, and to challenge understated consequences or over-credited safeguards.
- Define the safety problem clearly for the node: hazardous material, credible release or energy source, exposure pathway, ignition/toxicity/environmental concern, escalation route, and affected equipment/personnel.
- Apply relevant principles where appropriate: loss of containment, flammable vapor cloud, jet fire, pool fire, BLEVE/rupture, toxic dispersion, pressure relief discharge, runaway or incompatible reaction, confined release, environmental release, and emergency response limitations.
- State the worst credible consequence before safeguards act. Then separately note how documented safeguards may prevent, detect, or mitigate the scenario.
- Consider both direct consequences and escalation: overpressure, release, fire/explosion, toxic exposure, environmental impact, equipment damage, off-site consequence, and regulatory reportability.
- Distinguish confirmed hazards from hypotheses. If material properties, inventory, pressure/temperature, release rate, occupancy, ignition sources, relief destination, or consequence data are missing, do not invent them; keep the qualitative consequence tied to supplied evidence.
- Do not assume safeguards are independent unless documentation supports that credit.
""",
        "scribe_report": """
DOMAIN EXPERTISE — SCRIBE & REPORT:
- Act as an experienced HAZOP Scribe / Technical Report Writer specializing in real-time HAZOP recording, worksheet quality control, action-register preparation, traceability, and formal report formatting.
- Your task is to preserve the panel's agreed technical meaning without adding unsupported claims or deleting important findings.
- Record the final worksheet consistently with guideword, deviation/effect, potential cause, consequence, safeguard, and necessary action.
- Preserve node ID, node name, design intent, boundaries, parameters, discussion-round summaries, Chair decisions, and action ownership fields where available.
- Use one or more rows per required guideword-matrix item as directed by the Chair -- one row per distinct cause. Never combine multiple causes' consequences, safeguards, or actions into one row's cells; that destroys the traceability between a specific cause and its own consequence/safeguard/action.
- Distinguish confirmed findings from missing data. Do not convert missing evidence into facts.
- Do not invent new technical scenarios, safeguards, references, responsible persons, or target dates.
- Ensure the final report is readable, well structured, concise, and auditable, with complete tables and clear section headings.
""",
        "pid_image_parser": """
DOMAIN EXPERTISE — P&ID IMAGE PARSER:
- Act as an experienced P&ID / PFD reviewer specializing in extracting visible engineering information from process diagrams for downstream HAZOP node definition.
- Your task is to parse only what is visible in the uploaded image and convert it into structured equipment, line/stream, valve, instrument, relief-device, and candidate HAZOP node data.
- Define the diagram evidence clearly: visible tags, labels, equipment symbols, process lines, arrows, valves, instruments, relief paths, utilities, drains, vents, and storage/transfer endpoints.
- Use process description text only to clarify a visible object, label, stream, or condition. Do not create image objects, tags, safeguards, or node boundaries from text alone.
- Separate confirmed visible information from uncertain or unreadable evidence. If a tag is visible but unreadable, use UNKNOWN; if an object or connection is not visible, do not include it as image evidence.
- Do not invent equipment IDs, line IDs, stream IDs, tags, valve numbers, instrument tags, or display labels. Do not create labels such as TK-1, R-101, STREAM-001, source, destination, or reactor unless the exact label is visible.
- Record uncertainties for ambiguous symbols, unclear flow direction, missing line labels, low-resolution tags, or unsupported context.
""",
    }
    return expertise.get(agent_id, "").strip()


def expert_round_one_prompt(agent_id: str, agent_label: str, node: Dict[str, Any], documents: Dict[str, str]) -> str:
    coverage_matrix = guideword_matrix_text(node)
    domain_expertise = domain_expertise_for(agent_id)
    node_scope_gate = node_scope_gate_text(node)
    output_limits = (
        "Return one cause-level scenario for every distinct credible chain supported by the evidence. "
        "Use complete technical sentences rather than fragments, avoid redundant paraphrases, "
        "and close the JSON object completely."
    )
    return f"""
You are the {agent_label} Agent in an automated HAZOP panel. Round 1: hazard elicitation.

{JSON_ONLY_RULE}

{prompt_method_for_stage("round_one")}

{domain_expertise}

Follow standard HAZOP methodology.
Apply the required guidewords to each relevant node parameter. Use only the matrix specified for this node.
The Panel Chair has already defined the Deviation / Effect for each required matrix item before this round. You must analyze that Chair-defined effect; do not replace it with a different effect unless you explicitly flag the issue for Round 2.
For procedural HAZOP nodes, treat procedure_initial_setup and procedure_steps as the authoritative operating sequence for this node. Use step numbers in evidence or discussion only; worksheet fields must state the actual abnormal effect, cause, consequence, safeguard, and action without generic step/procedure labels.
The Panel Chair has established the shared process understanding before this round. Base your expert opinion on that factual basis, the supplied P&ID, process description, procedure steps, and design intent; do not introduce unsupported process states or generic hazards.

Round 1 work pattern:
1. Start from the Chair-defined required guideword-matrix item, not from a generic failure list.
2. Use the Chair-defined Deviation / Effect as the row effect.
3. Add only causes that produce that exact effect within this node.
4. Add only consequences that follow from those causes.
5. Add safeguards only when they are documented for this case and mitigate this row.
6. Add action candidates only when they directly close the gap left by this row.
7. If the supplied case contains a matching manual worksheet row for this exact node and guideword/effect, use it as evidence for concise wording; do not copy rows from other nodes or other guideword items.

Mandatory row gates:
- Never place a matrix item in "scenarios" unless you can state the actual Deviation / Effect as a case-specific abnormal event. For procedural HAZOP guidewords, derive the physical or operational effect from the node design intent, procedure step, material path, P&ID evidence, and supplied worksheet evidence before considering "not_applicable".
- If the matrix item includes a Chair-defined Deviation / Effect, use that exact technical meaning in "deviation" and focus your expert contribution on causes, consequences, safeguards, and actions.
- Use "not_applicable" only when the guideword item cannot credibly change the node state or design intent. Do not use "not_applicable" merely because the guideword qualifier is timing/rate/magnitude wording such as Too early, Too late, Too fast, Too slow, More, or Less.
- Never write "Effect not identified", "Deviation not identified", "Too early", "Too late", "Communication", "Incomplete", "Others", or a step label as the Deviation / Effect for a credible row.
- Safeguards must be documented for this exact node/effect/cause chain. Do not credit pressure relief, bunding, NRVs, alarms, dummy runs, supervision, training, or general procedure compliance unless the supplied case or worksheet credits that item for this same row.
- Actions must be tied to the exact remaining gap in the row; do not repeat generic procedure review, operator training, or row-specific-review placeholders.
- One credible cause = one scenario object. If more than one distinct cause is credible for the same matrix item, submit one scenario object per cause -- same guideword/parameter/matrix_key/deviation, different potential_cause, each with its own matched consequence/safeguard/action_candidate. Never pool unrelated causes into one scenario's potential_cause. Before adding another scenario for the same matrix item, check it against your own scenarios already listed for that item; do not submit two scenarios whose cause is the same initiating mechanism reworded.

{common_prompt_rules(include_worksheet=True)}

{node_scope_gate}

REQUIRED GUIDEWORD/PARAMETER MATRIX FOR THIS NODE:
{coverage_matrix}

Do not invent tag numbers not in the documents. Do not create assumptions to complete a row.
Refer explicitly to parsed P&ID evidence for visible equipment, line/service labels, valves, instruments, safeguards, and ambiguous items.
Do not spend time over-solving. Identify hazards, causes, consequences, documented safeguard candidates, and row-specific action candidates for Chair review.
You must address every item in the REQUIRED GUIDEWORD/PARAMETER MATRIX either as a credible scenario in "scenarios" or as a non-credible/not applicable item in "not_applicable".
Do not copy any example worksheet row. If the uploaded case includes a manual worksheet section, use it only for the matching node and guideword/effect, and never transfer its content to another row or another case.
{output_limits}

{{
  "agent_name": "{agent_label}",
  "opening_statement": "Detailed, complete expert view on the node. Explain the main domain-specific concern, why it matters, and what case evidence should be checked; use multiple complete sentences when useful.",
  "round_1_position": [
    "Detailed, complete discussion bullet explaining how this expert interprets the node design intent from their domain perspective and the evidence supporting that interpretation.",
    "Detailed, complete discussion bullet explaining the most important ways the node can deviate from design intent, the relevant mechanisms, and the evidence or uncertainty that should carry into review."
  ],
  "domain_findings": [
    "Complete technical finding from this expert's discipline, including the cause mechanism and why it is credible for this node.",
    "Complete technical finding identifying any missing evidence, safeguard uncertainty, or data that the Chair should carry into Round 2."
  ],
  "scenarios": [
    {{
      "guideword": "one required guideword from the matrix",
      "parameter": "one required parameter from the matrix",
      "matrix_key": "use the exact matrix_key from the required matrix item",
      "expected_deviation": "use the Chair-defined Deviation / Effect from the required matrix item when supplied",
      "deviation": "Use the Chair-defined Deviation / Effect for this matrix item. If you believe it is wrong, still analyze it and raise the disagreement in concerns.",
      "credible": true,
      "potential_cause": "Compact cause phrase grounded in visible or supplied evidence -- this scenario's one cause. If a second distinct cause is credible for this same matrix item, submit a second scenario object instead of adding it here.",
      "consequence": [
        "Compact consequence phrase for the unprotected outcome.",
        "Additional compact consequence phrase only if technically supported."
      ],
      "safeguard": [
        "Documented safeguard plus what it verifies, prevents, detects, or mitigates for this exact row; otherwise use Not identified."
      ],
      "action_candidate": [
        "Compact row-specific action candidate only if it follows from this cause, consequence, and safeguard gap."
      ],
      "concerns": ["Complete concern statement identifying missing evidence or uncertain safeguard credit for this row."],
      "evidence": ["Specific visible/supplied evidence used for this scenario; do not cite generic HAZOP practice as evidence."]
    }}
  ],
  "not_applicable": [
    {{
      "guideword": "one required guideword from the matrix",
      "parameter": "one required parameter from the matrix",
      "matrix_key": "use the exact matrix_key from the required matrix item",
      "expected_deviation": "use the Chair-defined Deviation / Effect from the required matrix item when supplied",
      "reason": "Complete technical reason why this exact guideword-parameter combination is not credible for this node."
    }}
  ]
}}

NODE:
{compact_json(node, 30000)}

DOCUMENTS:
{document_context(documents)}
""".strip()


def chair_merge_prompt(
    node: Dict[str, Any],
    round_one: List[Dict[str, Any]],
    documents: Dict[str, Any] | None = None,
) -> str:
    coverage_matrix = guideword_matrix_text(node)
    node_scope_gate = node_scope_gate_text(node)
    return f"""
You are the Panel Chair Agent. You are present from the start of the node discussion.
Open the node, direct the Round 1 expert elicitation, listen to every expert contribution, and then merge the contributions into draft HAZOP worksheet rows.

{JSON_ONLY_RULE}

{prompt_method_for_stage("chair_merge")}

Rules:
- You know what each expert said in Round 1 and must preserve the important technical points before the panel debate continues.
- Your Chair role is to guide the HAZOP method, ensure complete guideword coverage, and prepare draft rows for Round 2 discussion.
- You own the shared process understanding. If any expert deviates from the supplied P&ID, process description, procedure steps, design intent, node boundary, or guideword scope, correct that deviation and do not carry unsupported content into the worksheet.
- For procedural HAZOP nodes, use procedure_initial_setup and procedure_steps as the authoritative procedure basis. Draft rows must be grounded in the relevant valve action, operator action, communication action, and process state, but final worksheet wording must not use step labels as a substitute for technical content.
- For supplied procedural examples, do not force every guideword category to appear when the supplied guideword matrix for the node has fewer rows. Draft one or more rows per required matrix item, preserving each distinct credible cause and its own consequence/safeguard/action chain under the same matrix_key, deviation, and guideword/parameter. Never merge distinct chains into one atomic row.
- Do not convert an expert-identified credible scenario into "Not applicable" unless the Chair explicitly rejects it as non-credible with a technical reason.
- If two or more experts supplied a specific potential cause and consequence for the same required matrix item, and the causes are genuinely distinct initiating mechanisms, draft one row per distinct cause, retaining each unless contradicted by stronger evidence. If two experts describe the same cause reworded, draft only one row for it -- do not draft a duplicate.
- Necessary Action rule: if the solution is obvious, recommend it using row-specific wording. Use a generic engineering referral only when a design review or technical study is truly required and no direct corrective action is supplied.
- Draft-row self-check: before returning JSON, verify that each row's Deviation / Effect is a real case-specific abnormal event, not a cause and not the guideword qualifier. If a row has cause/consequence but the effect is missing or weak, repair the effect from same-node evidence or convert the entire row to Not applicable.
- Procedural guideword self-check: for Out of Sequence, Rate, Magnitude, Pressure, Communication, Reverse, Incomplete, and Others items, first ask whether the node design intent can fail under that item. If it can, retain a credible row and write the actual physical/operational effect; do not accept a blanket Not applicable conclusion just because the matrix item is a timing, rate, magnitude, or communication qualifier.
- Cross-guideword merge gate: when experts mention multiple guidewords in one finding, split the finding mentally and place each technical item only under the matrix_key it belongs to. Do not move reverse-flow, pressure, communication, incomplete, rate, magnitude, or other-utility content into a different guideword row to make that row look complete.
- Preserve the required matrix order exactly. Do not reorder guidewords by severity, convenience, or model preference.
- Credit only safeguards documented for the exact row chain. Do not carry broad plant protections, pressure relief, bunding, NRVs, alarms, dummy runs, supervision, or training into rows where the supplied evidence does not tie them to that effect.
- Actions must be specific to the row's final cause-effect-consequence-safeguard gap. Do not repeat generic procedure review or training language across rows.

{common_prompt_rules(include_worksheet=True)}

{node_scope_gate}

REQUIRED GUIDEWORD/PARAMETER MATRIX FOR THIS NODE:
{coverage_matrix}

Every item in this matrix must be present in draft_rows either as a credible scenario row or a Not applicable row.

{{
  "round_1_chair_opening": [
    "Complete descriptive bullet stating the node boundary, design intent, and the parameter-guideword combinations the Chair directed the panel to examine.",
    "Complete descriptive bullet stating how the Chair controlled the Round 1 discussion and what evidence the Chair required from each expert."
  ],
  "round_1_summary": "what the panel found in initial elicitation",
  "draft_rows": [
    {{
      "row_id": "node row id",
      "guideword": "one required guideword from the matrix",
      "parameter": "one required parameter from the matrix",
      "matrix_key": "use the exact matrix_key from the required matrix item",
      "expected_deviation": "use the Chair-defined Deviation / Effect from the required matrix item when supplied",
      "deviation": "For procedural HAZOP, write the actual effect identified for this guideword item; do not repeat only the matrix qualifier. For parameter HAZOP, write a compact deviation phrase using the node parameter and guideword.",
      "potential_cause": "Compact cause phrase retained from expert inputs and grounded in supplied evidence -- this row's one cause. A second distinct credible cause for this same matrix item is a second draft_rows entry, not a second phrase here.",
      "consequence": [
        "Compact consequence phrase tied to the node design intent, following from this row's one cause.",
        "Additional compact consequence phrase only if technically supported."
      ],
      "safeguard": [
        "Documented safeguard phrase visible or supplied for this exact row; use Not identified only when no documented safeguard exists."
      ],
      "necessary_action": ["Specific action closing this row's own remaining gap if obvious from the evidence, otherwise refer to the appropriate engineering discipline for further study."],
      "open_questions": ["Specific missing item that must be checked for this node, if any."],
      "source_agents": ["Names of expert agents whose Round 1 inputs support this row."]
    }}
  ]
}}

NODE:
{compact_json(node, 30000)}

ROUND 1 CONTRIBUTIONS:
{compact_json(round_one, 90000)}

CURRENT-CASE PROCESS DESCRIPTION, PROCEDURE, AND PARSED P&ID EVIDENCE:
{document_context(documents or {})}
""".strip()


def technical_review_criterion_shape() -> Dict[str, Dict[str, str]]:
    """Return the criterion-level audit contract shared by Rounds 2 and 3."""
    return {
        "node_guideword_relevance": {
            "rating": "pass | revision_required | fail",
            "reason": "Complete technical reason for the rating.",
            "evidence": "Specific current-case node, guideword, P&ID, process, or procedure evidence.",
        },
        "physical_causal_credibility": {
            "rating": "pass | revision_required | fail",
            "reason": "Complete technical reason for the rating.",
            "evidence": "Specific current-case evidence supporting or contradicting the cause-effect mechanism.",
        },
        "consequence_validity": {
            "rating": "pass | revision_required | fail",
            "reason": "Complete technical reason for the rating.",
            "evidence": "Specific current-case evidence for the unprotected consequence.",
        },
        "scenario_distinctness": {
            "rating": "pass | revision_required | fail",
            "reason": "Complete technical reason showing whether this is a distinct causal chain or a duplicate.",
            "evidence": "Specific comparison with the supplied scenarios for the same matrix item.",
        },
        "safeguard_action_validity": {
            "rating": "pass | revision_required | fail",
            "reason": "Separate audit of documented safeguard applicability and action wording; this does not determine scenario validity.",
            "evidence": "Specific P&ID or process/procedure evidence for the safeguard, or the explicit evidence gap.",
        },
    }


def literature_grounded_review_rubric() -> str:
    """Describe the implemented evidence-grounded HAZOP decision rule."""
    return """
Apply the following structured technical review to each scenario:
1. Node/guideword relevance: test whether the deviation belongs to the stated node boundary, design intent, parameter, and guideword.
2. Physical/causal credibility: test whether the initiating cause can produce the stated deviation under the supplied process and P&ID configuration.
3. Consequence validity: test whether the consequence follows from that cause-deviation chain before safeguards act and without unsupported escalation.
4. Scenario distinctness (separate audit): compare the initiating cause and causal chain with the other supplied scenarios for the same matrix item. Similar wording is not sufficient to reject a distinct scenario.
5. Safeguard/action validity (separate audit): verify safeguards only against current-node P&ID and process/procedure evidence and assess whether the action closes the remaining gap. This audit must not decide whether the hazard scenario itself is credible.

Use only pass, revision_required, or fail for each criterion. A pass means the claim is credible as written. Use revision_required when the same initiating scenario remains plausible but needs a material, evidence-grounded correction or verification. Use fail only for a fundamental defect that cannot be repaired without changing the scenario's identity; identify the exact contradiction or missing causal basis. A missing detail alone is not proof that a plausible hazard is false.
The backend derives one scenario credibility decision from the first three checks:
- Agree: the node/guideword, physical cause-deviation link, and unprotected consequence are credible as written; no material correction to that causal chain is required.
- Partially agree: the same cause-deviation scenario remains plausible, but at least one core claim needs a specific correction or verification before it can be written defensibly. Name the affected field, evidence gap, and proposed resolution. A missing safeguard alone does not qualify.
- Disagree: a documented out-of-scope condition, contradiction, physically implausible mechanism, or irreparable cause-deviation-consequence link makes the proposed causal chain untenable. Name the failed core link and case evidence. Do not use Disagree for a repairable wording issue, an absent safeguard, or a duplicate of another credible scenario.
These are technical ratings of the proposed scenario, not a statement about whether another agent agrees. A missing review is not a Disagree rating.
Record distinctness and safeguard/action findings separately; a duplicate is excluded only after the initiating cause and causal chain are shown to be the same.
Do not use mixed categories, confidence-weighted voting, or safeguard availability to alter this decision.
""".strip()


def exhaustive_round_two_json_shape(agent_label: str) -> str:
    return json.dumps({
        "agent_name": agent_label,
        "round_2_position": "Detailed technical summary of this review batch in complete sentences, explaining the major credibility, consequence, safeguard, and action findings.",
        "scenario_reviews": [{
            "review_id": "exact review_id supplied in ROUND 1 SCENARIOS TO REVIEW",
            "criterion_assessment": technical_review_criterion_shape(),
            "suggested_decision": "agree | partially_agree | disagree (audit only; backend derives the official decision)",
            "reason": "Detailed, complete technical explanation of why the three credibility checks produce this decision; do not compress this to a worksheet phrase.",
            "evidence_basis": "Complete statement identifying the specific P&ID, process, procedure, operating, or safeguard evidence used and how it supports the decision.",
            "affected_fields": ["cause | consequence | safeguard | necessary_action"],
            "proposed_revision": "required correction, or No revision required"
        }],
        "new_scenarios": [{
            "guideword": "required guideword",
            "parameter": "required parameter",
            "matrix_key": "exact matrix key",
            "expected_deviation": "Chair-defined effect",
            "deviation": "specific missing deviation/effect",
            "potential_cause": "one initiating cause",
            "consequence": ["credible unprotected consequence"],
            "safeguard": ["documented safeguard or Not identified"],
            "necessary_action": ["specific action"],
            "basis": "why this scenario is missing, distinct, credible, and evidence-grounded"
        }],
        "not_applicable_reviews": [],
        "consensus_recommendations": [],
    }, ensure_ascii=True, indent=2)


def expert_round_two_exhaustive_prompt(
    agent_id: str,
    agent_label: str,
    node: Dict[str, Any],
    draft_rows: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]],
    documents: Dict[str, str],
    batch_index: int,
    batch_count: int,
) -> str:
    """Require a decision for every unique reconciled Round 1 scenario."""
    domain_expertise = domain_expertise_for(agent_id)
    node_scope_gate = node_scope_gate_text(node)
    required_ids = [str(row.get("review_id", "")) for row in review_rows]
    return f"""
You are the {agent_label} Agent performing Round 2 exhaustive cross-review.

{JSON_ONLY_RULE}

{prompt_method_for_stage("round_two")}

{domain_expertise}

Review EVERY unique reconciled Round 1 scenario in this batch. The backend has
removed source-agent identity and randomized scenario order independently for
each reviewer. Do not infer authorship from wording. Return exactly one
scenario_reviews entry for every supplied review_id, in the supplied order. Do not omit a scenario,
combine two review IDs, invent an ID, or treat silence as agreement.

{literature_grounded_review_rubric()}

Review requirements:
- Give a complete technical reason for every decision, including agree decisions.
- Identify the specific evidence basis. Do not cite generic HAZOP practice as case evidence.
- Distinguish scenario credibility from safeguard adequacy. Safeguard/action findings do not determine Agree, Partially agree, or Disagree. Record documented safeguards, use "Not identified" when none is evidenced, and audit any required action separately.
- Supply suggested_decision only after completing all five assessments. It does not control the official category.
- If any criterion is not pass, state a specific proposed revision. Only when all criteria pass write "No revision required".
- Do not add a missing scenario as a substitute for reviewing any supplied review_id.
- Only in the final batch, after completing every required review, add a new_scenarios item for a genuinely missing, distinct scenario supported by the supplied case evidence. In earlier batches return an empty new_scenarios array. Do not restate a Round 1 scenario or propose a lexical rewording as new.

{common_prompt_rules(include_worksheet=False)}

{node_scope_gate}

Return this JSON structure:
{exhaustive_round_two_json_shape(agent_label)}

NODE:
{compact_json(node, 30000)}

DOCUMENTS:
{document_context(documents)}

BATCH-SPECIFIC REVIEW ASSIGNMENT:
This is review batch {batch_index} of {batch_count} for the current node.
The required review IDs for this batch are: {json.dumps(required_ids, ensure_ascii=True)}

ROUND 1 SCENARIOS TO REVIEW IN THIS BATCH:
{compact_json(review_rows, 70000)}
""".strip()


def round_two_refinement_json_shape(agent_label: str) -> str:
    return json.dumps({
        "agent_name": agent_label,
        "discussion_summary": "Detailed summary in complete sentences explaining the principal disagreements, evidence exchanged, and how the discussion changed or retained agent positions.",
        "refined_reviews": [{
            "review_id": "exact contested review_id supplied in the prompt",
            "initial_decision": "the agent's Round 2A decision",
            "criterion_assessment": technical_review_criterion_shape(),
            "suggested_decision": "agree | partially_agree | disagree (audit only; backend derives final_decision)",
            "responds_to_reviewers": ["anonymous reviewer IDs whose arguments were addressed"],
            "technical_response": "Detailed response in complete sentences to the conflicting technical arguments, naming what was accepted or rejected and why.",
            "evidence_basis": "Complete statement identifying the specific P&ID, process, procedure, operating, or safeguard evidence and how it resolves or leaves the issue unresolved.",
            "position_changed": False,
            "change_reason": "Complete explanation of why the position changed or was retained after considering the other agents' arguments.",
            "proposed_resolution": "Complete explanation of the recommended worksheet treatment; include compact replacement wording only when a field must be revised."
        }],
    }, ensure_ascii=True, indent=2)


def round_two_new_scenario_review_json_shape(agent_label: str) -> str:
    return json.dumps({
        "agent_name": agent_label,
        "new_scenario_reviews": [{
            "proposal_id": "exact proposal_id supplied in the prompt",
            "criterion_assessment": technical_review_criterion_shape(),
            "suggested_decision": "agree | partially_agree | disagree (audit only; backend derives the official decision)",
            "reason": "technical assessment of novelty and credibility",
            "evidence_basis": "specific case evidence",
            "affected_fields": ["deviation | cause | consequence | safeguard | necessary_action"],
            "proposed_revision": "required correction, or No revision required"
        }],
    }, ensure_ascii=True, indent=2)


def expert_round_two_new_scenario_review_prompt(
    agent_id: str,
    agent_label: str,
    node: Dict[str, Any],
    proposals: List[Dict[str, Any]],
    round_one_review_rows: List[Dict[str, Any]],
    documents: Dict[str, str],
    batch_index: int,
    batch_count: int,
) -> str:
    """Require every agent to review every proposed Round 2 addition."""
    domain_expertise = domain_expertise_for(agent_id)
    node_scope_gate = node_scope_gate_text(node)
    required_ids = [str(proposal.get("proposal_id", "")) for proposal in proposals]
    # The full Round 1 records contain long evidence and discussion fields that
    # are unnecessary for duplicate checking and are repeated for every agent.
    # Send only the stable semantic identity of each existing scenario.
    round_one_duplicate_index = [{
        "review_id": row.get("review_id", ""),
        "matrix_key": row.get("matrix_key", ""),
        "guideword": row.get("guideword", ""),
        "parameter": row.get("parameter", ""),
        "deviation": row.get("deviation") or row.get("effect") or "",
        "potential_cause": row.get("potential_cause") or row.get("cause") or "",
        "consequence": row.get("consequence", ""),
    } for row in round_one_review_rows if isinstance(row, dict)]
    return f"""
You are the {agent_label} Agent reviewing proposed new scenarios during Round 2B.

{JSON_ONLY_RULE}

{prompt_method_for_stage("round_two")}

{domain_expertise}

Review EVERY proposed new scenario in this batch. These were raised after the
Round 1 exhaustive review. Decide whether each proposal is genuinely missing,
distinct from the Round 1 universe, physically credible within this node, and
supported by the supplied evidence. Proposal authorship is hidden, and the
proposal author is excluded from reviewing their own proposal.

{literature_grounded_review_rubric()}

Requirements:
- Return exactly one new_scenario_reviews entry per proposal_id, in the supplied order.
- Give a reason and evidence basis for every decision.
- Supply suggested_decision only after completing all five assessments. It does not control the official category.
- If any criterion is not pass, give a specific proposed revision or disposition. Only when all criteria pass write "No revision required".
- A majority opinion is not evidence. Apply your assigned discipline and the supplied case evidence.

{common_prompt_rules(include_worksheet=False)}

{node_scope_gate}

Return this JSON structure:
{round_two_new_scenario_review_json_shape(agent_label)}

NODE:
{compact_json(node, 30000)}

ROUND 1 REVIEW UNIVERSE FOR DUPLICATE CHECKING:
{compact_json(round_one_duplicate_index, 50000)}

DOCUMENTS:
{document_context(documents)}

BATCH-SPECIFIC PROPOSAL-REVIEW ASSIGNMENT:
This is proposal-review batch {batch_index} of {batch_count} for the current node.
Required proposal IDs: {json.dumps(required_ids, ensure_ascii=True)}

PROPOSED NEW SCENARIOS:
{compact_json(proposals, 70000)}
""".strip()


def expert_round_two_refinement_prompt(
    agent_id: str,
    agent_label: str,
    node: Dict[str, Any],
    contested_packets: List[Dict[str, Any]],
    documents: Dict[str, str],
    batch_index: int,
    batch_count: int,
    reviewer_id: str = "",
    reconciliation_pass: bool = False,
) -> str:
    """Prompt an evidence-based discussion or targeted reconciliation pass."""
    domain_expertise = domain_expertise_for(agent_id)
    node_scope_gate = node_scope_gate_text(node)
    required_ids = [str(packet.get("review_id", "")) for packet in contested_packets]
    prior_stage = "Round 2B discussion" if reconciliation_pass else "Round 2A cross-review"
    assignment = (
        "This is a targeted Round 2B reconciliation pass for scenarios that still lack a "
        "strict panel majority. The supplied anonymous reviews are the panel's latest "
        "post-discussion positions. Reassess the strongest conflicting case evidence "
        "and provide your own final category; do not change a category just to force a vote."
        if reconciliation_pass else
        "The scenarios below require focused discussion because categories differ, "
        "criterion ratings conflict, proposed corrections differ materially, or review "
        "coverage is incomplete."
    )
    return f"""
You are the {agent_label} Agent in Round 2B: cross-agent discussion refinement.

{JSON_ONLY_RULE}

{prompt_method_for_stage("round_two")}

{domain_expertise}

{prior_stage} preserved each agent's judgment. {assignment}
Read every anonymous reviewer's decision, reason, evidence basis, and proposed
revision. Then respond to the technical arguments and return one refined decision
for every supplied review_id. The panel-level result is reported as Agree,
Partially agree, or Disagree when a strict majority of the expected panel selects one category.
An unresolved vote receives a targeted reconciliation pass. If the panel still
lacks a majority, the Chair resolves the existing scenario in Round 3 using
the recorded arguments and case-specific evidence; the result is not labeled
as a panel majority.

{literature_grounded_review_rubric()}

Discussion requirements:
- Your anonymous identity in this discussion is {reviewer_id or 'the reviewer ID associated with your stored review'}.
- Address other positions only by anonymous reviewer ID in responds_to_reviewers and explain which technical argument you accept or reject. Do not infer or name engineering roles.
- Reconsider your supplied prior position; change it only when another agent supplied stronger case-specific evidence or a more coherent cause-effect chain.
- Do not change a decision merely to create unanimity. Unresolved disagreement is valid when the supplied evidence remains insufficient.
- If you are the Panel Chair, lead the technical resolution by explicitly reconciling the strongest conflicting evidence; do not use vote counts as technical proof.
- Distinguish scenario credibility from safeguard adequacy. Safeguard/action findings do not determine the scenario decision. A missing safeguard is recorded as "Not identified" and does not by itself cause Partially agree or Disagree.
- Reassess the three credibility checks and two separate audits after considering the other agents' arguments, then provide suggested_decision for audit. The backend derives final_decision.
- position_changed is recomputed by the backend from the derived final decision and the supplied prior decision.
- Give a complete technical_response, evidence_basis, change_reason, and proposed_resolution for every item.
- Every packet below includes the reviewed scenario and the anonymous {prior_stage} reviews. Read those exact fields before judging. If a field is genuinely empty, identify that field precisely; do not claim the whole packet or your prior decision is absent when it is supplied.
- Return exactly one refined_reviews entry for each required review_id, in the supplied order. Do not omit, merge, or invent IDs.

{common_prompt_rules(include_worksheet=False)}

{node_scope_gate}

Return this JSON structure:
{round_two_refinement_json_shape(agent_label)}

NODE:
{compact_json(node, 30000)}

DOCUMENTS:
{document_context(documents)}

BATCH-SPECIFIC DISCUSSION ASSIGNMENT:
This is refinement batch {batch_index} of {batch_count} for the current node.
Required review IDs: {json.dumps(required_ids, ensure_ascii=True)}

CONTESTED {prior_stage.upper()} REVIEW PACKETS:
{json.dumps(contested_packets, ensure_ascii=True, separators=(",", ":"))}
""".strip()


def chair_refine_prompt(
    node: Dict[str, Any],
    draft_rows: List[Dict[str, Any]],
    challenges: Any,
    round_one_review_rows: List[Dict[str, Any]] | None = None,
    documents: Dict[str, Any] | None = None,
) -> str:
    coverage_matrix = guideword_matrix_text(node)
    node_scope_gate = node_scope_gate_text(node)
    return f"""
You are the Panel Chair Agent in Round 3. Finalize the node HAZOP worksheet from the final Round 2B panel record and current-case technical evidence.

{JSON_ONLY_RULE}

{prompt_method_for_stage("chair_final")}

Rules:
- You synthesize the final worksheet for this node. Review every candidate again against the case evidence. A completed Round 2B panel majority controls the final category; record your assessment separately.
- You own and enforce the shared process understanding. Final rows must follow the supplied P&ID, material-flow/process-flow description, procedure steps, node boundary, design intent, guideword scope, and worksheet headings.
- Apply this evidence hierarchy to every final call: (1) current node boundary and design intent; (2) current-case P&ID image-parser evidence and shared process understanding; (3) current-case process description, procedures, operating conditions, and documented safeguards; (4) Round 2B rubric evidence and technical arguments. Generic engineering knowledge may test plausibility but may not create an undocumented case fact.
- For every criterion rating, cite the specific node field, P&ID tag/connection, process statement, operating condition, procedure statement, safeguard statement, or Round 2 evidence that supports it. A generic phrase such as "engineering judgment" is not sufficient case evidence.
- You already defined the Deviation / Effect for every required matrix item before Round 1. In Round 3, preserve that Chair-defined effect unless Round 2 established a stronger same-node technical correction.
- For procedural HAZOP nodes, finalize rows against the node procedure_initial_setup and procedure_steps, but write final worksheet cells in industrial HAZOP style. The row wording must state the actual abnormal effect, cause, consequence, safeguard, and action; do not write step labels or generic procedure phrases unless the effect/action itself explicitly needs that wording.
- Follow a completed Round 2B panel majority. If your evidence check identifies a possible technical error in that majority, document the exact evidence and discrepancy in chair_resolution for the audit and subsequent human expert review; do not silently reverse the category. If no majority remains after targeted reconciliation, make a final evidence-based Agree, Partially agree, or Disagree decision for the existing scenario and explain the conflicting arguments in chair_resolution. The backend records this as a Chair resolution, never as panel consensus.
- Apply the three credibility checks plus separate distinctness and safeguard/action audits to every revised_rows candidate. Complete chair_criterion_assessment before suggesting a decision. Your suggested label does not replace a completed panel majority.
- Agree is retained, Partially agree is retained with a necessary correction or open action, and Disagree is rejected. Apply this treatment to the panel majority or, if no majority remains, the documented Chair decision.
- Do not silently omit a candidate. Return every candidate in revised_rows with a complete assessment and a Chair resolution. Link same-meaning complete-chain duplicates to a retained scenario; preserve different initiating causes, consequences, safeguards, and actions as separate findings.
- Return at least one adjudicated candidate for every distinct draft-row candidate and every Round 2 new-scenario proposal supplied below. This makes retained, revised, duplicate, and rejected treatments auditable rather than inferring them from omission.
- If you identify a genuinely missing scenario that was never reviewed in Round 2, put it in round_3_escalated_proposals with all five assessments and case evidence. It requires targeted review by every eligible non-author panel member before worksheet inclusion.
- Use the supplied final Round 2B scenario_deliberations and new_scenario_deliberations records as the panel decision basis. Earlier independent ratings are retained in the backend audit record and are not supplied as competing decisions in Round 3.
- In the scenario-centric digest, round_2b_decisions are the final post-discussion decisions, including unchanged ratings carried forward where discussion was unnecessary. Missing assignments remain blank and must not be interpreted as agreement or disagreement.
- Use round_2b_panel_classification as the post-discussion panel conclusion only when a strict majority of the expected panel selected one category. The Chair has one equal vote. A tie is not converted into a panel conclusion; after targeted reconciliation you must document your own final decision from the evidence. Missing reviews are never imputed.
- Treat incomplete review coverage as an audit limitation. Follow a completed panel majority. If the panel still has no majority after reconciliation, explain the evidence gap and make the final decision from the available technical record; do not attribute it to the panel.
- Use round_2b_criterion_summary to identify which exact criteria passed, required revision, or failed across the panel. Your Chair assessment must address those stored findings and may not replace them with an unsupported general impression.
- Use agent_arguments to understand each member's final technical reason, criterion assessment, evidence basis, and proposed resolution. Do not reconstruct an earlier vote.
- Document remaining contested evidence. Preserve dissent and the supporting technical reasons, especially when the Chair had to resolve a panel tie.
- Review every new_scenario_deliberations item using its panel conclusion, decision counts, dissent, and coverage. Safeguard availability is recorded separately and must not change the scenario credibility decision.
- Reason internally through the guideword logic, credibility checks, and expert disagreements. Output only the Chair decision summary and final worksheet rows, not private chain-of-thought.
- Write a clear final Chair synthesis explaining what you accepted, rejected, changed, and why.
- Finalize only rows that can be traced to parsed P&ID evidence, supplied process/procedure text, supplied operating data, supplied safeguards, or supplied worksheet evidence for the same node and same guideword/effect.
- Treat any supplied example worksheet as style evidence only unless it belongs to the current uploaded case and the same node/guideword/effect. Never copy reference/example rows into a different case or unrelated guideword item.
- Apply valid wording corrections to panel-supported scenarios. If you believe a panel category is technically wrong, document the exact error and evidence for the audit and subsequent human expert review; do not silently change the category. Escalate genuinely missing scenarios for targeted review by all eligible non-author panel members.
- Reference which agent positions were accepted or rejected when important.
- Preserve unresolved panel votes in the audit and identify each Chair resolved tie explicitly in the final synthesis.
- Final row chain rule: for each required guideword item, first finalize the Deviation/Effect, then accept only causes that produce that exact effect, only consequences that follow from those causes, only safeguards that mitigate that chain, and only actions that close the remaining row-specific gap.
- Cross-guideword finalization rule: never use another row as a source of content for the current row. A bullet may be retained only if it belongs to the current matrix_key and current Deviation/Effect. If a bullet belongs to a different guideword category, remove it from this row even if it is technically true elsewhere in the node.
- Effect gate: no final credible row may have a missing, placeholder, qualifier-only, or cause-like Deviation/Effect. If the actual effect cannot be repaired from same-node P&ID/process/procedure evidence, reject the candidate with the failed criterion and evidence gap recorded; do not insert a placeholder final row.
- Procedural guideword gate: for Out of Sequence, Rate, Magnitude, Pressure, Communication, Reverse, Incomplete, and Others items, first test whether the node design intention can fail under that item. If it can, write the physical/operational effect and complete the cause-consequence-safeguard-action chain. Do not mark the item Not applicable just because the matrix qualifier is "Too early", "Too late", "Too fast", "Too slow", "More", or "Less".
- Order gate: return rows in the exact order of the REQUIRED GUIDEWORD/PARAMETER MATRIX. Do not move Communication, Reverse, Incomplete, or Others content into earlier guideword rows.
- Safeguard evidence search order: first inspect safeguards documented on the current matrix item; then inspect confirmed current-node safeguards assembled from both parsed P&ID evidence and the supplied process/procedure description. Do not search or borrow controls from unrelated nodes or facility background.
- Safeguard gate: preserve a safeguard only when its documented mechanism mitigates the same current-node cause-effect-consequence chain. Consider applicable hardware, instrumented, passive, detection/mitigation, and specifically documented administrative controls; do not restrict the search to a short predefined device list, and do not infer a control that is absent from the supplied evidence.
- Action gate: every action must address the exact row gap. Do not repeat one generic action across rows unless the supplied case uses that same action for the same row chain.
- Sequential consistency rule: use earlier rows in the same node only as context for operation sequence and equipment state. Do not let an earlier guideword row donate causes, consequences, safeguards, or actions to a later guideword row unless the Chair explicitly confirms the same causal chain applies.
- Repetition control rule: if the same safeguard or action appears in multiple rows, keep it only where it is documented for this case and technically mitigates that row's exact cause-effect-consequence chain. Remove repeated generic administrative or placeholder items unless the supplied case documents the exact control and the row needs it.
- Matrix control rule: one or more rows may be returned for each matrix_key, each with a traceable cause, consequence, safeguard, and action chain. Before finalizing, compare the complete meaning of each chain. Link a duplicate only when its deviation, initiating cause, consequences, safeguards, and actions all mean the same thing; preserve any different technical element as a separate finding. Never merge distinct chains into one atomic row.
- Reference-style rule: use supplied reference/manual worksheet rows only to understand the level of specificity and compactness. Do not copy their content unless the row is for this same uploaded case, same node, same matrix_key, and same effect/deviation.
- Treat Round 2 disagreement about safeguards as a safeguard adequacy issue, not as a reason to delete the deviation.
- If a cause and consequence are credible but safeguards are disputed, keep the row and set safeguard to "Not identified"; place any verification need in Necessary Action.
- Reject a scenario when a credibility check fails or the entire same-meaning causal and control chain duplicates a retained scenario; document the technical reason and retained row ID.
- Do not replace credible Round 1 or Round 2 scenarios with "Not applicable" rows.
- A "Not applicable" row is allowed only when no credible expert scenario remains for that required matrix item.
- Final safeguard cleanup rule: if a row has a real documented safeguard, remove "Not identified", "None", "N/A", and "Not applicable" from that Safeguard cell.
- For every retained safeguard, provide safeguard_evidence naming the exact same-node tag, control, or supplied statement. If that evidence is unavailable, set safeguard to "Not identified" and leave safeguard_evidence empty.
- Populate source_review_ids and source_proposal_ids from the supplied Round 1/2 identifiers. Do not invent identifiers; if no conservative match exists, return the candidate with a Disagree assessment so the backend can record and exclude it.
- A credible final row must link to at least one supplied source_review_id or source_proposal_id. An unlinked Chair-generated row is excluded by the backend because it has no Round 2 panel basis.
- Final safeguard cleanup rule: remove procedure step labels and intended procedure text from Safeguard cells; convert needed procedure changes into Necessary Action.
- Final action cleanup rule: replace repeated generic referrals with row-specific actions tied to the guideword/effect, supplied safeguards, and the exact missing protection or correction.
- No placeholder completion: do not write "Assign row-specific review", "Identify row-specific safeguard", "Correct valve operating sequence", or similar filler. If an action is needed, name the specific technical correction, check, review, or responsible discipline for that row.
- Final worksheet length rule: after the technical decision is complete, run a final compression pass on every worksheet cell. Keep complete technical phrases, preserve tags/materials/actions, remove explanatory filler, and obey the worksheet cell length contract.

{common_prompt_rules(include_worksheet=True)}

{node_scope_gate}

REQUIRED GUIDEWORD/PARAMETER MATRIX FOR THIS NODE:
{coverage_matrix}

{{
  "round_3_chair_summary": "(discussion field) Detailed overview in complete sentences explaining how the Chair finalized the node findings from the Round 2B panel record and current-case evidence.",
  "round_3_chair_synthesis": [
    "(discussion field, full sentences) Complete descriptive bullet explaining the main credible deviations retained by the Chair and the technical basis for retaining them.",
    "(discussion field, full sentences) Complete descriptive bullet explaining which Round 2 suggestions were rejected or modified, and why they were not technically appropriate.",
    "(discussion field, full sentences) Complete descriptive bullet explaining the final safeguard and necessary-action logic for the worksheet."
  ],
  "round_3_escalated_proposals": [
    {{
      "proposal_id": "N-001-R3P-001",
      "guideword": "exact required guideword",
      "parameter": "required parameter",
      "matrix_key": "required matrix_key",
      "deviation": "missing deviation/effect identified by the Chair",
      "potential_cause": "specific initiating cause",
      "consequence": ["credible consequence"],
      "safeguard": ["documented safeguard or Not identified"],
      "necessary_action": ["specific action needed"],
      "chair_criterion_assessment": {json.dumps(technical_review_criterion_shape(), ensure_ascii=True)},
      "chair_suggested_decision": "agree | partially_agree | disagree (audit only)",
      "chair_evidence_basis": ["specific supplied case evidence"],
      "proposal_reason": "why this scenario was missing from Round 1 and Round 2",
      "required_next_step": "Full-panel review before final worksheet inclusion"
    }}
  ],
  "revised_rows": [
    {{
      "row_id": "N-001-R001",
      "guideword": "use one exact guideword from the required matrix",
      "parameter": "use the required parameter from the same matrix item",
      "matrix_key": "use the exact matrix_key from the required matrix item",
      "expected_deviation": "use the Chair-defined Deviation / Effect from the required matrix item when supplied",
      "deviation": "write the actual Chair-finalized deviation/effect for this matrix item",
      "potential_cause": "Compact initiating cause -- this row's one cause. A second distinct credible cause for this same matrix item is a second revised_rows entry sharing this matrix_key, not a second phrase here.",
      "consequence": [
        "Compact credible consequence following from this row's one cause.",
        "Distinct compact consequence, still tied to this row's cause only."
      ],
      "safeguard": [
        "Documented row-specific safeguard."
      ],
      "necessary_action": ["Compact row-specific action."],
      "source_review_ids": ["Round 1 review_id supporting this final row"],
      "source_proposal_ids": ["Round 2 proposal_id when this row originated as a panel-reviewed addition"],
      "safeguard_evidence": ["Exact same-node tag, control, or supplied statement supporting each retained safeguard"],
      "chair_criterion_assessment": {json.dumps(technical_review_criterion_shape(), ensure_ascii=True)},
      "chair_suggested_decision": "agree | partially_agree | disagree (audit only; backend derives the official Chair decision)",
      "chair_evidence_basis": ["Detailed, complete statement identifying the case-specific evidence supporting the final treatment and how it was applied."],
      "chair_revision_summary": "(discussion field) Complete explanation of the correction or open action applied for partially_agree; otherwise explain why no revision was required or why the candidate was excluded.",
      "chair_resolution": "(discussion field) Detailed statement, using as many complete sentences as needed, explaining which expert points were accepted, modified, or rejected, why, and how that decision is represented in the concise worksheet row."
    }}
  ]
}}

NODE:
{compact_json(node, 30000)}

DRAFT ROWS:
{compact_json(draft_rows, 70000)}

FINAL ROUND 2B PANEL RECORD:
{compact_json(challenges, 180000)}

CURRENT-CASE PROCESS AND P&ID EVIDENCE:
{document_context(documents or {})}
""".strip()


def scribe_prompt(node: Dict[str, Any], revised_rows: List[Dict[str, Any]]) -> str:
    coverage_matrix = guideword_matrix_text(node)
    node_scope_gate = node_scope_gate_text(node)
    return f"""
You are the Scribe & Report Agent producing final HAZOP worksheet rows.

{JSON_ONLY_RULE}

{prompt_method_for_stage("scribe")}

Keep node and parameter fields so the worksheet remains traceable node-by-node.
Use "Not applicable" for required matrix items that are explicitly not applicable only when the chair retained them.
Do not drop Not applicable coverage rows.
Do not rewrite credible Chair-finalized scenarios as Not applicable.
If revised_rows contain specific causes and consequences, preserve them in final_rows. If revised_rows has more than one row for the same matrix_key (one row per distinct cause), preserve every one of them as its own final_rows entry -- never re-merge multiple causes back into a single final row. Before finalizing, run one more check across final_rows sharing a matrix_key: if two rows' causes are the same initiating mechanism reworded, keep only one.
Your main task is formatting and compression. Preserve the Chair's technical meaning, but rewrite long bullets into short, complete HAZOP-table phrases that satisfy the worksheet cell length contract.
Do not add explanatory text, bracketed placeholders, source references, assumptions, or new technical scenarios.
Do not copy any example worksheet content unless the Chair-finalized row already uses it for the current uploaded case and same node/guideword/effect.
Do not borrow a safeguard or action from another row. If the Chair row has "Not identified", preserve "Not identified" instead of inventing a generic safeguard.
Do not introduce filler actions such as "Assign row-specific review", "Identify row-specific safeguard", "Check procedure", or "Correct valve operating sequence" unless the Chair row already names that exact technical action with case evidence.
If a Chair-finalized row is verbose, compress wording by deleting explanation, not by deleting technical actors, materials, valve/equipment tags, destinations, or consequence mechanisms.
If a Chair-finalized row is internally inconsistent, preserve the row but flag the issue in a compact action rather than inventing a new scenario.
If a Chair-finalized row contains content from another guideword, do not blend it into the current row. Remove the cross-guideword bullet from the current row and preserve only the row's own guideword-effect-cause-consequence-safeguard-action chain.
Preserve the Chair row order exactly as the required matrix order.
Do not finalize a credible row with "Effect not identified", "Deviation not identified", a guideword qualifier only, or a cause-like phrase in Deviation / Effect. If the Chair row has no real effect, return that matrix item as Not applicable.
Do not convert a Chair-finalized procedural row to Not applicable unless the Chair already finalized that matrix item as Not applicable. For Out of Sequence, Rate, Magnitude, Pressure, Communication, Reverse, Incomplete, and Others rows, preserve the Chair's physical/operational effect.
Do not add, copy, or broaden safeguards/actions during formatting. Safeguards and actions must remain row-specific and evidence-supported.

{common_prompt_rules(include_worksheet=True, final_worksheet=True)}

{node_scope_gate}

REQUIRED GUIDEWORD/PARAMETER MATRIX FOR THIS NODE:
{coverage_matrix}

{{
  "final_rows": [
    {{
      "row_id": "N-001-R001",
      "node_description": "node description text",
      "design_intent": "design intent text",
      "guideword": "use one exact guideword from the required matrix",
      "parameter": "use the required parameter from the same matrix item",
      "matrix_key": "use the exact matrix_key from the required matrix item",
      "expected_deviation": "use the Chair-defined Deviation / Effect from the required matrix item when supplied",
      "deviation": "write the actual Chair-finalized deviation/effect for this matrix item",
      "potential_cause": "Compact Chair-finalized cause -- this row's one cause. A distinct second cause for this same matrix_key is a second final_rows entry, not a second phrase here.",
      "consequence": [
        "Compact Chair-finalized consequence following from this row's one cause.",
        "Distinct compact consequence, still tied to this row's cause only."
      ],
      "safeguard": [
        "Compact documented safeguard with its row-specific protective function."
      ],
      "necessary_action": ["Compact Chair-finalized action."],
      "action_required": ["Compact Chair-finalized action."]
    }}
  ]
}}

NODE:
{compact_json(node, 30000)}

REVISED ROWS:
{compact_json(revised_rows, 80000)}
""".strip()


def report_prompt(nodes: List[Dict[str, Any]], worksheet: List[Dict[str, Any]], actions: List[Dict[str, Any]]) -> str:
    return f"""
You are the Scribe & Report Agent producing the final HAZOP report summary.

{JSON_ONLY_RULE}

{prompt_method_for_stage("report")}

Summarize only the qualitative HAZOP result:
- nodes studied and their design intents
- significant deviations, causes, consequences, and safeguards
- necessary actions
- study limitations, if any
- external URL/DOI source references used for technical claims, if any
Keep the report limited to qualitative HAZOP worksheet content and action follow-up.

{common_prompt_rules()}

{{
  "executive_summary": "short summary of scope and results",
  "key_findings": ["finding 1", "finding 2"],
  "assumptions_and_limitations": ["missing data limitation 1"],
  "source_references": ["external reference title - URL or DOI"],
  "node_count": 0,
  "worksheet_row_count": 0,
  "action_count": 0
}}

NODES:
{compact_json(nodes, 50000)}

WORKSHEET ROWS:
{compact_json(worksheet, 100000)}

ACTION REGISTER:
{compact_json(actions, 60000)}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def system_prompt_for(agent_id: str) -> str:
    labels = {agent_id_: label for agent_id_, label, _ in AGENTS}
    label = labels.get(agent_id, agent_id)
    domain_expertise = domain_expertise_for(agent_id)
    return (
        f"You are the {label} Agent in a fully automated HAZOP study. "
        f"{domain_expertise} "
        "Follow the stage-specific task instructions in the user prompt. "
        "Do not apply worksheet-generation rules to extraction, planning, or discussion stages unless that stage explicitly asks for worksheet rows. "
        "The study purpose is to identify hazards and scenarios through qualitative HAZOP; do not over-solve. "
        "If a solution is obvious, recommend it in the required action field; otherwise refer to the appropriate engineering team. "
        "Reason internally step by step using your assigned expert role, but output only the requested JSON fields and concise technical rationale; do not output private chain-of-thought. "
        "Be technically rigorous, conservative, concise, and auditable. "
        "Use stable wording and row structure across repeated runs when the supplied evidence is the same. "
        "When the stage asks for final worksheet cells, use compact industrial HAZOP phrases, not long narrative paragraphs. "
        "Do not copy illustrative example tags, numeric values, or safeguards unless they appear in the supplied documents. "
        f"{JSON_ONLY_RULE} "
        "When information is missing, do not invent it; record the missing item in the affected technical cell or in a row-specific Necessary Action."
    )


def compact_retry_prompt(agent_label: str, expected_shape: str, invalid_response: str) -> str:
    return f"""
Your previous {agent_label} response was not valid JSON or was truncated.

Return one compact valid JSON object only. No markdown. No prose outside JSON.
Use this exact shape:
{expected_shape}

Rules:
- Use complete technical sentences; do not return clipped phrases or half statements.
- Keep content compact without removing essential cause, consequence, safeguard, or action context.
- Return every required scenario_reviews item when repairing an exhaustive Round 2 review. Do not truncate that required list. Other optional lists may contain at most 5 items.
- Close every bracket and brace.
- Do not add new claims not already supported by the node/documents or your previous answer.

INVALID/TRUNCATED RESPONSE TO REPAIR:
{clamp_text(invalid_response, 12000)}
""".strip()


def round_one_json_shape(agent_label: str) -> str:
    return json.dumps({
        "agent_name": agent_label,
        "opening_statement": "complete expert view",
        "round_1_position": [
            "Complete sentence describing this expert's domain interpretation of the node.",
            "Complete sentence describing the main credible deviations this expert wants the panel to discuss."
        ],
        "domain_findings": [
            "Complete technical finding with cause mechanism and node-specific basis.",
            "Complete technical finding with missing evidence or safeguard uncertainty."
        ],
        "scenarios": [{
            "guideword": "use one exact guideword from the required matrix",
            "parameter": "use the required parameter from the same matrix item",
            "matrix_key": "use the exact matrix_key from the required matrix item",
            "expected_deviation": "use the Chair-defined Deviation / Effect from the required matrix item when supplied",
            "deviation": "actual deviation or effect for this matrix item",
            "credible": True,
            "potential_cause": "Compact cause phrase -- this scenario's one cause.",
            "consequence": ["Compact consequence phrase."],
            "safeguard": ["Documented safeguard phrase only when supplied; otherwise one Not identified entry."],
            "safeguards": ["Compact safeguard phrase."],
            "concerns": ["Complete concern statement."],
            "evidence": ["Complete visible or supplied evidence statement."]
        }],
        "not_applicable": []
    }, ensure_ascii=True, indent=2)


def chair_effect_definition_prompt(node: Dict[str, Any], documents: Dict[str, str]) -> str:
    coverage_matrix = guideword_matrix_text(node)
    node_scope_gate = node_scope_gate_text(node)
    return f"""
You are the Panel Chair Agent before Round 1. Your task is to define the Deviation / Effect for every required guideword-matrix item.

{JSON_ONLY_RULE}

{prompt_method_for_stage("chair_effect_definition")}

{domain_expertise_for("panel_chair")}

This is a Chair-only setup step. Do not identify causes, consequences, safeguards, or actions here.
The specialist experts will use your Deviation / Effect definitions as the fixed starting point for Round 1 and Round 2.

Rules:
- Use the supplied P&ID, process description, procedure steps, node boundary, design intent, and any supplied manual worksheet evidence for this same case and same node.
- Preserve the exact order of the REQUIRED GUIDEWORD/PARAMETER MATRIX.
- Define one Deviation / Effect for each matrix_key.
- For each matrix_key, first ask: "If this guideword happens to this node design intent, what abnormal state would the panel write in the worksheet Effect/Deviation column?"
- Apply this same test to every guideword, including Out of Sequence, Rate, Magnitude, Pressure, Communication, Reverse, Incomplete, and Others. Do not start from Not applicable; start from the node design intent and derive the physical/operational effect, then use Not applicable only if no credible abnormal state exists.
- For procedural HAZOP, the Deviation / Effect must be the actual abnormal physical or operational state created by the wrong sequence, wrong rate, wrong magnitude, pressure change, communication failure, reverse path, incomplete step, or other deviation.
- For Out of Sequence (Too early) and Out of Sequence (Too late), do not return Not applicable only because the timing qualifier is abstract. Derive the effect from the node design intent and sequence. Prefer manual-HAZOP effect wording such as no service flow, high service flow, residual air/material, contamination, reverse flow, or abnormal pressure over abstract timing wording.
- The Deviation / Effect is not the cause. Do not write operator error, valve failure, distraction, regulator failure, or maintenance failure in this field.
- The Deviation / Effect is not the consequence. Do not write injury, environmental impact, fire, delay, or cost in this field unless that is the immediate abnormal process state.
- The Deviation / Effect is not just the guideword qualifier. Do not write placeholder phrases such as "Effect not identified", "Deviation not identified", "Out of Sequence sequence", "Too early", "Too late", "Too fast", "Too slow", "Communication", "Incomplete", or "Others".
- If the supplied text contains an explicit same-node worksheet row, same-node effect/evidence, or same matrix item with cause/consequence/safeguard/action evidence, that matrix item is not Not applicable. Use the supplied same-node evidence to define the Deviation / Effect.
- If the guideword item is credible, write a compact case-specific effect/deviation phrase tied to the node boundary, material/service, valve/equipment tags, and design intent.
- Use Not applicable only when the guideword item cannot credibly change the node state or design intent after checking the P&ID, procedure/process text, and supplied guideword basis.
- Never use Not applicable for a matrix item only because the effect is hard to word. If evidence exists, write the best case-specific effect phrase and let experts analyze causes, consequences, safeguards, and actions.
- Do not copy example rows from another node or another case. If the uploaded case contains a matching worksheet row for this same node/effect, use it only as evidence for compact wording.
- Do not invent new tags, equipment, process states, or safeguards.
- Self-check before output: every credible Deviation / Effect must be understandable as a worksheet cell without reading the cause column, and every Not applicable item must have no credible abnormal node state.

{node_scope_gate}

REQUIRED GUIDEWORD/PARAMETER MATRIX:
{coverage_matrix}

Return exactly this JSON object:
{{
  "chair_effect_definition_summary": [
    "Complete sentence stating how the Chair used the node design intent, P&ID evidence, and procedure/process basis to define effects before expert elicitation.",
    "Complete sentence stating that the experts must analyze these effects rather than redefine the guideword matrix."
  ],
  "effect_definitions": [
    {{
      "matrix_key": "exact matrix_key from the required matrix",
      "guideword": "guideword from the required matrix",
      "parameter": "parameter from the required matrix",
      "deviation": "case-specific abnormal Deviation / Effect, or Not applicable",
      "applicability": "credible | not_applicable",
      "basis": "short technical basis tied to this node boundary and design intent"
    }}
  ]
}}

NODE:
{compact_json(node, 30000)}

DOCUMENTS:
{document_context(documents)}
""".strip()
