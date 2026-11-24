"""
pid_parser.py — Production P&ID Parser for HaznovA HAZOP Agent System
=======================================================================
Three-stage pipeline:
  Stage 1 — Local CV (OpenCV + Tesseract OCR)
            Deterministic, runs offline, no API cost.
            Extracts: pipe-line evidence, tag positions, symbol bounding boxes,
            text labels, elevations, flow directions.

  Stage 2 — Vision LLM overlay (GPT-4o / Claude / Gemini)
            Visible-object classification, disambiguation of tags,
            connectivity inference, node boundary suggestion.
            Uses Stage 1 output as structured hints so the LLM
            doesn't have to hallucinate positions.

  Stage 3 — Merge + structured output
            Fuses CV detections with LLM understanding into a single
            structured PIDParse dict that the HAZOP planner consumes.

Usage inside hazop_backend.py:
    from pid_parser import parse_pid
    result = parse_pid(payload)
    # result is a PIDParse dict with keys:
    #   equipment, instruments, pipes, nodes, uncertainties, raw_ocr_words
"""

from __future__ import annotations

import base64
import io
import json
import math
import subprocess
import re
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
try:
    import pytesseract
except ImportError:
    pytesseract = None
from PIL import Image


# ─────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

@dataclass
class BBox:
    x: int; y: int; w: int; h: int
    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2
    @property
    def area(self): return self.w * self.h
    def as_fraction(self, img_w: int, img_h: int) -> Dict:
        return {
            "x1": round(self.x / img_w, 4), "y1": round(self.y / img_h, 4),
            "x2": round((self.x + self.w) / img_w, 4), "y2": round((self.y + self.h) / img_h, 4),
        }

@dataclass
class OcrWord:
    text: str; conf: int; x: int; y: int; w: int; h: int

@dataclass
class TaggedElement:
    tag: str
    element_type: str   # 'equipment' | 'instrument' | 'valve' | 'line' | 'unknown'
    subtype: str        # 'heat_exchanger' | 'vessel' | 'pump' | 'column' | ...
    bbox: Optional[BBox] = None
    nearby_tags: List[str] = field(default_factory=list)
    description: str = ""
    confidence: str = "medium"   # high / medium / low

@dataclass
class Pipe:
    pipe_id: str
    orientation: str   # 'horizontal' | 'vertical'
    x1: int; y1: int; x2: int; y2: int
    length: int
    connected_tags: List[str] = field(default_factory=list)

@dataclass
class PIDParse:
    image_width: int
    image_height: int
    equipment: List[TaggedElement] = field(default_factory=list)
    instruments: List[TaggedElement] = field(default_factory=list)
    valves: List[TaggedElement] = field(default_factory=list)
    pipes: List[Pipe] = field(default_factory=list)
    raw_ocr_words: List[OcrWord] = field(default_factory=list)
    candidate_nodes: List[Dict] = field(default_factory=list)
    elevations: List[Dict] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    cv_summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "image_size": [self.image_width, self.image_height],
            "equipment": [_elem_dict(e) for e in self.equipment],
            "instruments": [_elem_dict(e) for e in self.instruments],
            "valves": [_elem_dict(e) for e in self.valves],
            "pipes": {
                "horizontal_count": sum(1 for p in self.pipes if p.orientation == "horizontal"),
                "vertical_count":   sum(1 for p in self.pipes if p.orientation == "vertical"),
                "total_count":      len(self.pipes),
                "sample": [asdict(p) for p in self.pipes[:20]],
            },
            "candidate_nodes": self.candidate_nodes,
            "elevations": self.elevations,
            "labels": self.labels[:40],
            "raw_ocr_words": [asdict(w) for w in self.raw_ocr_words[:200]],
            "uncertainties": self.uncertainties,
            "cv_summary": self.cv_summary,
        }


def _elem_dict(e: TaggedElement) -> Dict:
    d = {
        "tag": e.tag, "type": e.element_type, "subtype": e.subtype,
        "description": e.description, "confidence": e.confidence,
        "nearby_tags": e.nearby_tags,
    }
    if e.bbox:
        d["bbox_px"] = {"x": e.bbox.x, "y": e.bbox.y, "w": e.bbox.w, "h": e.bbox.h,
                        "cx": e.bbox.cx, "cy": e.bbox.cy}
    return d


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


# ─────────────────────────────────────────────────────────────────
# STAGE 1A: IMAGE LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────

def load_image_from_data_url(data_url: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Load a data:image/...;base64,... URL.
    Returns (rgb_array, gray_array, width, height).
    """
    match = re.match(r"data:image/[^;]+;base64,(.+)", data_url, re.DOTALL)
    if not match:
        raise ValueError("Not a valid data URL")
    img_bytes = base64.b64decode(match.group(1))
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Resize very large images to max 4000px on longest side for speed
    max_dim = 4000
    orig_w, orig_h = pil.size
    scale = min(1.0, max_dim / max(orig_w, orig_h))
    if scale < 1.0:
        pil = pil.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)

    rgb = np.array(pil)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    return rgb, gray, w, h


def preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    """
    Best preprocessing for OCR on engineering drawings:
    1. Sharpen (engineering text is often small and aliased)
    2. Adaptive threshold (handles uneven illumination and background grids)
    3. Upscale if image is small (Tesseract works best at ~300 DPI)
    """
    h, w = gray.shape
    # Upscale small images
    if w < 2000:
        scale = 2000 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Adaptive threshold — handles grid lines and background boxes
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10,
    )
    return binary


def preprocess_for_lines(gray: np.ndarray) -> np.ndarray:
    """
    Preprocessing optimised for pipe/line detection.
    Gaussian blur removes noise; Canny finds sharp edges.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return blurred


# ─────────────────────────────────────────────────────────────────
# STAGE 1B: OCR — EXTRACT ALL TEXT WITH SPATIAL POSITIONS
# ─────────────────────────────────────────────────────────────────

# Tesseract config: PSM 11 (sparse text, no word grouping) for P&IDs
# OEM 3 uses the LSTM engine (most accurate)
_TESS_CONFIG = "--psm 11 --oem 3"
_MIN_CONF = 30

def run_ocr(gray: np.ndarray, orig_w: int, orig_h: int) -> List[OcrWord]:
    """
    Run Tesseract OCR on the image.
    Returns word-level detections with pixel positions in original image space.
    """
    binary = preprocess_for_ocr(gray)
    ocr_h, ocr_w = binary.shape
    scale_x = orig_w / ocr_w
    scale_y = orig_h / ocr_h

    if pytesseract is None:
        return merge_ocr_words([
            run_ocr_cli(binary, orig_w, orig_h, psm="11"),
            run_ocr_cli(gray, orig_w, orig_h, psm="6"),
            run_ocr_cli(gray, orig_w, orig_h, psm="12"),
        ])
    try:
        data = pytesseract.image_to_data(
            Image.fromarray(binary),
            config=_TESS_CONFIG,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return merge_ocr_words([
            run_ocr_cli(binary, orig_w, orig_h, psm="11"),
            run_ocr_cli(gray, orig_w, orig_h, psm="6"),
            run_ocr_cli(gray, orig_w, orig_h, psm="12"),
        ])
    words: List[OcrWord] = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        conf = int(data["conf"][i])
        if not txt or conf < _MIN_CONF or len(txt) < 2:
            continue
        words.append(OcrWord(
            text=txt,
            conf=conf,
            x=int(data["left"][i] * scale_x),
            y=int(data["top"][i] * scale_y),
            w=int(data["width"][i] * scale_x),
            h=int(data["height"][i] * scale_y),
        ))
    if len(words) < 10:
        words = merge_ocr_words([
            words,
            run_ocr_cli(gray, orig_w, orig_h, psm="6"),
            run_ocr_cli(gray, orig_w, orig_h, psm="12"),
        ])
    return words


def run_ocr_cli(image: np.ndarray, orig_w: int, orig_h: int, psm: str = "11") -> List[OcrWord]:
    """
    Fallback OCR path using the installed tesseract binary when pytesseract
    is not installed in the active Python environment.
    """
    ocr_h, ocr_w = image.shape[:2]
    scale_x = orig_w / ocr_w
    scale_y = orig_h / ocr_h
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        Image.fromarray(image).save(tmp.name)
        try:
            proc = subprocess.run(
                ["tesseract", tmp.name, "stdout", "--psm", psm, "--oem", "3", "tsv"],
                check=False,
                text=True,
                capture_output=True,
                timeout=60,
            )
        except Exception:
            return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    lines = proc.stdout.splitlines()
    if not lines:
        return []
    headers = lines[0].split("\t")
    index = {name: i for i, name in enumerate(headers)}
    required = {"text", "conf", "left", "top", "width", "height"}
    if not required.issubset(index):
        return []
    words: List[OcrWord] = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < len(headers):
            continue
        txt = (cols[index["text"]] or "").strip()
        try:
            conf = int(float(cols[index["conf"]]))
        except ValueError:
            conf = -1
        if not txt or conf < _MIN_CONF or len(txt) < 2:
            continue
        try:
            left = float(cols[index["left"]])
            top = float(cols[index["top"]])
            width = float(cols[index["width"]])
            height = float(cols[index["height"]])
        except ValueError:
            continue
        words.append(OcrWord(
            text=txt,
            conf=conf,
            x=int(left * scale_x),
            y=int(top * scale_y),
            w=int(width * scale_x),
            h=int(height * scale_y),
        ))
    return words


def merge_ocr_words(word_sets: List[List[OcrWord]]) -> List[OcrWord]:
    merged: List[OcrWord] = []
    seen: set[tuple[str, int, int]] = set()
    for words in word_sets:
        for word in words:
            key = (word.text.lower(), round(word.x / 12), round(word.y / 12))
            if key in seen:
                continue
            seen.add(key)
            merged.append(word)
    return sorted(merged, key=lambda item: (item.y, item.x))


# ─────────────────────────────────────────────────────────────────
# STAGE 1C: TAG CLASSIFICATION
# ─────────────────────────────────────────────────────────────────

# Comprehensive P&ID tag patterns grouped by category
_INSTRUMENT_RE = re.compile(
    r"^("
    # Flow
    r"FI[CRTQSAEF]?|FC|FT|FRC[A]?|FFI[C]?|FR[C]?|"
    # Pressure
    r"PI[CRTS]?|PC|PT|PRC[A]?|PDI|PDIC|"
    r"PSV|PRV|PSH|PSL|PAH|PAL|PAHH|PALL|"
    # Temperature
    r"TI[CRTS]?|TC|TT|TRC[A]?|TAH|TAL|TAHH|TALL|"
    # Level
    r"LI[CRTS]?|LC|LT|LRC[A]?|LAH|LAL|LAHH|LALL|"
    r"LSH|LSL|LSHH|LSLL|LIA[+-]?|"
    # Analysis / other
    r"AI[CRS]?|AT|AIC|AR[C]?|"
    # Generic alarms / switches
    r"FAH|FAL|FAHH|FALL|"
    r"HS|HV|HIC|"
    # Valves (actuated)
    r"XV|FV|PV|TV|LV|"
    # Safety
    r"ESD|SIS|SIL|"
    # Signal nodes (AN = Anschluss/node in German P&IDs)
    r"AN"
    r")[- _]?\d*[A-Z]?$",
    re.IGNORECASE,
)

_EQUIPMENT_RE = {
    "heat_exchanger":  re.compile(r"^(HE|E|HX)[- _]?\d+[A-Z]?$", re.I),
    "vessel":          re.compile(r"^(VE|V|TK|T|D|DR|FA|FB)[- _]?\d+[A-Z]?$", re.I),
    "pump":            re.compile(r"^(PL|PU|P|G)[- _]?\d+[A-Z]?$", re.I),
    "column":          re.compile(r"^(CO|C|DC|SC|AC|K|T)[- _]?\d+[A-Z]?$", re.I),
    "reactor":         re.compile(r"^(R|RE|RX)[- _]?\d+[A-Z]?$", re.I),
    "compressor":      re.compile(r"^(C|K|COMP)[- _]?\d+[A-Z]?$", re.I),
    "filter":          re.compile(r"^(F|FL|SF|DF)[- _]?\d+[A-Z]?$", re.I),
    "agitator":        re.compile(r"^(AG|MX|M)[- _]?\d+[A-Z]?$", re.I),
    "valve_manual":    re.compile(r"^(VV|V|BV|GV|BFV)[- _]?\d+[A-Z]?$", re.I),
}

_LINE_RE = re.compile(
    r"^\d{1,3}[- /][A-Z0-9]{1,10}[- /]\d{2,6}([- /][A-Z0-9]{1,8})?$",
    re.I,
)

_ELEVATION_RE = re.compile(r"^[+-]\d+[,.]?\d*\s*[mM]$")

_PROCESS_LABEL_RE = re.compile(
    r"^(Cooling\s*[Ww]ater|Steam|Condensate|Feed|Product|Waste|Inert|"
    r"Exhaust|Drain|Vent|Flare|Nitrogen|CW|CS|BFW|LP|MP|HP|"
    r"Vacuum|Overhead|Bottoms|Reflux|Reboiler|Inlet|Outlet|"
    r"Ammonia|Phosphoric|Acid|Phosphate|Diammonium)$",
    re.I,
)

_SHORT_VALVE_TAG_RE = re.compile(r"^V[- _]?\d{1,2}[A-Z]?$", re.I)


def classify_ocr_words(words: List[OcrWord]) -> Dict[str, List[OcrWord]]:
    """Classify OCR words into P&ID evidence categories."""
    categories: Dict[str, List[OcrWord]] = {
        "instrument": [], "equipment": [], "line_label": [],
        "elevation": [], "process_label": [], "other": [],
    }
    for w in words:
        txt = w.text.strip()
        # Clean common OCR artifacts from P&ID special characters.
        # Keep hyphen/underscore because V-1 style tags need that context.
        txt_clean = re.sub(r"[()[\]{}<>+=*#@!]", "", txt).strip()
        txt_tag = txt_clean.upper().replace(" ", "").replace("_", "-")
        if not txt_clean or len(txt_clean) < 2:
            categories["other"].append(w)
            continue

        if _SHORT_VALVE_TAG_RE.match(txt_tag):
            categories["equipment"].append(OcrWord(txt_tag, w.conf, w.x, w.y, w.w, w.h))
        elif _INSTRUMENT_RE.match(txt_clean):
            categories["instrument"].append(w)
        elif _ELEVATION_RE.match(txt_clean):
            categories["elevation"].append(w)
        elif _LINE_RE.match(txt_clean):
            categories["line_label"].append(w)
        elif _PROCESS_LABEL_RE.match(txt_clean):
            categories["process_label"].append(w)
        else:
            # Try equipment patterns
            matched_equip = False
            for pat in _EQUIPMENT_RE.values():
                if pat.match(txt_clean):
                    # Avoid single-letter false positives (C, P, etc.)
                    if len(txt_clean) >= 3 or re.search(r"\d", txt_clean):
                        w_copy = OcrWord(txt_tag, w.conf, w.x, w.y, w.w, w.h)
                        categories["equipment"].append(w_copy)
                        matched_equip = True
                        break
            if not matched_equip:
                categories["other"].append(w)

    return categories


def get_subtype(tag: str) -> str:
    if _SHORT_VALVE_TAG_RE.match(tag):
        return "valve_manual"
    for subtype, pat in _EQUIPMENT_RE.items():
        if pat.match(tag):
            return subtype
    return "unknown"


# ─────────────────────────────────────────────────────────────────
# STAGE 1D: PIPE / LINE DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_pipes(gray: np.ndarray, img_w: int, img_h: int) -> List[Pipe]:
    """
    Detect horizontal and vertical pipe runs using probabilistic Hough transform.
    P&ID pipes are nearly perfectly H/V so a strict angle tolerance works well.
    """
    preprocessed = preprocess_for_lines(gray)
    edges = cv2.Canny(preprocessed, 30, 100, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,           # votes required
        minLineLength=35,       # px — shorter than this is noise
        maxLineGap=20,          # px — close gaps to join broken lines
    )
    pipes: List[Pipe] = []
    if lines is None:
        return pipes

    h_segs, v_segs = [], []
    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        length = int(math.hypot(dx, dy))
        if dy <= 8 and dx > 35:       # horizontal (±8px tolerance)
            h_segs.append((x1, y1, x2, y2, length))
        elif dx <= 8 and dy > 35:     # vertical
            v_segs.append((x1, y1, x2, y2, length))

    # Merge close parallel segments
    def merge_segs(segs, orientation):
        merged = []
        segs_sorted = sorted(segs, key=lambda s: (s[1] if orientation == "h" else s[0], s[0]))
        for seg in segs_sorted:
            x1, y1, x2, y2, _ = seg
            absorbed = False
            for m in merged:
                coord_same = abs((y1 if orientation == "h" else x1) -
                                 (m[1] if orientation == "h" else m[0])) < 6
                overlapping = (min(x2, m[2]) > max(x1, m[0])) if orientation == "h" else \
                              (min(y2, m[3]) > max(y1, m[1]))
                if coord_same and overlapping:
                    # Extend
                    m[0] = min(m[0], x1); m[1] = min(m[1], y1)
                    m[2] = max(m[2], x2); m[3] = max(m[3], y2)
                    m[4] = int(math.hypot(m[2]-m[0], m[3]-m[1]))
                    absorbed = True; break
            if not absorbed:
                merged.append(list(seg))
        return merged

    for idx, seg in enumerate(merge_segs(h_segs, "h")):
        x1, y1, x2, y2, length = seg
        pipes.append(Pipe(f"H-{idx+1:03d}", "horizontal", x1, y1, x2, y2, length))
    for idx, seg in enumerate(merge_segs(v_segs, "v")):
        x1, y1, x2, y2, length = seg
        pipes.append(Pipe(f"V-{idx+1:03d}", "vertical", x1, y1, x2, y2, length))

    return pipes


# ─────────────────────────────────────────────────────────────────
# STAGE 1E: SYMBOL BOX DETECTION (for LLM hint regions)
# ─────────────────────────────────────────────────────────────────

def detect_symbol_regions(gray: np.ndarray, img_w: int, img_h: int) -> List[BBox]:
    """
    Detect closed symbol regions (circles, squares, diamonds)
    that represent instruments and equipment on P&IDs.
    Strategy:
      1. Threshold to binary (dark on white)
      2. Morphological close to connect dashed borders
      3. Find external contours
      4. Filter by size and aspect ratio for typical P&ID symbols
    """
    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, min(int(ret * 0.85), 200), 255, cv2.THRESH_BINARY_INV)

    # Close gaps in symbol outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # P&ID symbols are typically 20–200px in each dimension
    min_dim = max(15, min(img_w, img_h) * 0.004)
    max_dim_w = img_w * 0.06
    max_dim_h = img_h * 0.06

    boxes: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_dim ** 2:
            continue
        if w > max_dim_w or h > max_dim_h:
            continue
        aspect = w / max(h, 1)
        if aspect > 8 or aspect < 0.125:
            continue
        # Fill ratio — symbols are often outlines, not solid fills
        fill = area / (w * h) if w * h > 0 else 0
        if fill < 0.05:
            continue
        boxes.append(BBox(x, y, w, h))

    # Deduplicate by IoU
    boxes.sort(key=lambda b: -b.area)
    kept: List[BBox] = []
    for b in boxes:
        overlap = False
        for k in kept:
            ix = max(0, min(b.x + b.w, k.x + k.w) - max(b.x, k.x))
            iy = max(0, min(b.y + b.h, k.y + k.h) - max(b.y, k.y))
            if ix * iy > 0.4 * min(b.area, k.area):
                overlap = True; break
        if not overlap:
            kept.append(b)

    return kept[:300]


# ─────────────────────────────────────────────────────────────────
# STAGE 1F: ASSOCIATION — Link tags to symbols and pipes
# ─────────────────────────────────────────────────────────────────

def _dist_to_box(px: int, py: int, box: BBox) -> float:
    dx = max(box.x - px, 0, px - (box.x + box.w))
    dy = max(box.y - py, 0, py - (box.y + box.h))
    return math.hypot(dx, dy)


def associate_tags_to_symbols(
    words_by_category: Dict[str, List[OcrWord]],
    symbol_boxes: List[BBox],
    pipes: List[Pipe],
    img_w: int,
    img_h: int,
) -> Tuple[List[TaggedElement], List[TaggedElement], List[TaggedElement]]:
    """
    Associate instrument and equipment tags to their nearest symbol box.
    If no symbol is nearby, create an element with position from the OCR word itself.
    Also associate pipe segments to tags near their endpoints.
    """
    proximity_px = max(80, min(img_w, img_h) * 0.04)

    def find_nearest_box(wx: int, wy: int) -> Optional[BBox]:
        best_box, best_d = None, proximity_px
        for box in symbol_boxes:
            d = _dist_to_box(wx, wy, box)
            if d < best_d:
                best_box, best_d = box, d
        return best_box

    def nearby_tags(cx: int, cy: int, all_words: List[OcrWord], radius: int = 120) -> List[str]:
        found = []
        for w in all_words:
            if math.hypot(w.x - cx, w.y - cy) < radius:
                found.append(w.text.upper())
        return found

    all_words_flat = [w for cat in words_by_category.values() for w in cat]

    # ── Instruments
    instruments: List[TaggedElement] = []
    seen_instrument_tags: set = set()
    for w in words_by_category.get("instrument", []):
        tag = w.text.upper()
        if tag in seen_instrument_tags:
            continue
        seen_instrument_tags.add(tag)
        box = find_nearest_box(w.x, w.y)
        elem_box = box if box else BBox(w.x, w.y, w.w, w.h)
        nearby = [t for t in nearby_tags(elem_box.cx, elem_box.cy, all_words_flat) if t != tag]
        instruments.append(TaggedElement(
            tag=tag, element_type="instrument",
            subtype=_instrument_subtype(tag),
            bbox=elem_box,
            nearby_tags=nearby[:6],
            confidence="high" if box else "medium",
        ))

    # ── Equipment
    equipment: List[TaggedElement] = []
    seen_equip_tags: set = set()
    for w in words_by_category.get("equipment", []):
        tag = w.text.upper()
        if tag in seen_equip_tags:
            continue
        seen_equip_tags.add(tag)
        box = find_nearest_box(w.x, w.y)
        elem_box = box if box else BBox(w.x, w.y, w.w, w.h)
        nearby = [t for t in nearby_tags(elem_box.cx, elem_box.cy, all_words_flat) if t != tag]
        subtype = get_subtype(tag)
        equipment.append(TaggedElement(
            tag=tag, element_type="equipment",
            subtype=subtype,
            bbox=elem_box,
            nearby_tags=nearby[:6],
            confidence="high" if box else "medium",
        ))

    # ── Pipe connectivity: mark which tags are near pipe endpoints
    for pipe in pipes[:200]:
        endpoints = [(pipe.x1, pipe.y1), (pipe.x2, pipe.y2)]
        for ex, ey in endpoints:
            for elem in instruments + equipment:
                if elem.bbox and _dist_to_box(ex, ey, elem.bbox) < proximity_px:
                    if elem.tag not in pipe.connected_tags:
                        pipe.connected_tags.append(elem.tag)

    # ── Valves (from equipment with valve subtypes)
    valves = [e for e in equipment if e.subtype == "valve_manual"]
    equip_non_valve = [e for e in equipment if e.subtype != "valve_manual"]

    return equip_non_valve, instruments, valves


def _instrument_subtype(tag: str) -> str:
    t = tag.upper()
    if re.match(r"^(FI|FC|FT|FRC|FR|FIC|FAL|FAH|FFI)", t): return "flow"
    if re.match(r"^(PI|PC|PT|PRC|PIC|PSV|PRV|PSH|PSL|PAH|PAL|PDI)", t): return "pressure"
    if re.match(r"^(TI|TC|TT|TRC|TIC|TAH|TAL|TSH|TSL)", t): return "temperature"
    if re.match(r"^(LI|LC|LT|LRC|LIC|LAH|LAL|LSH|LSL|LIA)", t): return "level"
    if re.match(r"^(XV|FV|PV|TV|LV|HV)", t): return "control_valve"
    if re.match(r"^ESD", t): return "emergency_shutdown"
    if re.match(r"^AN", t): return "signal_node"
    return "general"


# ─────────────────────────────────────────────────────────────────
# STAGE 1G: NODE BOUNDARY INFERENCE
# ─────────────────────────────────────────────────────────────────

def infer_nodes(
    equipment: List[TaggedElement],
    instruments: List[TaggedElement],
    img_w: int,
    img_h: int,
) -> List[Dict]:
    """
    Infer candidate HAZOP nodes by spatial clustering of equipment.
    Equipment items that are close together and connected by pipes
    likely belong to the same node.
    Simple approach: horizontal band clustering (left→right process flow).
    """
    if not equipment:
        return [{"node_id": "N-001", "node_name": "Overall process",
                 "boundary": "Entire P&ID", "design_intent": "Operate described process",
                 "equipment_tags": [], "instrument_tags": []}]

    # Sort equipment left to right by x position
    equip_sorted = sorted(equipment, key=lambda e: e.bbox.cx if e.bbox else 0)

    # Group into columns / clusters
    nodes: List[Dict] = []
    cluster: List[TaggedElement] = []
    cluster_max_x = 0
    gap_threshold = img_w * 0.15  # 15% of width gap = new node

    for elem in equip_sorted:
        ex = elem.bbox.cx if elem.bbox else 0
        if cluster and ex - cluster_max_x > gap_threshold:
            nodes.append(_make_node(nodes, cluster, instruments, img_w, img_h))
            cluster = []
        cluster.append(elem)
        cluster_max_x = max(cluster_max_x, ex)

    if cluster:
        nodes.append(_make_node(nodes, cluster, instruments, img_w, img_h))

    return nodes


def _make_node(
    existing: List[Dict],
    cluster: List[TaggedElement],
    instruments: List[TaggedElement],
    img_w: int, img_h: int,
) -> Dict:
    node_num = len(existing) + 1
    equip_tags = [e.tag for e in cluster]
    # Find instruments near any equipment in the cluster
    cluster_xs = [e.bbox.cx for e in cluster if e.bbox]
    cluster_ys = [e.bbox.cy for e in cluster if e.bbox]
    if cluster_xs:
        cx_min, cx_max = min(cluster_xs) - 200, max(cluster_xs) + 200
        cy_min, cy_max = min(cluster_ys) - 300, max(cluster_ys) + 300
        nearby_instr = [
            i.tag for i in instruments
            if i.bbox and cx_min <= i.bbox.cx <= cx_max and cy_min <= i.bbox.cy <= cy_max
        ]
    else:
        nearby_instr = []
    # Infer subtype from dominant equipment
    subtypes = [e.subtype for e in cluster]
    dominant = max(set(subtypes), key=subtypes.count) if subtypes else "process"
    name_map = {
        "heat_exchanger": "Heat Exchange Section",
        "vessel": "Storage / Buffer Section",
        "pump": "Pumping Section",
        "column": "Separation / Distillation Section",
        "reactor": "Reaction Section",
        "filter": "Filtration Section",
    }
    node_name = name_map.get(dominant, f"Process Section {node_num}")
    return {
        "node_id": f"N-{node_num:03d}",
        "node_name": node_name,
        "boundary": f"Equipment cluster: {', '.join(equip_tags[:6])}",
        "design_intent": f"Normal operation of {node_name.lower()} per design intent.",
        "equipment_tags": equip_tags,
        "instrument_tags": nearby_instr[:12],
        "confidence": "cv_inferred",
    }


# ─────────────────────────────────────────────────────────────────
# STAGE 2: VISION LLM OVERLAY
# ─────────────────────────────────────────────────────────────────

def build_vision_prompt(cv_result: Dict, text_context: str) -> str:
    """
    Build the vision LLM prompt using structured CV output as context.
    This is far better than asking the LLM to read the diagram cold —
    the CV gives it verified tag positions so it can focus on
    visible P&ID object and connection verification rather than text reading.
    """
    equip = cv_result.get("equipment", [])
    instruments = cv_result.get("instruments", [])
    pipes = cv_result.get("pipes", {})
    nodes = cv_result.get("candidate_nodes", [])

    return f"""
You are the P&ID Image Parser Agent for a HAZOP study system.
A computer vision pipeline has already read this P&ID and extracted:
- {len(equip)} equipment items: {[e['tag'] for e in equip[:20]]}
- {len(instruments)} instruments: {[i['tag'] for i in instruments[:20]]}
- {pipes.get('horizontal_count', 0)} horizontal + {pipes.get('vertical_count', 0)} vertical pipe segments
- {len(nodes)} candidate HAZOP nodes inferred from equipment clustering

Your tasks:
1. VERIFY and CORRECT the CV-extracted tags only when the corrected tag is visible in the image.
2. ADD equipment, instruments, valves, relief devices, line tags, utilities, drains, vents, and endpoints only when they are visible in the diagram.
3. CLASSIFY each visible equipment item by function (column, furnace, pump, reboiler, condenser, receiver, etc.).
4. IDENTIFY visible piping-line relationships using visible lines, arrows, line labels, inline valves, and equipment sequence.
    5. SUGGEST HAZOP node boundaries only from visible connected process sections.
    6. IDENTIFY process streams and utility streams, but do not let stream summaries bypass inline valves or equipment.

    STRICT EVIDENCE RULES:
    - Use the supporting text only to clarify a visible object, visible label, visible stream, or visible operating intent.
    - Do not create objects, tags, line IDs, instruments, safeguards, connections, or node boundaries from text alone.
- Preserve visible P&ID tags and line labels exactly, including simple labels such as L0, L1, L7, L12, H1, H2, H3, C1, P1, P2, P3, V0, V1, V10, V12, LIC, TIC, and FIC when they are visible.
- Do not invent generic labels such as TK-1, R-101, STREAM-001, source, destination, reactor, or storage unless those exact labels are visible.
- If a visible item is readable by function but not by tag, use UNKNOWN for the tag and explain the uncertainty.
    - For complex diagrams, keep separate process, utility, vent, residue, product, recycle, bypass, and control sections instead of collapsing the whole unit into one overall section.

OUTPUT FORMAT — return exactly this compact JSON and nothing else:
{{
  "verified_equipment": [
    {{"tag": "visible tag or empty string", "type": "visible equipment type", "function": "visible label or function", "connects_to": ["visible connected tag or label"]}}
  ],
  "verified_instruments": [
    {{"tag": "visible instrument tag", "type": "visible instrument type", "controls": "visible controlled item or empty string", "associated_equipment": "visible associated equipment or line"}}
  ],
  "ocr_corrections": [
    {{"original": "visible OCR text", "corrected": "visible corrected text", "reason": "visible image basis only"}}
  ],
  "process_streams": [
    {{"name": "visible stream/service label", "from": "visible upstream object", "to": "visible downstream object", "material": "visible material label"}}
  ],
  "revised_nodes": [
    {{
      "node_id": "N-001",
      "node_name": "visible section name",
      "boundary": "visible start object to visible end object",
      "design_intent": "normal function evident from visible labels, arrows, line evidence, and equipment sequence",
      "equipment_tags": ["visible equipment tags or labels"],
      "instrument_tags": ["visible instrument/control tags"],
      "parameters": ["flow", "temperature", "pressure"]
    }}
  ],
  "missed_elements": ["any equipment or instruments visible but not in CV list"],
  "uncertainties": ["anything ambiguous or unreadable"]
}}

    LINE AND STREAM RULES:
    - Every listed line or stream must be supported by a visible pipe, arrow, label, inline valve, equipment connection, utility, vent, drain, recycle, bypass, relief, or transfer endpoint.
    - Lines and streams must follow the physical direction shown by arrows or the clearest visible line direction.
    - If the image shows equipment to valve to equipment, keep the valve in the parsed line or stream sequence.
    - Do not list control, measurement, located-on, interlock, or protection metadata as physical flow paths. Capture instruments and controls only in verified_instruments.
    - Do not create one generic overall-process node when the drawing shows multiple connected sections.
    - Limit output to visible items only. If the diagram is complex, prefer the most important connected process sections over verbose explanation.
    - Maximum output counts: 60 equipment items, 80 instruments/controls, 80 valves/relief devices, 80 lines/streams, and 20 HAZOP node candidates.

CV PRE-PARSE OUTPUT (verified by computer vision):
{json.dumps(json_safe(cv_result), indent=2)[:8000]}

SUPPORTING TEXT CONTEXT:
{text_context[:4000]}
""".strip()


def build_vision_system_prompt() -> str:
    return (
        "You are the P&ID Image Parser Agent for an automated HAZOP study. "
        "You are given a P&ID diagram and structured computer-vision pre-parse results. "
        "Your response MUST be a single valid JSON object. "
        "Do NOT include any text before or after the JSON. Do NOT use markdown fences. "
        "Start with { and end with }. "
        "Do not invent tags that are not visible. Use 'UNKNOWN' for unreadable text."
    )


def build_compact_vision_retry_prompt(cv_result: Dict, text_context: str, previous_error: str) -> str:
    equip = cv_result.get("equipment", [])
    instruments = cv_result.get("instruments", [])
    pipes = cv_result.get("pipes", {})
    return f"""
Your previous P&ID parse response was not accepted because it was not complete valid JSON:
{previous_error[:800]}

Return a shorter JSON object only. Do not include prose or markdown.

Required schema:
{{
  "parsed_pid_summary": "one concise sentence",
  "verified_equipment": [
    {{"tag": "visible tag or empty string", "type": "equipment type", "function": "visible label/function", "connects_to": ["visible connected object"]}}
  ],
  "verified_instruments": [
    {{"tag": "visible tag", "type": "instrument/control type", "controls": "visible controlled item", "associated_equipment": "visible associated item"}}
  ],
  "process_streams": [
    {{"name": "visible stream/service label", "from": "visible upstream object", "to": "visible downstream object", "material": "visible material label"}}
  ],
  "revised_nodes": [
    {{"node_id": "N-001", "node_name": "visible section name", "boundary": "visible start to visible end", "design_intent": "visible function and line evidence", "equipment_tags": ["visible item"], "instrument_tags": ["visible control"], "parameters": ["flow"]}}
  ],
  "uncertainties": ["ambiguous visible items only"]
}}

Strict rules:
- Use only visible image evidence. Supporting text can clarify visible labels but cannot create objects or connections.
    - Do not invent tags, line IDs, physical connections, or HAZOP node candidates.
    - Preserve visible tags exactly.
    - Put only visible process, utility, vent, drain, relief, recycle, or bypass paths in process_streams or revised_nodes.
    - Parsed line relationships must connect physically adjacent visible objects in sequence.
    - Do not include raw OCR words or long explanations.
    - Cap output at 40 equipment items, 60 instruments/controls, 60 valves/relief devices, 80 lines/streams, and 16 node candidates.

CV summary:
- equipment candidates: {[e.get('tag', '') for e in equip[:30]]}
- instrument candidates: {[i.get('tag', '') for i in instruments[:30]]}
- pipe segment summary: {json.dumps(pipes)[:1000]}

Supporting context, for label clarification only:
{text_context[:2500]}
""".strip()


# ─────────────────────────────────────────────────────────────────
# STAGE 3: MERGE CV + LLM → FINAL STRUCTURED OUTPUT
# ─────────────────────────────────────────────────────────────────

def merge_cv_and_llm(
    cv_parse: PIDParse,
    llm_result: Optional[Dict],
) -> Dict:
    """
    Merge CV detections with LLM corrections into a single structured dict.
    CV provides: approximate positions and pipe-line evidence.
    LLM provides: object labels, corrections, connectivity, node boundaries.
    """
    cv_dict = cv_parse.to_dict()

    if not llm_result:
        # No LLM — return CV only with a warning
        cv_dict["uncertainties"].append(
            "Vision LLM not available. Results are CV-only: positions are accurate but "
            "equipment function and stream names are inferred from tags only."
        )
        cv_dict["parse_method"] = "cv_only"
        return cv_dict

    # Apply OCR corrections to equipment and instrument tags
    corrections = {c["original"].upper(): c["corrected"].upper()
                   for c in llm_result.get("ocr_corrections", [])
                   if "original" in c and "corrected" in c}

    def apply_corrections(items: List[Dict]) -> List[Dict]:
        for item in items:
            tag = item.get("tag", "")
            if tag.upper() in corrections:
                item["tag"] = corrections[tag.upper()]
                item["tag_corrected"] = True
        return items

    # Merge equipment
    cv_equip_tags = {e["tag"].upper() for e in cv_dict["equipment"]}
    merged_equip = apply_corrections(cv_dict["equipment"])
    merged_valves = apply_corrections(cv_dict["valves"])

    # Add LLM-found equipment that CV missed
    for e in llm_result.get("verified_equipment", []):
        tag = str(e.get("tag", "")).upper()
        if tag and tag not in cv_equip_tags and tag != "UNKNOWN":
            if get_subtype(tag) == "valve_manual":
                merged_valves.append({
                    "tag": tag,
                    "type": "valve",
                    "subtype": "valve_manual",
                    "function": e.get("function", ""),
                    "connects_to": e.get("connects_to", []),
                    "description": e.get("function", ""),
                    "confidence": "llm_added",
                    "nearby_tags": [],
                })
                continue
            merged_equip.append({
                "tag": tag,
                "type": e.get("type", "equipment"),
                "subtype": e.get("type", "unknown"),
                "function": e.get("function", ""),
                "connects_to": e.get("connects_to", []),
                "description": e.get("function", ""),
                "confidence": "llm_added",
                "nearby_tags": [],
            })

    # Merge instruments
    cv_instr_tags = {i["tag"].upper() for i in cv_dict["instruments"]}
    merged_instr = apply_corrections(cv_dict["instruments"])
    for i in llm_result.get("verified_instruments", []):
        tag = str(i.get("tag", "")).upper()
        if tag and tag not in cv_instr_tags and tag != "UNKNOWN":
            merged_instr.append({
                "tag": tag,
                "type": "instrument",
                "subtype": i.get("type", "unknown"),
                "description": i.get("controls", ""),
                "associated_equipment": i.get("associated_equipment", ""),
                "confidence": "llm_added",
                "nearby_tags": [],
            })

    # Use LLM nodes if available (better object and connection verification)
    nodes = llm_result.get("revised_nodes") or cv_dict["candidate_nodes"]

    return {
        "image_size": cv_dict["image_size"],
        "equipment": merged_equip,
        "instruments": merged_instr,
        "valves": merged_valves,
        "pipes": cv_dict["pipes"],
        "process_streams": llm_result.get("process_streams", []),
        "candidate_nodes": nodes,
        "elevations": cv_dict["elevations"],
        "labels": cv_dict["labels"],
        "uncertainties": cv_dict["uncertainties"] + llm_result.get("uncertainties", []),
        "ocr_corrections": llm_result.get("ocr_corrections", []),
        "missed_elements_llm": llm_result.get("missed_elements", []),
        "cv_summary": cv_dict["cv_summary"],
        "parse_method": "cv+llm" if llm_result else "cv_only",
    }


# ─────────────────────────────────────────────────────────────────
# PUBLIC API — called from hazop_backend.py
# ─────────────────────────────────────────────────────────────────

def run_cv_pipeline(data_url: str) -> PIDParse:
    """
    Run Stage 1 (CV-only) on a single P&ID image data URL.
    Returns a PIDParse object.
    """
    _, gray, img_w, img_h = load_image_from_data_url(data_url)

    # OCR
    ocr_words = run_ocr(gray, img_w, img_h)
    words_by_category = classify_ocr_words(ocr_words)

    # Pipe detection
    pipes = detect_pipes(gray, img_w, img_h)

    # Symbol detection
    symbol_boxes = detect_symbol_regions(gray, img_w, img_h)

    # Associate tags to symbols
    equipment, instruments, valves = associate_tags_to_symbols(
        words_by_category, symbol_boxes, pipes, img_w, img_h
    )

    # Node inference
    nodes = infer_nodes(equipment, instruments, img_w, img_h)

    # Elevations
    elevations = [
        {"label": w.text, "x": w.x, "y": w.y}
        for w in words_by_category.get("elevation", [])
    ]

    # Process labels
    labels = sorted(set(w.text for w in words_by_category.get("process_label", [])))

    parse = PIDParse(
        image_width=img_w,
        image_height=img_h,
        equipment=equipment,
        instruments=instruments,
        valves=valves,
        pipes=pipes[:400],
        raw_ocr_words=ocr_words[:200],
        candidate_nodes=nodes,
        elevations=elevations,
        labels=labels,
        uncertainties=[],
        cv_summary={
            "ocr_words_total": len(ocr_words),
            "equipment_found": len(equipment),
            "instruments_found": len(instruments),
            "valves_found": len(valves),
            "pipes_detected": len(pipes),
            "symbol_regions_detected": len(symbol_boxes),
            "nodes_inferred": len(nodes),
        },
    )

    if len(equipment) == 0 and len(instruments) == 0:
        parse.uncertainties.append(
            "CV pipeline found no tagged equipment or instruments. "
            "The image may be low-resolution, heavily compressed, or use non-standard tag formats. "
            "Vision LLM will attempt direct reading."
        )
    if len(ocr_words) == 0:
        parse.uncertainties.append(
            "Local OCR returned no words. Tesseract may be missing or the drawing text may be unreadable; "
            "pipe detection and vision LLM parsing will still be used."
        )

    return parse


def parse_pid(
    images: List[str],
    run_vision_model_fn=None,
    vision_agent_id: str = "pid_image_parser",
    agent_config: Dict = None,
    api_keys: Dict = None,
    text_context: str = "",
) -> Dict:
    """
    Full three-stage P&ID parsing pipeline.

    Args:
        images: list of data:image/...;base64,... strings
        run_vision_model_fn: callable(agent_id, config, keys, sys_prompt, user_prompt, images) -> str
                             Pass the run_vision_model function from hazop_backend.
                             If None, Stage 2 is skipped.
        vision_agent_id: agent ID for the vision model
        agent_config: agent config dict
        api_keys: API keys dict
        text_context: extracted text from process description, P&ID notes, etc.

    Returns:
        merged dict suitable for document_context() injection
    """
    if not images:
        return {"parse_method": "none", "uncertainties": ["No images provided"]}

    # Stage 1: CV pipeline on all images
    all_equipment: List[TaggedElement] = []
    all_instruments: List[TaggedElement] = []
    all_valves: List[TaggedElement] = []
    all_pipes: List[Pipe] = []
    all_nodes: List[Dict] = []
    all_elevations: List[Dict] = []
    all_labels: List[str] = []
    all_ocr_words: List[OcrWord] = []
    all_uncertainties: List[str] = []
    img_sizes: List[List[int]] = []
    cv_summaries: List[Dict] = []

    for idx, image_data_url in enumerate(images[:6]):
        try:
            cv_parse = run_cv_pipeline(image_data_url)
            all_equipment.extend(cv_parse.equipment)
            all_instruments.extend(cv_parse.instruments)
            all_valves.extend(cv_parse.valves)
            all_pipes.extend(cv_parse.pipes)
            all_nodes.extend(cv_parse.candidate_nodes)
            all_elevations.extend(cv_parse.elevations)
            all_labels.extend(cv_parse.labels)
            all_ocr_words.extend(cv_parse.raw_ocr_words)
            all_uncertainties.extend(cv_parse.uncertainties)
            img_sizes.append([cv_parse.image_width, cv_parse.image_height])
            cv_summaries.append(cv_parse.cv_summary)
        except Exception as err:
            all_uncertainties.append(f"Image {idx+1} CV pipeline error: {err}")

    # Deduplicate tags across images
    def dedup(items, key="tag"):
        seen, out = set(), []
        for item in items:
            k = (item.tag if hasattr(item, "tag") else item.get("tag", "")).upper()
            if k and k not in seen:
                seen.add(k); out.append(item)
        return out

    all_equipment = dedup(all_equipment)
    all_instruments = dedup(all_instruments)
    all_valves = dedup(all_valves)

    # Renumber nodes
    for i, node in enumerate(all_nodes):
        node["node_id"] = f"N-{i+1:03d}"

    # Build combined CV dict for LLM
    img_w = img_sizes[0][0] if img_sizes else 0
    img_h = img_sizes[0][1] if img_sizes else 0

    combined_cv = PIDParse(
        image_width=img_w, image_height=img_h,
        equipment=all_equipment, instruments=all_instruments,
        valves=all_valves, pipes=all_pipes[:200],
        raw_ocr_words=all_ocr_words[:100],
        candidate_nodes=all_nodes[:10],
        elevations=all_elevations[:20],
        labels=list(set(all_labels))[:30],
        uncertainties=all_uncertainties,
        cv_summary={"images": len(images), "per_image": cv_summaries},
    )
    cv_dict = combined_cv.to_dict()

    # Stage 2: Vision LLM overlay (if available)
    llm_result: Optional[Dict] = None
    if run_vision_model_fn is not None:
        raw_response = ""
        try:
            from hazop_backend import extract_json_object
            vision_prompt = build_vision_prompt(cv_dict, text_context)
            vision_sys_prompt = build_vision_system_prompt()
            raw_response = run_vision_model_fn(
                vision_agent_id,
                agent_config or {},
                api_keys or {},
                vision_sys_prompt,
                vision_prompt,
                images,
            )
            try:
                llm_result = extract_json_object(raw_response)
                llm_result["llm_raw_text"] = raw_response
            except Exception as parse_err:
                retry_prompt = build_compact_vision_retry_prompt(cv_dict, text_context, str(parse_err))
                retry_raw = run_vision_model_fn(
                    vision_agent_id,
                    agent_config or {},
                    api_keys or {},
                    vision_sys_prompt,
                    retry_prompt,
                    images,
                )
                llm_result = extract_json_object(retry_raw)
                llm_result["llm_raw_text"] = retry_raw
                all_uncertainties.append(
                    f"Vision LLM required compact JSON retry after parse error: {parse_err}"
                )
        except Exception as err:
            all_uncertainties.append(
                f"Vision LLM failed — using CV-only results. Error: {err}"
            )

    # Stage 3: Merge
    return merge_cv_and_llm(combined_cv, llm_result)


# ─────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pid_parser.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        cli_raw = base64.b64encode(f.read()).decode()
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    cli_data_url = f"data:{mime};base64,{cli_raw}"

    result = parse_pid([cli_data_url])
    print(json.dumps(result, indent=2, default=str))
