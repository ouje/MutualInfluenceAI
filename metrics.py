# metrics.py
# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2025 Vaclav Oujezsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import re
import json
from typing import Set, Optional

# ---------- JSON helpers ----------

def safe_json_loads(s: str) -> dict | None:
    """Safe JSON loader: returns dict or None."""
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_features_from_json(s: str) -> Set[str]:
    """
    Extracts a normalized set of feature names from a JSON string:
    { "features": ["name1", "name2", ...] }
    """
    d = safe_json_loads(s)
    if not isinstance(d, dict):
        return set()
    feats = d.get("features", [])
    if not isinstance(feats, list):
        return set()
    norm = lambda x: " ".join(str(x).lower().split())
    return {norm(x) for x in feats if isinstance(x, (str, int, float)) and str(x).strip()}

def critic_decision_from_json(s: str) -> str | None:
    """
    Parses critic decision from JSON:
    { "decision": "APPROVE" | "REVISE" }
    """
    d = safe_json_loads(s)
    if not isinstance(d, dict):
        return None
    dec = d.get("decision")
    if isinstance(dec, str):
        dec = dec.strip().upper()
        if dec in ("APPROVE", "REVISE"):
            return dec
    return None

def rounds_to_approval_json(critic_json_text: str) -> Optional[int]:
    """
    Convenience: returns 1 if {"decision":"APPROVE"}, else None.
    """
    return 1 if critic_decision_from_json(critic_json_text) == "APPROVE" else None

# ---------- Canonical overlap (token-based) ----------

_CANON_TAGS = {
    "duration", "bytes", "packets", "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol", "flags", "entropy", "dns", "http", "tls", "ja3", "user_agent",
    "flow_count", "rate", "iat", "window", "payload_len"
}

def canonical_overlap(text_a: str, text_b: str) -> Optional[float]:
    """
    Token-based overlap against a fixed vocabulary of network-flow concepts.
    Returns Jaccard(A,B) over canonical tokens, or None if both empty.
    """
    def to_tags(t: str):
        t = (t or "").lower()
        toks = set(re.findall(r"[a-z0-9_]+", t))
        return {tok for tok in toks if tok in _CANON_TAGS}
    A, B = to_tags(text_a), to_tags(text_b)
    if not A and not B:
        return None
    denom = len(A | B)
    return 0.0 if denom == 0 else len(A & B) / denom

# ---------- Generic text keypoints & metrics (legacy) ----------

def extract_keypoints(text: str) -> Set[str]:
    """
    Extracts bullet/numbered lines or sentence-like chunks as keypoints.
    Used by legacy Jaccard, not required when using JSON outputs.
    """
    if not text:
        return set()
    lines = [l.strip() for l in text.splitlines()]
    bullets = []
    for l in lines:
        if re.match(r"^\s*([-•*]|\d+\.)\s+", l):
            bullets.append(re.sub(r"^\s*([-•*]|\d+\.)\s+", "", l).strip())
    items = bullets or [s.strip() for s in re.split(r"[.;]\s+", text) if len(s.split()) >= 3]
    items = [re.sub(r"\s+", " ", s).strip().lower() for s in items if s.strip()]
    return set(items)

def jaccard(a: Set[str], b: Set[str]) -> Optional[float]:
    """Jaccard similarity with None for both-empty inputs."""
    if not a and not b:
        return None
    denom = len(a | b)
    return 0.0 if denom == 0 else len(a & b) / denom

def revision_depth(prev_points: Set[str], curr_points: Set[str], critic_text: str = "") -> int:
    """
    Legacy revision depth: newly introduced points + indicator of incorporating critic points.
    For JSON-based pipeline prefer set-difference on features directly.
    """
    new_pts = len(curr_points - prev_points)
    crit_pts = extract_keypoints(critic_text) if critic_text else set()
    inc = 1 if (crit_pts & curr_points) else 0
    return new_pts + inc

def rounds_to_approval(critic_text: str) -> Optional[int]:
    """
    Legacy regex-based APPROVE detector for free-form critic output.
    Prefer rounds_to_approval_json for JSON critic.
    """
    return 1 if isinstance(critic_text, str) and re.search(r"\bAPPROVE\b", critic_text.upper()) else None
