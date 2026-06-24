"""Train the distilled router heads on frozen MiniLM embeddings.

Embeds every (real + synthetic) message once with a frozen distilled sentence
encoder, then fits small sklearn linear heads for the categorical RouterDecision
fields. Fast on CPU, sample-efficient, and servable in ~20-40ms. The free-form
fields (search_query / track_titles) are intentionally NOT modelled — those turns
fall back to the LLM.

Run:  uv run python ml/router/train.py
Out:  ml/router/heads.joblib, labelmaps.json, eval_report.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
ENCODER = "all-MiniLM-L6-v2"
_CAT_FIELDS = ("intent", "tone", "engagement_mode")
_BOOL_FIELDS = ("needs_search", "needs_memory", "needs_music", "needs_artist_lookup")


def _load() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for fname in ("dataset.jsonl", "synthetic.jsonl"):
        p = HERE / fname
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = r["message"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def main() -> None:
    rows = _load()
    print(f"training rows (real + synthetic, deduped): {len(rows)}")
    messages = [r["message"] for r in rows]

    print(f"embedding with frozen {ENCODER} ...")
    encoder = SentenceTransformer(ENCODER)
    X = encoder.encode(messages, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    X = np.asarray(X, dtype=np.float32)

    # Stratify the split on intent so every class appears in train AND test.
    idx = np.arange(len(rows))
    tr, te = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=[rows[i]["intent"] for i in idx]
    )

    report_lines: list[str] = [f"rows={len(rows)}  encoder={ENCODER}  train={len(tr)}  test={len(te)}\n"]
    heads: dict = {"encoder": ENCODER}
    labelmaps: dict = {}

    # ── Categorical heads (intent / tone / engagement) ────────────────────────
    for field in _CAT_FIELDS:
        le = LabelEncoder()
        y = le.fit_transform([r[field] for r in rows])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
        clf.fit(X[tr], y[tr])
        acc = accuracy_score(y[te], clf.predict(X[te]))
        heads[field] = clf
        labelmaps[field] = le.classes_.tolist()
        report_lines.append(f"{field:16} accuracy: {acc:.3f}")
        if field == "intent":
            report_lines.append("\nintent classification report (held-out):\n")
            report_lines.append(classification_report(
                y[te], clf.predict(X[te]), target_names=le.classes_, zero_division=0
            ))
            cm = confusion_matrix(y[te], clf.predict(X[te]))
            report_lines.append("confusion matrix (rows=true, cols=pred):")
            report_lines.append("labels: " + ", ".join(le.classes_))
            report_lines.append(np.array2string(cm))
            report_lines.append("")
        heads[f"_le_{field}"] = le

    # ── Binary needs_* heads ──────────────────────────────────────────────────
    report_lines.append("\nneeds_* (binary) accuracy:")
    for field in _BOOL_FIELDS:
        y = np.array([1 if r[field] else 0 for r in rows])
        if len(set(y[tr])) < 2:  # degenerate (all one class) — store a constant
            heads[field] = int(y[tr][0]) if len(y[tr]) else 0
            report_lines.append(f"  {field:22} constant={heads[field]}")
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
        clf.fit(X[tr], y[tr])
        acc = accuracy_score(y[te], clf.predict(X[te]))
        heads[field] = clf
        report_lines.append(f"  {field:22} accuracy: {acc:.3f}")

    joblib.dump(heads, HERE / "heads.joblib")
    (HERE / "labelmaps.json").write_text(json.dumps(labelmaps, indent=2), encoding="utf-8")
    report = "\n".join(str(x) for x in report_lines)
    (HERE / "eval_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nsaved heads.joblib ({(HERE / 'heads.joblib').stat().st_size // 1024} KB), labelmaps.json, eval_report.txt")


if __name__ == "__main__":
    main()
