"""
Aggregate Gate0-G4 JSON reports under Phase2_outputs/gates/.
  python m3_audit_gates.py
"""
from __future__ import annotations

import json
from pathlib import Path

import m3_mineral_constants as C


def main():
    C.GATES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(C.GATES_DIR.glob("G*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"file": p.name, "pass": False, "error": str(exc)})
            continue
        rows.append({
            "file": p.name,
            "gate": d.get("gate"),
            "scene": d.get("scene"),
            "pass": d.get("pass"),
            "reasons": d.get("reasons", []),
        })
    summary = {"n": len(rows), "n_pass": sum(1 for r in rows if r.get("pass")), "gates": rows}
    out = C.GATES_DIR / "gates_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    return 0 if summary["n_pass"] == summary["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
