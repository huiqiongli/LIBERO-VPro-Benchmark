#!/usr/bin/env python3
import json
from pathlib import Path

here = Path(__file__).resolve().parent
manifest = json.loads((here / "manifest.json").read_text())
rows = []
for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
    report = json.loads((here / "validation" / f"validation_report_{suite}.json").read_text())
    rows.extend(report["rows"])
summary = {
    "schema_version": "d4_contact4_3severity_validation.v1", "total": len(rows),
    "passed": sum(r["status"] == "passed" for r in rows),
    "failed": sum(r["status"] == "failed" for r in rows),
    "errors": sum(r["status"] == "error" for r in rows),
    "skipped_tasks": manifest["skipped_tasks"], "rows": rows,
}
(here / "validation" / "validation_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
print(json.dumps({k:summary[k] for k in ("total","passed","failed","errors")}))
if summary["total"] != len(manifest["cases"]) or summary["passed"] != summary["total"]:
    raise SystemExit(1)
