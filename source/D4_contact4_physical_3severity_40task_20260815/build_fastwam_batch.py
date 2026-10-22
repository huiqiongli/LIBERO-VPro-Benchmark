#!/usr/bin/env python3
import json
from pathlib import Path

here = Path(__file__).resolve().parent
source = json.loads((here / "manifest.json").read_text())
cases = []
for c in source["cases"]:
    cases.append({
        "case_id": f"{c['suite']}_t{c['task_id']:02d}__{c['case_id']}",
        "suite": c["suite"], "task_id": c["task_id"],
        "instruction": c["instruction"], "condition": "perturbed",
        "mechanism": c["mechanism"], "severity": c["severity"], "spec": c["spec_path"],
    })
out = {"schema_version": "d4_contact4_3severity_fastwam_batch.v1", "cases": cases}
(here / "fastwam_batch_manifest.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")
print(len(cases))
