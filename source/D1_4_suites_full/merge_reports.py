#!/usr/bin/env python3
"""Merge per-suite D1 validation reports and build cross-suite preview indexes."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
VALIDATION = HERE / "validation"
PREVIEWS = HERE / "previews"
SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
CONDITIONS = [
    "same_category",
    "same_color",
    "same_shape",
    "same_affordance",
    "same_position",
    "trajectory",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def stack_images(paths: list[Path], labels: list[str], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    header = 34
    height = sum(image.height + header for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, image in zip(labels, images):
        draw.text((8, y + 6), label, fill="black", font=font(18))
        canvas.paste(image, (0, y + header))
        y += image.height + header
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    for image in images:
        image.close()


def save_grid(
    paths: list[Path],
    labels: list[str],
    output: Path,
    *,
    columns: int,
) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    header = 32
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (tile_width * columns, (tile_height + header) * rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + header)
        draw.text((x + 7, y + 6), label, fill="black", font=font(16))
        canvas.paste(image, (x, y + header))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    for image in images:
        image.close()


def main() -> None:
    reports = {
        suite: load_json(VALIDATION / f"{suite}_validation.json")
        for suite in SUITES
    }
    root = {
        "schema_version": "d1_4_suites_validation.v1",
        "model_evaluation": "not_run_by_user_request",
        "suites": reports,
        "case_count": sum(report["case_count"] for report in reports.values()),
        "passed_count": sum(report["passed_count"] for report in reports.values()),
    }
    root["all_passed"] = root["passed_count"] == root["case_count"]
    write_json(root, VALIDATION / "validation_report.json")

    # Selected-case reruns intentionally overwrite individual images only.
    # Rebuild every task and suite contact sheet from those final case images so
    # no partial rerun can leave a one-case overview in place.
    for suite in SUITES:
        for task_id in range(10):
            task_dir = PREVIEWS / suite / f"task_{task_id:02d}"
            for view in ("agentview", "wrist"):
                raw_paths = [
                    task_dir / f"{condition}_{view}.png"
                    for condition in CONDITIONS
                ]
                annotated_paths = [
                    task_dir / f"{condition}_{view}_annotated.png"
                    for condition in CONDITIONS
                ]
                if all(path.exists() for path in raw_paths):
                    save_grid(
                        raw_paths,
                        CONDITIONS,
                        task_dir / f"six_conditions_{view}.png",
                        columns=3,
                    )
                if all(path.exists() for path in annotated_paths):
                    save_grid(
                        annotated_paths,
                        CONDITIONS,
                        task_dir / f"six_conditions_{view}_annotated.png",
                        columns=3,
                    )
        overview_dir = PREVIEWS / suite / "overview"
        for condition in CONDITIONS:
            for view in ("agentview", "wrist"):
                paths = [
                    PREVIEWS
                    / suite
                    / f"task_{task_id:02d}"
                    / f"{condition}_{view}_annotated.png"
                    for task_id in range(10)
                ]
                if all(path.exists() for path in paths):
                    save_grid(
                        paths,
                        [f"task {task_id}" for task_id in range(10)],
                        overview_dir / f"{condition}_{view}_all_tasks.png",
                        columns=5,
                    )

    cross_suite = PREVIEWS / "cross_suite_overview"
    for condition in CONDITIONS:
        for view in ("agentview", "wrist"):
            paths = [
                PREVIEWS
                / suite
                / "overview"
                / f"{condition}_{view}_all_tasks.png"
                for suite in SUITES
            ]
            if all(path.exists() for path in paths):
                stack_images(
                    paths,
                    SUITES,
                    cross_suite / f"{condition}_{view}_all_40_tasks.png",
                )
    representative = [
        PREVIEWS / suite / "task_00" / "six_conditions_agentview_annotated.png"
        for suite in SUITES
    ]
    if all(path.exists() for path in representative):
        stack_images(
            representative,
            [f"{suite} task 0" for suite in SUITES],
            cross_suite / "representative_six_conditions.png",
        )

    report_lines = [
        "# D1 semantic distractor validation",
        "",
        "- Scope: Object, Spatial, Goal, and Long (`libero_10`)",
        f"- Cases: {root['passed_count']}/{root['case_count']} PASS",
        "- Conditions per task: 6",
        "- Model evaluation: not run",
        "",
    ]
    for suite, report in reports.items():
        report_lines.append(
            f"- `{suite}`: {report['passed_count']}/{report['case_count']} PASS"
        )
    (VALIDATION / "validation_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    index_lines = [
        "# D1 preview index",
        "",
        "All images use official init state 0. Blue boxes mark distractors, red boxes mark task targets, and green boxes mark affordance references.",
        "",
        "## Cross-suite",
        "",
        "- `cross_suite_overview/representative_six_conditions.png`",
    ]
    for condition in CONDITIONS:
        index_lines.append(
            f"- `{condition}`: "
            f"`cross_suite_overview/{condition}_agentview_all_40_tasks.png`, "
            f"`cross_suite_overview/{condition}_wrist_all_40_tasks.png`"
        )
    for suite in SUITES:
        index_lines.extend(["", f"## {suite}", ""])
        for condition in CONDITIONS:
            index_lines.append(
                f"- `{condition}`: "
                f"`{suite}/overview/{condition}_agentview_all_tasks.png`, "
                f"`{suite}/overview/{condition}_wrist_all_tasks.png`"
            )
        index_lines.append(
            f"- Per-task six-condition sheets: `{suite}/task_00/` through `{suite}/task_09/`"
        )
    (PREVIEWS / "PREVIEW_INDEX.md").write_text(
        "\n".join(index_lines) + "\n",
        encoding="utf-8",
    )

    if not root["all_passed"]:
        raise SystemExit(
            f"validation incomplete: {root['passed_count']}/{root['case_count']}"
        )
    print(
        json.dumps(
            {
                "cases": root["case_count"],
                "passed": root["passed_count"],
                "all_passed": root["all_passed"],
                "model_evaluation": root["model_evaluation"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
