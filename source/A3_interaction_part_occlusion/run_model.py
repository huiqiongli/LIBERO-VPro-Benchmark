#!/usr/bin/env python3
"""Reference FastWAM runner for the accepted A3-v26 blur conditions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from libero_vpro.runtime import prepare_setting_env, resolve_runtime
from runnable_settings.runner_common import Selection, add_selection_arguments

SIM = resolve_runtime(require=False).sim
MODELS = ("fastwam",)
CONDITIONS = {
    "gripper": HERE / "specs/A3_blur_gripper_candidate.json",
    "target_contact": HERE / "specs/A3_blur_target_contact_candidate.json",
    "fixture_contact": HERE / "specs/A3_blur_fixture_contact_candidate.json",
    "bounded_receiver": HERE / "specs/A3_blur_bounded_opening_candidate.json",
}
CONDITION_PARTS = {
    "gripper": "gripper",
    "target_contact": "object_contact",
    "fixture_contact": "object_contact",
    "bounded_receiver": "receptacle_contact",
}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CONDITION_TASKS = {
    "gripper": {
        "libero_spatial": tuple(range(10)),
        "libero_object": tuple(range(10)),
        "libero_goal": tuple(range(10)),
        "libero_10": tuple(range(10)),
    },
    "target_contact": {
        "libero_object": (0, 2, 3, 4, 5, 7, 9),
        "libero_goal": (2, 9),
        "libero_10": (2,),
    },
    "fixture_contact": {
        "libero_goal": (0, 3, 7),
        "libero_10": (2, 3, 9),
    },
    "bounded_receiver": {
        "libero_object": tuple(range(10)),
        "libero_10": (0, 1, 7),
    },
}
SMOKE_CASES = {
    "gripper": ("libero_spatial", 2),
    "target_contact": ("libero_object", 0),
    "fixture_contact": ("libero_goal", 0),
    "bounded_receiver": ("libero_object", 0),
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    add_selection_arguments(parser)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.selection = Selection.from_args(args)
        self.root = args.output_root.resolve() / args.model
        if not args.plan:
            self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.tsv"
        if not args.plan and not self.status_path.exists():
            self.status_path.write_text(
                "timestamp\tmodel\tsuite\tcondition\ttask\tstatus\n",
                encoding="utf-8",
            )
        self.children: list[subprocess.Popen] = []
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

    @property
    def trials(self) -> int:
        return 1 if self.args.smoke else self.selection.trials

    def conditions(self) -> tuple[str, ...]:
        values = tuple(
            value.strip()
            for value in self.args.conditions.split(",")
            if value.strip()
        )
        unknown = sorted(set(values) - set(CONDITIONS))
        if not values or unknown:
            raise ValueError(f"invalid A3 conditions: {unknown}")
        return values

    def cases(self):
        rows = []
        if self.args.smoke:
            for condition in self.conditions():
                suite, task_id = SMOKE_CASES[condition]
                rows.append(
                    (
                        suite,
                        condition,
                        task_id,
                        self.root / suite / condition / f"task_{task_id:02d}",
                    )
                )
        else:
            for suite in SUITES:
                for condition in self.conditions():
                    task_ids = CONDITION_TASKS[condition].get(suite, ())
                    for task_id in task_ids:
                        rows.append(
                            (
                                suite,
                                condition,
                                task_id,
                                self.root / suite / condition / f"task_{task_id:02d}",
                            )
                        )
        rows = [row for row in rows
                if (self.selection.suite is None or row[0] == self.selection.suite)
                and (self.selection.task is None or row[2] == self.selection.task)]
        if self.selection.trial_start != 0 or self.trials != 10:
            suffix = f"trials_{self.selection.trial_start:02d}_{self.selection.trial_start + self.trials - 1:02d}"
            rows = [(suite, condition, task, job / suffix) for suite, condition, task, job in rows]
        for case_index, row in enumerate(rows):
            if case_index % self.args.num_shards == self.args.shard_index:
                yield row

    def write_status(
        self, suite: str, condition: str, task_id: int, status: str
    ) -> None:
        with self.status_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{self.args.model}\t"
                f"{suite}\t{condition}\t{task_id}\t{status}\n"
            )

    def stop(self, *_: object) -> None:
        for process in reversed(self.children):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise SystemExit(143)

    def base_env(self, *, server: bool = False) -> dict[str, str]:
        env = prepare_setting_env()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(self.args.gpu),
                "MUJOCO_EGL_DEVICE_ID": str(self.args.gpu),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "PYTHONUNBUFFERED": "1",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        if server:
            env.pop("MUJOCO_EGL_DEVICE_ID", None)
        return env

    def case_env(
        self,
        suite: str,
        condition: str,
        task_id: int,
        job: Path,
        *,
        automatic_metadata: bool,
        trial_start: int = 0,
    ) -> dict[str, str]:
        env = self.base_env()
        env.update(
            {
                "WAM_A3_SPEC": str(CONDITIONS[condition]),
                "WAM_A3_SUITE": suite,
                "WAM_A3_TASK_ID": str(task_id),
                "WAM_A3_CONDITION": condition,
                "WAM_TRACK": "A3",
                "WAM_SUBTASK": "A3",
                "WAM_CONDITION": condition,
                "WAM_CASE_ID": f"A3_{condition}",
                "WAM_RECORD_EPISODES": "1",
                "WAM_RECORD_PROFILE": "compact",
                "WAM_RECORD_COMPRESS": "1",
                "WAM_EPISODE_RECORD_ROOT": str(job / "episode_records"),
            }
        )
        if automatic_metadata:
            env["WAM_A3_METADATA_DIR"] = str(job / "a3_metadata")
            env["WAM_A3_TRIAL_START"] = str(trial_start)
        else:
            env.pop("WAM_A3_METADATA_DIR", None)
            env.pop("WAM_A3_TRIAL_START", None)
        return env

    def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log: Path,
        timeout: int,
    ) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", buffering=1) as handle:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.children.append(process)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise TimeoutError(f"A3 command timed out: {command}")
            finally:
                self.children.remove(process)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)

    def start(
        self, command: list[str], *, cwd: Path, env: dict[str, str], log: Path
    ) -> subprocess.Popen:
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", buffering=1)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process._a3_log_handle = handle  # type: ignore[attr-defined]
        self.children.append(process)
        return process

    def stop_child(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process in self.children:
            self.children.remove(process)
        getattr(process, "_a3_log_handle").close()

    @staticmethod
    def wait_port(
        port: int, process: subprocess.Popen, timeout: int = 1200
    ) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"policy server exited with {process.returncode}"
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=2):
                    return
            except OSError:
                time.sleep(3)
        raise TimeoutError(f"policy port {port} did not become ready")

    def metadata_valid(
        self, path: Path, *, suite: str, condition: str, task_id: int
    ) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("suite") != suite or int(payload.get("task_idx", -1)) != task_id:
            return False
        if Path(payload.get("spec_path", "")).resolve() != CONDITIONS[condition].resolve():
            return False
        expected_part = CONDITION_PARTS[condition]
        actual_parts = payload.get("parts")
        if actual_parts is None:
            actual_parts = (payload.get("spec") or {}).get("interaction_parts")
        return actual_parts == [expected_part]

    def case_complete(
        self, job: Path, suite: str, condition: str, task_id: int
    ) -> bool:
        valid = [
            path
            for path in job.rglob("*a3_parts.json")
            if self.metadata_valid(
                path, suite=suite, condition=condition, task_id=task_id
            )
        ]
        return len(valid) >= self.trials

    def run_fastwam(self) -> None:
        runtime = resolve_runtime(require=not self.args.plan)
        repo, python = runtime.fastwam, runtime.python
        for suite, condition, task_id, job in self.cases():
            command = [
                str(python), str(runtime.runner), "task=libero_uncond_2cam224_1e-4",
                f"ckpt={runtime.checkpoint}", f"EVALUATION.dataset_stats_path={runtime.dataset_stats}",
                f"EVALUATION.task_suite_name={suite}", f"EVALUATION.task_id={task_id}",
                f"++EVALUATION.trial_start={self.selection.trial_start}",
                f"EVALUATION.num_trials={self.trials}", f"EVALUATION.output_dir={job}",
                "EVALUATION.device=cuda", "EVALUATION.num_inference_steps=10",
                "EVALUATION.num_steps_wait=30", "EVALUATION.replan_steps=10",
                "model.skip_dit_load_from_pretrain=true",
                "seed=42", f"gpu_id={self.args.gpu}",
            ]
            if self.args.plan:
                print(json.dumps({"condition": condition, "command": command}))
                continue
            if self.case_complete(job, suite, condition, task_id):
                self.write_status(suite, condition, task_id, "skip")
                continue
            job.mkdir(parents=True, exist_ok=True)
            env = self.case_env(
                suite, condition, task_id, job, automatic_metadata=True,
                trial_start=self.selection.trial_start,
            )
            env.update(
                {
                    "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                }
            )
            self.write_status(suite, condition, task_id, "start")
            self.run(
                command,
                cwd=repo,
                env=env,
                log=job / "eval.log",
                timeout=43200,
            )
            if not self.case_complete(job, suite, condition, task_id):
                raise RuntimeError(f"incomplete FastWAM A3 metadata: {job}")
            self.write_status(suite, condition, task_id, "done")


    def main(self) -> None:
        if self.args.gpu < 0:
            raise ValueError("--gpu must be non-negative")
        if not 1 <= self.trials <= 10:
            raise ValueError("--trials must be in [1,10]")
        if self.args.num_shards < 1:
            raise ValueError("--num-shards must be positive")
        if not 0 <= self.args.shard_index < self.args.num_shards:
            raise ValueError("--shard-index must be in [0,num-shards)")
        if not list(self.cases()):
            raise ValueError("selection matches no A3 cases")
        for condition in self.conditions():
            if not CONDITIONS[condition].is_file():
                raise FileNotFoundError(CONDITIONS[condition])
            payload = json.loads(CONDITIONS[condition].read_text(encoding="utf-8"))
            if payload.get("full_experiment_approved") is not True:
                raise RuntimeError(f"A3 spec is not approved for full run: {CONDITIONS[condition]}")
        self.run_fastwam()


def main() -> int:
    Runner(parse_args()).main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
