"""Temporal perturbation scheduling for LIBERO-VPro.

Schedules gate observation transforms by timestep, manipulation phase or replan
index. Per-view buffers implement B1 camera delay, freeze and replay. With no
schedule spec, observation transforms retain their static behavior.

Schedule spec examples (JSON, via LIBERO_SCHEDULE_SPEC):

  # Generic timestep window:
  {"trigger": "time_window", "time_window": {"start": 40, "end": 90}}

  # A4 discrete corruption events:
  {"trigger": "decision_indices", "decision_indices": [2, 5, 8]}

  # Generic continuous phase gate, distinct from the A4 one-shot setting:
  {"trigger": "phase", "phases": ["pre_grasp"]}

  # A4: three consecutive replans starting when the target phase is first
  # observed at a policy-replan boundary.
  # The window can continue after that phase ends.
  {"trigger": "phase_pulse", "phases": ["pre_grasp"], "replan_count": 3, "once": true}

  # B1 camera staleness, independent of observation-mask gating:
  {"camera": {"robot0_eye_in_hand": {"mode": "delay", "lag": 3}}}
"""
from __future__ import annotations

import numpy as np


# ---- manipulation phases (heuristic, no GT in LIBERO) ----
PHASES = ["initial", "reach", "pre_grasp", "grasp", "transport", "pre_place", "release"]
LAST_DIAG = {}
_GRASP_HOLD = {}
_INIT_HOLD  = {}   # hold the initial phase for the configured number of frames
_RELEASE_HOLD = {}  # keep the transient release event visible at a replan boundary
_PHASE_CONTEXT = {}  # task-aware source / destination state for phase detection


def reset_phase_state():
    """Reset process-global phase hold state before a new episode."""
    global LAST_DIAG, _GRASP_HOLD, _INIT_HOLD, _RELEASE_HOLD, _PHASE_CONTEXT
    LAST_DIAG = {}
    _GRASP_HOLD = {}
    _INIT_HOLD = {}
    _RELEASE_HOLD = {}
    _PHASE_CONTEXT = {}


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


class PerturbationScheduler:
    """Decide, per timestep, whether the obs perturbation should be applied."""

    def __init__(self, spec):
        spec = spec or {}
        self.spec = dict(spec)
        self.enabled = bool(spec) and _as_bool(spec.get("enabled", True))
        self.trigger = str(spec.get("trigger", "always")).lower()
        self.time_window = spec.get("time_window", {}) or {}
        self.replan_window = spec.get("replan_window", {}) or {}
        self.phases = set(spec.get("phases", []) or [])
        # A4 event-count schedules select discrete replan indices. Contiguous
        # duration windows are represented separately by replan_window.
        self.random_replans = spec.get("random_replans", {}) or {}
        self._rr_set = None
        if self.trigger in {"random_replans", "decision_indices"}:
            # Explicit indices fix the decision numbers across models; physical
            # timing can differ with each policy's action horizon. Otherwise use
            # a seeded random draw.
            explicit = spec.get("decision_indices") or self.random_replans.get("indices")
            if explicit:
                self._rr_set = set(int(x) for x in explicit)
            else:
                cnt = int(self.random_replans.get("count", 1))
                lo = int(self.random_replans.get("start", 2))
                hi = int(self.random_replans.get("end", 20))
                seed = int(self.random_replans.get("seed", 0))
                rng = np.random.default_rng(seed)
                pool = list(range(lo, max(hi, lo + cnt)))
                self._rr_set = set(int(x) for x in rng.choice(pool, size=min(cnt, len(pool)),
                                                              replace=False))
        self.invert = _as_bool(spec.get("invert", False))  # perturb OUTSIDE the window
        # grasp_pulse: a NARROW window of N frames right before / after the grasp event.
        # side="before" -> the final-approach frames just before grasp (phase==pre_grasp,
        #   which infer_phase now defines as a tight final-approach band -> few frames).
        # side="after"  -> the first N frames right after grasp is first detected.
        self.grasp_pulse = spec.get("grasp_pulse", {}) or {}
        self._grasp_seen = False       # has the grasp event happened yet
        self._frames_after_grasp = 0   # counter since grasp
        # place_pulse: analogous window around the PLACEMENT event (A2 receptacle).
        # The placement event = release = grasp-latch going True->False after a grasp.
        # side="before" -> the pre_place phase (holding object at the target, before
        #   release). side="after" -> the first N frames after release.
        self.place_pulse = spec.get("place_pulse", {}) or {}
        self.repeat_place_events = _as_bool(spec.get("repeat_place_events", False))
        self._place_seen = False       # has the placement (release) event happened
        self._frames_after_place = 0   # counter since placement
        self._was_grasped = False      # previous grasp state, to detect release edge
        self._phase_pulse_start = None
        self._verified_pre_place_seen = False
        self._verified_place_started = False
        self._frames_after_verified_place = 0

    def should_perturb(self, *, timestep=0, replan_idx=0, phase=None, grasped=False):
        """Return True if the obs perturbation should be applied at this step.

        `grasped` is the current grasp-latch state from infer_phase; the transition
        False->True marks the grasp event (grasp_pulse), True->False marks the
        placement/release event (place_pulse).
        """
        # track grasp event + frames since grasp (stateful, called once per step)
        if grasped and not self._grasp_seen:
            self._grasp_seen = True
            self._frames_after_grasp = 0
        elif self._grasp_seen:
            self._frames_after_grasp += 1
        # placement event = release edge (was grasped, now not) after a grasp happened
        if self._grasp_seen and self._was_grasped and (not grasped) and not self._place_seen:
            self._place_seen = True
            self._frames_after_place = 0
        elif self._place_seen:
            self._frames_after_place += 1
        self._was_grasped = grasped

        if phase == "pre_place" and not self._verified_place_started:
            self._verified_pre_place_seen = True
        just_started = False
        if (
            phase == "release"
            and self._verified_pre_place_seen
            and not self._verified_place_started
        ):
            self._verified_place_started = True
            self._frames_after_verified_place = 0
            just_started = True
        if self._verified_place_started and not just_started:
            self._frames_after_verified_place += 1

        if not self.enabled:
            active = False
        elif self.trigger == "always":
            active = True
        elif self.trigger == "time_window":
            start = int(self.time_window.get("start", 0))
            end = int(self.time_window.get("end", 10 ** 9))
            active = start <= int(timestep) < end
        elif self.trigger == "replan_window":
            start = int(self.replan_window.get("start_replan", 0))
            num = int(self.replan_window.get("num_replans", 10 ** 9))
            active = start <= int(replan_idx) < start + num
        elif self.trigger in {"random_replans", "decision_indices"}:
            # corrupt only the pre-drawn scattered replan indices
            active = int(replan_idx) in (self._rr_set or set())
        elif self.trigger == "phase":
            active = phase in self.phases
        elif self.trigger in {"phase_pulse", "phase_once"}:
            count = max(1, int(self.spec.get("replan_count", self.spec.get("num_replans", 3))))
            if self._phase_pulse_start is None:
                if phase in self.phases:
                    self._phase_pulse_start = int(replan_idx)
                    active = True
                else:
                    active = False
            else:
                active = self._phase_pulse_start <= int(replan_idx) < self._phase_pulse_start + count
        elif self.trigger == "grasp_pulse":
            side = str(self.grasp_pulse.get("side", "after")).lower()
            n = int(self.grasp_pulse.get("n_frames", 20))
            if side == "before":
                # frames before grasp = the pre_grasp phase (approach to the object).
                # Widen/shrink the window via the pre_grasp distance band in
                # infer_phase (pre_grasp_band in the schedule spec): a larger band
                # makes pre_grasp start earlier (arm farther from object) and last
                # longer. Stops once the grasp fires.
                active = (phase == "pre_grasp") and (not self._grasp_seen)
            else:  # "after"
                active = self._grasp_seen and (self._frames_after_grasp < n)
        elif self.trigger == "place_pulse":
            side = str(self.place_pulse.get("side", "after")).lower()
            n = int(self.place_pulse.get("n_frames", 40))
            if side == "before":
                # frames before placement = the pre_place phase (holding the object at
                # the target, before release). Ends when the release/placement fires.
                active = (phase == "pre_place") and (not self._place_seen)
            else:  # "after"
                active = self._place_seen and (self._frames_after_place < n)
        elif self.trigger == "post_place_verified":
            n = int(self.place_pulse.get("n_frames", 40))
            active = (
                self._verified_place_started
                and self._frames_after_verified_place < n
            )
            if (
                self.repeat_place_events
                and self._verified_place_started
                and self._frames_after_verified_place >= n
            ):
                self._verified_pre_place_seen = False
                self._verified_place_started = False
                self._frames_after_verified_place = 0
        else:
            raise ValueError(f"Unsupported schedule trigger: {self.trigger}")
        return (not active) if self.invert else active


def _base_env(env_in):
    """Return the LIBERO task env below the lightweight render wrapper."""
    base = env_in
    seen = set()
    while id(base) not in seen:
        seen.add(id(base))
        if hasattr(base, "parsed_problem") and hasattr(base, "sim"):
            return base
        nxt = getattr(base, "env", None)
        if nxt is None or nxt is base:
            break
        base = nxt
    return base


def _goal_relations(base):
    """Read exact manipulation goals from the active BDDL task.

    A relation is ``(predicate, source, destination)``.  Unary articulated
    goals use the operated region / fixture as ``source`` and ``None`` as the
    destination.  Binary ``in`` / ``on`` goals preserve the exact destination
    region (for example ``basket_1_contain_region`` rather than ``basket_1``).
    """
    parsed = getattr(base, "parsed_problem", {}) or {}
    rows = []
    for raw in parsed.get("goal_state", []) or []:
        if not raw or len(raw) < 2:
            continue
        pred = str(raw[0]).strip().lower()
        if pred in {"in", "on", "stack"} and len(raw) >= 3:
            rows.append((pred, str(raw[1]), str(raw[2])))
        elif pred in {"open", "close", "turnon", "turnoff"}:
            rows.append((pred, str(raw[1]), None))
    return rows


def _task_instruction(base):
    raw = (getattr(base, "parsed_problem", {}) or {}).get("language_instruction", "")
    if isinstance(raw, (list, tuple)):
        raw = " ".join(str(x) for x in raw)
    return str(raw).strip().lower()


def _site_parent(base, name):
    site_obj = (getattr(base, "object_sites_dict", {}) or {}).get(name)
    return getattr(site_obj, "parent_name", None) if site_obj is not None else None


def _entity_position(base, name):
    """Resolve an object body or an exact BDDL region site in world space."""
    if not name:
        return None
    sim = getattr(base, "sim", None)
    if sim is None:
        return None
    # Goal regions are MuJoCo sites and must take precedence over their parent
    # body.  This is what distinguishes top/middle/bottom drawer and table zones.
    if name in (getattr(base, "object_sites_dict", {}) or {}):
        try:
            sid = sim.model.site_name2id(name)
            return np.asarray(sim.data.site_xpos[int(sid)], dtype=float).reshape(-1)[:3]
        except Exception:
            pass
    bid = (getattr(base, "obj_body_id", {}) or {}).get(name)
    if bid is not None:
        try:
            return np.asarray(sim.data.body_xpos[int(bid)], dtype=float).reshape(-1)[:3]
        except Exception:
            pass
    try:
        sid = sim.model.site_name2id(name)
        return np.asarray(sim.data.site_xpos[int(sid)], dtype=float).reshape(-1)[:3]
    except Exception:
        return None


def _target_body_ids(base, target_names, obj_body_id):
    """Resolve target bodies, including the parent / joint body of a region."""
    model = base.sim.model
    bodies = set()
    for name in target_names:
        bid = obj_body_id.get(name)
        if bid is None:
            parent = _site_parent(base, name)
            bid = obj_body_id.get(parent)
        if bid is not None:
            bodies.add(int(bid))
        # An articulated region can identify a particular drawer.  Prefer the
        # body carrying that region's joint over the cabinet root when possible.
        site_obj = (getattr(base, "object_sites_dict", {}) or {}).get(name)
        for joint_name in list(getattr(site_obj, "joints", []) or []):
            try:
                jid = int(model.joint_name2id(joint_name))
                bodies.add(int(model.jnt_bodyid[jid]))
            except Exception:
                pass
    return bodies


def _requested_sources(base, requested_names, relations):
    """Keep phase targets on manipulated sources, never on receptacles."""
    relation_sources = [source for _, source, _ in relations]
    binary_sources = [source for pred, source, _ in relations if pred in {"in", "on", "stack"}]
    # If a task contains both an auxiliary articulated goal and a placement
    # goal (e.g. turn on stove, then place moka pot), the manipulation phases
    # refer to the transported object.  Pure articulated tasks still use their
    # exact handle / fixture region.
    preferred = binary_sources or relation_sources
    requested = [str(x) for x in (requested_names or [])]
    if requested:
        matched = []
        for source in preferred:
            parent = _site_parent(base, source)
            if source in requested or (parent and parent in requested):
                matched.append(source)
        if matched:
            preferred = matched
    # Preserve BDDL order while removing duplicates.
    return list(dict.fromkeys(preferred))


def _destination_metrics(base, source, destination):
    """Return task-aware proximity from a manipulated object to its goal.

    The lateral threshold is derived from the exact destination region when it
    exists.  For object destinations such as a plate, its horizontal radius is
    used.  A small approach margin admits the final hover immediately before
    release without labelling distant low transport as pre-place.
    """
    source_pos = _entity_position(base, source)
    dest_pos = _entity_position(base, destination)
    if source_pos is None or dest_pos is None:
        return None
    delta = np.asarray(source_pos) - np.asarray(dest_pos)
    lateral = float(np.linalg.norm(delta[:2]))
    vertical = float(delta[2])
    lateral_limit = 0.11
    vertical_limit = 0.22
    site_obj = (getattr(base, "object_sites_dict", {}) or {}).get(destination)
    if site_obj is not None:
        try:
            size = np.abs(np.asarray(site_obj.size, dtype=float).reshape(-1))
            if size.size >= 2:
                lateral_limit = float(max(size[0], size[1]) + 0.045)
            if size.size >= 3:
                vertical_limit = float(max(0.12, size[2] + 0.13))
        except Exception:
            pass
    else:
        try:
            obj = base.get_object(destination)
            lateral_limit = float(max(0.08, float(obj.horizontal_radius) + 0.04))
            top = float(np.asarray(obj.top_offset).reshape(-1)[-1])
            vertical_limit = max(0.12, abs(top) + 0.14)
        except Exception:
            pass
    return {
        "source_pos": np.asarray(source_pos, dtype=float),
        "destination_pos": np.asarray(dest_pos, dtype=float),
        "distance": float(np.linalg.norm(delta)),
        "lateral": lateral,
        "vertical": vertical,
        "lateral_limit": lateral_limit,
        "vertical_limit": vertical_limit,
        "near": bool(lateral <= lateral_limit and -0.06 <= vertical <= vertical_limit),
    }


def _articulation_progress(base, predicate, source):
    """Estimate normalized progress for open/close/turn goals (0=start, 1=goal)."""
    site_obj = (getattr(base, "object_sites_dict", {}) or {}).get(source)
    parent = _site_parent(base, source) or source
    try:
        obj = base.get_object(parent)
    except Exception:
        return None
    joints = list(getattr(site_obj, "joints", []) or []) or list(getattr(obj, "joints", []) or [])
    if not joints:
        return None
    values = []
    for joint in joints:
        try:
            addr = base.sim.model.get_joint_qpos_addr(joint)
            values.append(float(np.asarray(base.sim.data.qpos[addr]).reshape(-1)[0]))
        except Exception:
            pass
    if not values:
        return None
    current = float(np.mean(values))
    key = (str(predicate), str(source))
    initial_map = _PHASE_CONTEXT.setdefault("initial_joint", {})
    initial = float(initial_map.setdefault(key, current))
    props = (getattr(obj, "object_properties", {}) or {}).get("articulation", {}) or {}
    ranges = props.get({
        "open": "default_open_ranges",
        "close": "default_close_ranges",
        "turnon": "default_turnon_ranges",
        "turnoff": "default_turnoff_ranges",
    }.get(predicate, ""), []) or []
    if not ranges:
        return None
    candidates = [float(x) for x in ranges]
    goal = min(candidates, key=lambda x: abs(x - initial))
    denom = goal - initial
    if abs(denom) < 1e-6:
        return 1.0
    return float(np.clip((current - initial) / denom, 0.0, 1.25))


def _finger_object_contact(base, sim, target_names, obj_body_id):
    """Return (touching, min_pad_to_object_dist).

    touching: True if a gripper finger-pad geom is in physical contact with a geom
      belonging to one of the target objects (via sim.data.contact).
    min_pad_to_object_dist: nearest distance from a finger-pad site to a target
      object body centre (fallback proxy for the approach band).
    """
    try:
        model, data = sim.model, sim.data
    except Exception:
        return False, None
    # geom ids for finger pads and for target-object geoms
    pad_geoms = set()
    for gname in ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision",
                  "gripper0_finger1_collision", "gripper0_finger2_collision"):
        try:
            gid = model.geom_name2id(gname)
            if gid >= 0:
                pad_geoms.add(int(gid))
        except Exception:
            pass
    target_bodies = _target_body_ids(base, target_names, obj_body_id)
    # a geom belongs to a target object if its body is (a descendant of) a target body
    def geom_is_target(gid):
        try:
            body = int(model.geom_bodyid[gid])
            # Articulated targets (cabinet drawer / stove controls) put their
            # contact geoms on child bodies. Walk the MuJoCo body tree instead
            # of requiring an exact top-level body id.
            while body > 0:
                if body in target_bodies:
                    return True
                body = int(model.body_parentid[body])
            return body in target_bodies
        except Exception:
            return False
    touching = False
    try:
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in pad_geoms and geom_is_target(g2)) or (g2 in pad_geoms and geom_is_target(g1)):
                touching = True
                break
    except Exception:
        pass
    # min pad-site to object distance (approach proxy)
    pad_dist = None
    try:
        obj_c = [np.asarray(data.body_xpos[b]) for b in target_bodies]
        for sname in ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"):
            try:
                gid = model.geom_name2id(sname)
                pc = np.asarray(data.geom_xpos[gid])
                for oc in obj_c:
                    d = float(np.linalg.norm(pc - oc))
                    pad_dist = d if pad_dist is None else min(pad_dist, d)
            except Exception:
                pass
    except Exception:
        pass
    return touching, pad_dist


def infer_phase(env_in, target_object_names=None, *, prev_phase="initial", grasped_latch=False,
                gripper_cmd=None, pre_grasp_band=0.20, grasp_hold_frames=24,
                initial_hold_frames=24, release_hold_frames=24,
                pre_place_progress=0.70):
    """Infer task-aware LIBERO manipulation phases from simulator state.

    Reach / pre-grasp / grasp are anchored on the *source* of the BDDL goal.
    Transport starts after physical acquisition/contact.  Pre-place is no longer
    a low-height fallback: for ``in`` / ``on`` goals it requires the active source
    to approach the exact destination region, and for articulated goals it
    requires substantial joint progress toward the goal predicate.
    """
    global LAST_DIAG, _GRASP_HOLD, _INIT_HOLD, _RELEASE_HOLD, _PHASE_CONTEXT

    base = _base_env(env_in)
    sim = getattr(base, "sim", None)
    if sim is None:
        return prev_phase, grasped_latch
    relations = _goal_relations(base)
    # Prime articulated initial values before the initial label is held.
    for predicate, source, destination in relations:
        if destination is None:
            _articulation_progress(base, predicate, source)

    # `initial` would otherwise vanish before any model replan can sample it.
    if prev_phase == "initial" and not _INIT_HOLD.get("armed"):
        _INIT_HOLD = {"armed": True, "left": int(initial_hold_frames)}
    if _INIT_HOLD.get("armed") and _INIT_HOLD.get("left", 0) > 0:
        _INIT_HOLD["left"] -= 1
        LAST_DIAG = {"detector": "task_aware_v2", "held_initial": True}
        return "initial", grasped_latch

    try:
        sid = sim.model.site_name2id("gripper0_grip_site")
        eef = np.asarray(sim.data.site_xpos[int(sid)], dtype=float).reshape(-1)[:3]
    except Exception:
        try:
            eef = np.asarray(sim.data.get_site_xpos("robot0_eef"), dtype=float).reshape(-1)[:3]
        except Exception:
            eef = None

    gap = 1.0  # unreadable -> open; a missing signal must not fake a grasp
    try:
        fj1 = float(sim.data.get_joint_qpos("gripper0_finger_joint1"))
        fj2 = float(sim.data.get_joint_qpos("gripper0_finger_joint2"))
        gap = abs(fj1) + abs(fj2)
    except Exception:
        pass

    names = _requested_sources(base, target_object_names, relations)
    if not names:
        names = [str(x) for x in (target_object_names or [])]
    if not names:
        names = [str(x) for x in (getattr(base, "obj_of_interest", []) or [])]
    completed = set(_PHASE_CONTEXT.get("completed_sources", set()))
    available = [name for name in names if name not in completed] or list(names)
    positions = {name: _entity_position(base, name) for name in available}
    positions = {name: pos for name, pos in positions.items() if pos is not None}
    if eef is None or not positions:
        LAST_DIAG = {"detector": "task_aware_v2", "error": "missing_eef_or_source",
                     "targets": list(names)}
        return prev_phase, grasped_latch

    active_source = _PHASE_CONTEXT.get("active_source") if grasped_latch else None
    if active_source not in positions:
        active_source = min(positions, key=lambda name: float(np.linalg.norm(eef - positions[name])))
    source_pos = positions[active_source]
    dmin = float(np.linalg.norm(eef - source_pos))

    relation = next((row for row in relations if row[1] == active_source), None)
    predicate, destination = ((relation[0], relation[2]) if relation else (None, None))
    obj_body_id = getattr(base, "obj_body_id", {}) or {}
    contact_targets = [active_source] if grasped_latch else list(positions)
    pad_touch, pad_min_dist = _finger_object_contact(
        base, sim, contact_targets, obj_body_id
    )

    # Measured Panda gap is ~0.078 open and ~0.040--0.059 around LIBERO objects.
    GAP_CLOSED = 0.062
    GAP_REOPENED = 0.070
    GRASP_NEAR = 0.12
    reached_before = prev_phase in (
        "reach", "pre_grasp", "grasp", "transport", "pre_place", "release"
    )
    closed_on_object = bool(gap < GAP_CLOSED and dmin < GRASP_NEAR)
    was_latched = bool(grasped_latch)
    if (closed_on_object or pad_touch) and reached_before and not _RELEASE_HOLD.get("armed"):
        grasped_latch = True
    just_latched = bool(grasped_latch and not was_latched)

    if just_latched:
        instruction = _task_instruction(base)
        interaction_mode = (
            "contact_transport"
            if destination is None or "push" in instruction
            else "pick_place"
        )
        _PHASE_CONTEXT.update({
            "active_source": active_source,
            "predicate": predicate,
            "destination": destination,
            "grasp_source_pos": np.asarray(source_pos, dtype=float).copy(),
            "latch_kind": "grasp" if closed_on_object else "contact",
            "interaction_mode": interaction_mode,
            "transport_seen": False,
        })
        initial_goal = _destination_metrics(base, active_source, destination)
        if initial_goal is not None:
            _PHASE_CONTEXT["initial_goal_lateral"] = float(initial_goal["lateral"])
    elif (grasped_latch and closed_on_object
          and _PHASE_CONTEXT.get("interaction_mode") == "pick_place"):
        _PHASE_CONTEXT["latch_kind"] = "grasp"

    # Hold only the acquisition edge, not every continuing contact frame.  The
    # hold is reset after release so multi-object tasks can have another grasp.
    if just_latched:
        _GRASP_HOLD = {"armed": True, "left": int(grasp_hold_frames)}
    in_grasp_hold = False
    if _GRASP_HOLD.get("armed") and _GRASP_HOLD.get("left", 0) > 0:
        _GRASP_HOLD["left"] -= 1
        in_grasp_hold = True

    goal_metrics = None
    articulation_progress = None
    moved = 0.0
    approach_gain = 0.0
    goal_progress = 0.0
    if grasped_latch:
        active_source = str(_PHASE_CONTEXT.get("active_source", active_source))
        predicate = _PHASE_CONTEXT.get("predicate", predicate)
        destination = _PHASE_CONTEXT.get("destination", destination)
        source_pos = _entity_position(base, active_source)
        grasp_pos = _PHASE_CONTEXT.get("grasp_source_pos")
        if source_pos is not None and grasp_pos is not None:
            moved = float(np.linalg.norm(np.asarray(source_pos) - np.asarray(grasp_pos)))
        if destination is not None:
            goal_metrics = _destination_metrics(base, active_source, destination)
            if goal_metrics is not None:
                initial_lat = float(_PHASE_CONTEXT.get(
                    "initial_goal_lateral", goal_metrics["lateral"]
                ))
                approach_gain = initial_lat - float(goal_metrics["lateral"])
                # Normalize progress against the exact BDDL destination's
                # lateral acceptance region.  Previously pick/place Pre-place
                # waited until the object was already inside that region.  A
                # successful rollout could then terminate after only one or two
                # replans, truncating a requested three-replan pulse.  Entering
                # Pre-place at the configured fraction of the path keeps the
                # mapping destination-specific while leaving enough decisions
                # for the complete perturbation window.
                travel_to_region = initial_lat - float(goal_metrics["lateral_limit"])
                if travel_to_region > 1e-6:
                    goal_progress = float(np.clip(
                        approach_gain / travel_to_region, 0.0, 1.25
                    ))
                elif bool(goal_metrics["near"]):
                    goal_progress = 1.0
        elif predicate in {"open", "close", "turnon", "turnoff"}:
            articulation_progress = _articulation_progress(base, predicate, active_source)

    if in_grasp_hold:
        phase = "grasp"
    elif not grasped_latch:
        if closed_on_object or pad_touch:
            phase = "grasp"
        elif dmin <= float(pre_grasp_band):
            phase = "pre_grasp"
        else:
            phase = "reach"
    else:
        latch_kind = str(_PHASE_CONTEXT.get("latch_kind", "grasp"))
        interaction_mode = str(_PHASE_CONTEXT.get("interaction_mode", "pick_place"))
        # Open fingers only indicate release if this interaction actually closed
        # them.  This preserves push / drawer / stove interactions, which use open
        # finger-pad contact throughout.
        if interaction_mode == "contact_transport":
            lost_grasp = bool(not pad_touch and dmin > 0.18)
        else:
            lost_grasp = (
                (latch_kind == "grasp" and (gap > GAP_REOPENED or dmin > 0.16))
                or (latch_kind == "contact" and not pad_touch and dmin > 0.18)
            )
        if lost_grasp:
            placed = bool(goal_metrics and goal_metrics.get("near"))
            if placed:
                done_sources = set(_PHASE_CONTEXT.get("completed_sources", set()))
                done_sources.add(active_source)
                _PHASE_CONTEXT["completed_sources"] = done_sources
            phase = "release"
            grasped_latch = False
            _PHASE_CONTEXT.pop("active_source", None)
            _PHASE_CONTEXT.pop("grasp_source_pos", None)
            _GRASP_HOLD = {}
            if not _RELEASE_HOLD.get("armed"):
                _RELEASE_HOLD = {"armed": True, "left": int(release_hold_frames)}
        else:
            transport_seen = bool(_PHASE_CONTEXT.get("transport_seen", False))
            if destination is not None and goal_metrics is not None:
                initial_lat = float(_PHASE_CONTEXT.get(
                    "initial_goal_lateral", goal_metrics["lateral"]
                ))
                min_gain = min(0.04, max(0.015, initial_lat * 0.15))
                approached = bool(approach_gain >= min_gain or moved >= 0.10)
                pre_place = bool(
                    transport_seen
                    and approached
                    and (
                        goal_metrics["near"]
                        or goal_progress >= float(pre_place_progress)
                    )
                )
            elif articulation_progress is not None:
                pre_place = bool(
                    transport_seen and articulation_progress >= float(pre_place_progress)
                )
            else:
                pre_place = False
            if pre_place:
                phase = "pre_place"
            else:
                phase = "transport"
                _PHASE_CONTEXT["transport_seen"] = True

    # Release is transient inside action chunks; keep the label visible at a
    # policy boundary without retaining the grasp latch or changing simulation.
    if _RELEASE_HOLD.get("armed") and _RELEASE_HOLD.get("left", 0) > 0:
        _RELEASE_HOLD["left"] -= 1
        phase = "release"
    elif _RELEASE_HOLD.get("armed"):
        _RELEASE_HOLD = {}

    LAST_DIAG = {
        "detector": "task_aware_v2",
        "gap": round(float(gap), 4),
        "dmin": round(float(dmin), 4),
        "pad_touch": bool(pad_touch),
        "pad_min_dist": None if pad_min_dist is None else round(float(pad_min_dist), 4),
        "active_source": active_source,
        "goal_predicate": predicate,
        "goal_destination": destination,
        "interaction_mode": _PHASE_CONTEXT.get("interaction_mode"),
        "source_displacement": round(float(moved), 4),
        "goal_approach_gain": round(float(approach_gain), 4),
        "goal_progress": round(float(goal_progress), 4),
        "articulation_progress": (
            None if articulation_progress is None else round(float(articulation_progress), 4)
        ),
    }
    if goal_metrics is not None:
        LAST_DIAG.update({
            "goal_lateral": round(float(goal_metrics["lateral"]), 4),
            "goal_vertical": round(float(goal_metrics["vertical"]), 4),
            "goal_lateral_limit": round(float(goal_metrics["lateral_limit"]), 4),
            "goal_vertical_limit": round(float(goal_metrics["vertical_limit"]), 4),
            "goal_near": bool(goal_metrics["near"]),
        })
    return phase, grasped_latch


class PerViewFrameBuffer:
    """B1 camera staleness — delay / freeze / replay, counted in REPLANS.

    The model only looks at images at replan boundaries (it executes a whole action
    chunk blind in between), so staleness must be measured in replans, not frames:
    "delayed by 3 replans" = the model sees the world as it was 3 decisions ago.
    A per-frame lag would often fall inside one chunk and be invisible to the model.

    B1 settings make one view stale and keep the other view clean.

    spec (per view): {"mode": "delay"|"freeze"|"replay",
                      "lag": N,            # delay: how many replans behind
                      "freeze_at": k,      # freeze: pin to this replan's frame
                      "replay_from": k,    # replay: first replan in recorded segment
                      "replay_length": n}  # replay: record n replans, then loop them
    """

    def __init__(self, camera_spec):
        self.spec = camera_spec or {}
        self._hist = {}        # view -> list of frames, one per replan
        self._replan_idx = 0   # how many replans recorded so far
        self.log = []          # staleness records for objective verification

    def push_replan(self, view, frames):
        """Record this replan's REAL frames for a view — call ONCE PER REPLAN with the
        full list of that replan's key frames (staleness is measured in replans, so the
        buffer must advance at replan boundaries, not per frame)."""
        if isinstance(frames, list):
            self._hist.setdefault(view, []).append([np.array(f, copy=True) for f in frames])
        else:
            self._hist.setdefault(view, []).append([np.array(frames, copy=True)])

    def end_replan(self):
        self._replan_idx += 1

    def get(self, view):
        """Return the (possibly stale) frame to feed the model for this view."""
        buf = self._hist.get(view) or []
        if not buf:
            return None
        cur = len(buf) - 1                 # index of this replan's real frame
        cfg = self.spec.get(view)
        if not cfg:
            src = cur                      # clean view
        else:
            mode = str(cfg.get("mode", "delay")).lower()
            if mode == "delay":
                src = max(0, cur - int(cfg.get("lag", 3)))
            elif mode == "freeze":
                src = min(int(cfg.get("freeze_at", 0)), cur)
            elif mode == "replay":
                # Replay a FIXED earlier segment on loop. Keep the live feed while
                # recording [replay_from, replay_from+length), then start replaying
                # that complete segment. This matches FastWAM's decision-level
                # semantics exactly: start=1,length=3 gives live replans 0..3,
                # followed by sources 1..3, 1..3, ... from replan 4 onward.
                # Unlike `delay` (which always shows fresh-but-late frames), the content
                # here is decoupled from reality — the model watches an old recording
                # while the world moves on to a completely different state.
                start = int(cfg.get("replay_from", 0))
                length = max(1, int(cfg.get("replay_length", 8)))
                src = cur if cur < start else min(
                    cur, start + ((cur - start) % length)
                )
            else:
                src = cur
        self.log.append({"replan": cur, "view": view,
                         "fed_replan": int(src), "staleness": int(cur - src)})
        return buf[src]
