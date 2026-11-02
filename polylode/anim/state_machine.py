"""Animation state machine with blend trees, crossfades and typed parameters."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clip import AnimationClip, Pose, blend_poses

OPS = {
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: abs(float(a) - float(b)) < 1e-6 if isinstance(a, (int, float))
    else a == b,
    "ne": lambda a, b: not (abs(float(a) - float(b)) < 1e-6) if isinstance(a, (int, float))
    else a != b,
    "true": lambda a, b: bool(a),
    "false": lambda a, b: not bool(a),
    "trigger": lambda a, b: bool(a),
}


@dataclass
class Condition:
    param: str
    op: str = "true"
    value: float | bool | str = 0.0

    def test(self, params: dict) -> bool:
        fn = OPS.get(self.op)
        if fn is None:
            return False
        return bool(fn(params.get(self.param, 0.0), self.value))


@dataclass
class Transition:
    to: str
    conditions: list[Condition] = field(default_factory=list)
    duration: float = 0.20
    exit_time: float | None = None      # normalised clip time that must be reached
    priority: int = 0
    interrupt: bool = True              # may fire while another blend is running

    def ready(self, params: dict, norm_time: float) -> bool:
        if self.exit_time is not None and norm_time < self.exit_time:
            return False
        return all(c.test(params) for c in self.conditions)


# --------------------------------------------------------------------------
# blend trees
# --------------------------------------------------------------------------
class BlendTree1D:
    """Blend clips along one parameter (e.g. speed -> idle/walk/run)."""

    def __init__(self, param: str, entries: list[tuple[float, AnimationClip]],
                 sync_length: bool = True):
        self.param = param
        self.entries = sorted(entries, key=lambda e: e[0])
        self.sync_length = sync_length

    def duration(self, params: dict) -> float:
        a, b, t = self._pick(params)
        if b is None:
            return a.duration
        return a.duration * (1 - t) + b.duration * t

    def _pick(self, params: dict):
        v = float(params.get(self.param, 0.0))
        pts = self.entries
        if not pts:
            return None, None, 0.0
        if v <= pts[0][0] or len(pts) == 1:
            return pts[0][1], None, 0.0
        if v >= pts[-1][0]:
            return pts[-1][1], None, 0.0
        for i in range(len(pts) - 1):
            x0, c0 = pts[i]
            x1, c1 = pts[i + 1]
            if x0 <= v <= x1:
                t = 0.0 if x1 - x0 < 1e-9 else (v - x0) / (x1 - x0)
                return c0, c1, t
        return pts[-1][1], None, 0.0

    def sample(self, time: float, params: dict) -> Pose:
        a, b, t = self._pick(params)
        if a is None:
            return {}
        if b is None:
            return a.sample(time)
        if self.sync_length:
            # phase-match so feet don't slide between blended locomotion clips.  The phase is the
            # blend's own length, which is what the state's clock wraps at (:meth:`duration`): read
            # off the first clip's, an idle of 4 s under a walk of 1.05 s played a quarter of the
            # walk's cycle and jumped back, the planted foot 0.5 m in one frame (A2, §7.1137)
            pa = (time / max(self.duration(params), 1e-6)) % 1.0
            return blend_poses(a.sample(pa * a.duration), b.sample(pa * b.duration), t)
        return blend_poses(a.sample(time), b.sample(time), t)

    def clips(self) -> list[AnimationClip]:
        return [c for _, c in self.entries]


class BlendTree2D:
    """Blend clips on a 2D plane (e.g. strafe X / forward Y) by inverse distance."""

    def __init__(self, param_x: str, param_y: str,
                 entries: list[tuple[float, float, AnimationClip]], power: float = 2.0):
        self.param_x = param_x
        self.param_y = param_y
        self.entries = entries
        self.power = power

    def _weights(self, params: dict) -> np.ndarray:
        p = np.array([float(params.get(self.param_x, 0.0)),
                      float(params.get(self.param_y, 0.0))])
        pts = np.array([[e[0], e[1]] for e in self.entries])
        d = np.linalg.norm(pts - p, axis=1)
        if np.any(d < 1e-6):
            w = (d < 1e-6).astype(np.float64)
        else:
            w = 1.0 / np.power(d, self.power)
        return w / max(w.sum(), 1e-12)

    def duration(self, params: dict) -> float:
        w = self._weights(params)
        return float(sum(wi * e[2].duration for wi, e in zip(w, self.entries)))

    def sample(self, time: float, params: dict) -> Pose:
        w = self._weights(params)
        order = np.argsort(-w)
        pose: Pose = {}
        acc = 0.0
        for i in order:
            wi = float(w[i])
            if wi < 1e-4:
                continue
            clip = self.entries[i][2]
            phase = (time / max(clip.duration, 1e-6)) % 1.0
            s = clip.sample(phase * clip.duration)
            acc += wi
            pose = blend_poses(pose, s, wi / max(acc, 1e-9))
        return pose

    def clips(self) -> list[AnimationClip]:
        return [e[2] for e in self.entries]


@dataclass
class State:
    name: str
    motion: AnimationClip | BlendTree1D | BlendTree2D | None = None
    speed: float = 1.0
    speed_param: str | None = None
    loop: bool = True
    transitions: list[Transition] = field(default_factory=list)
    on_enter: str | None = None      # event name emitted on entry
    tag: str = ""

    def duration(self, params: dict) -> float:
        if self.motion is None:
            return 1.0
        if isinstance(self.motion, AnimationClip):
            return self.motion.duration
        return self.motion.duration(params)

    def sample(self, time: float, params: dict) -> Pose:
        if self.motion is None:
            return {}
        if isinstance(self.motion, AnimationClip):
            return self.motion.sample(time)
        return self.motion.sample(time, params)

    def clips(self) -> list[AnimationClip]:
        if self.motion is None:
            return []
        if isinstance(self.motion, AnimationClip):
            return [self.motion]
        return self.motion.clips()


class StateMachine:
    """Evaluates one active state (plus an outgoing crossfade) into a pose."""

    def __init__(self, name: str = "fsm"):
        self.name = name
        self.states: dict[str, State] = {}
        self.any_transitions: list[Transition] = []
        self.params: dict = {}
        self._triggers: set[str] = set()
        self.current: str | None = None
        self.time = 0.0
        self.prev: str | None = None
        self.prev_time = 0.0
        self.blend = 1.0
        self.blend_dur = 0.2
        self.events: list[str] = []
        self.pose: Pose = {}

    # -- authoring ---------------------------------------------------------
    def add_state(self, state: State) -> State:
        self.states[state.name] = state
        if self.current is None:
            self.current = state.name
        return state

    def state(self, name: str, motion=None, **kw) -> State:
        return self.add_state(State(name, motion, **kw))

    def link(self, frm: str, to: str, *conditions, duration: float = 0.2,
             exit_time: float | None = None, priority: int = 0) -> Transition:
        t = Transition(to, list(conditions), duration, exit_time, priority)
        if frm == "*":
            self.any_transitions.append(t)
        else:
            self.states[frm].transitions.append(t)
        return t

    # -- parameters --------------------------------------------------------
    def set(self, key: str, value) -> None:
        self.params[key] = value

    def set_many(self, values: dict) -> None:
        self.params.update(values)

    def trigger(self, key: str) -> None:
        self.params[key] = True
        self._triggers.add(key)

    def get(self, key: str, default=0.0):
        return self.params.get(key, default)

    # -- playback ----------------------------------------------------------
    def goto(self, name: str, duration: float = 0.2) -> None:
        if name not in self.states or name == self.current:
            return
        self.prev = self.current
        self.prev_time = self.time
        self.current = name
        self.time = 0.0
        self.blend_dur = max(duration, 1e-4)
        self.blend = 0.0 if duration > 1e-4 else 1.0
        st = self.states[name]
        if st.on_enter:
            self.events.append(st.on_enter)

    def _norm_time(self) -> float:
        st = self.states.get(self.current)
        if st is None:
            return 0.0
        return self.time / max(st.duration(self.params), 1e-6)

    def _check_transitions(self) -> None:
        st = self.states.get(self.current)
        if st is None:
            return
        if self.blend < 1.0:
            candidates = [t for t in (self.any_transitions + st.transitions) if t.interrupt]
        else:
            candidates = self.any_transitions + st.transitions
        nt = self._norm_time()
        best = None
        for t in sorted(candidates, key=lambda x: -x.priority):
            if t.to == self.current or t.to not in self.states:
                continue
            if t.ready(self.params, nt):
                best = t
                break
        if best is not None:
            self.goto(best.to, best.duration)

    def update(self, dt: float) -> Pose:
        self.events = []
        st = self.states.get(self.current)
        if st is None:
            return {}
        speed = st.speed
        if st.speed_param:
            speed *= float(self.params.get(st.speed_param, 1.0))
        dur = max(st.duration(self.params), 1e-6)
        t0 = self.time
        self.time += dt * speed
        if st.loop:
            self.time %= dur
        else:
            self.time = min(self.time, dur)
        for clip in st.clips():
            self.events.extend(clip.events_between(clip.wrap(t0), clip.wrap(self.time)))

        pose = st.sample(self.time, self.params)
        if self.blend < 1.0 and self.prev in self.states:
            self.prev_time += dt * speed
            prev_pose = self.states[self.prev].sample(self.prev_time, self.params)
            k = self.blend
            k = k * k * (3 - 2 * k)
            pose = blend_poses(prev_pose, pose, k)
            self.blend = min(1.0, self.blend + dt / self.blend_dur)

        self._check_transitions()
        for key in self._triggers:
            self.params[key] = False
        self._triggers.clear()
        self.pose = pose
        return pose

    def describe(self) -> str:
        return f"{self.current} t={self.time:.2f} blend={self.blend:.2f}"


# --------------------------------------------------------------------------
def build_locomotion_fsm(lib: dict) -> StateMachine:
    """Idle / walk / run with gesture and jump interrupts."""
    fsm = StateMachine("locomotion")
    move = BlendTree1D("speed", [(0.0, lib["idle"]), (1.0, lib["walk"]),
                                 (2.6, lib["run"])])
    fsm.state("move", move, speed_param="anim_speed", tag="locomotion")
    fsm.state("jump", lib["jump"], loop=False, tag="action", on_enter="jump_start")
    fsm.state("wave", lib["wave"], loop=True, tag="gesture")
    fsm.state("sit", lib["sit_idle"], loop=True, tag="sitting")
    fsm.current = "move"

    fsm.link("*", "jump", Condition("jump", "trigger"), duration=0.10, priority=10)
    fsm.link("jump", "move", exit_time=0.98, duration=0.18)
    fsm.link("move", "wave", Condition("wave", "trigger"), duration=0.25)
    fsm.link("wave", "move", Condition("wave_stop", "trigger"), duration=0.30)
    fsm.link("wave", "move", Condition("speed", "gt", 0.25), duration=0.20)
    fsm.link("*", "sit", Condition("sit", "true"), duration=0.45, priority=5)
    fsm.link("sit", "move", Condition("sit", "false"), duration=0.45)
    return fsm
