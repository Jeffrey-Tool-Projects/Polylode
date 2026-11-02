"""Assemble a character rig for Illusion card assets to be attached to."""
from __future__ import annotations

from ..mesh import body as body_mod
from ..mesh.mesh import Model
from ..mesh.skeleton import Proportions, Skeleton


class BuiltCharacter:
    """Result of a character build: model + rig + measurements."""

    def __init__(self, model: Model, skeleton: Skeleton, prop: Proportions,
                 params: dict, sex: int = 0, pieces: dict | None = None):
        self.model = model
        self.skeleton = skeleton
        self.prop = prop
        self.params = params
        self.sex = sex
        self.pieces = pieces or {}
        self.hair_dynamic_parts: list[str] = []
        # ``EyesCtrl.OpenMax`` — how far open the lids can go, and it is not 1:
        # ``ChaControl.ChangeEyesOpenMax`` clamps the card's ``eyesOpenMax``
        # to **0.92** before it reaches the blend-shape controller.
        # ``figure.dress_character`` fills this in from the card; see
        # ``body_head.eyes_open_max``.
        self.eyes_open_max: float = 1.0
        # ``correctOpenMax`` — the same after ``FaceBlendShape.LateUpdate``'s
        # per-frame cap and gaze correction.  ``None`` means nothing has driven
        # a gaze yet, and the resting value is derived instead.
        self.eyes_open_correct: float | None = None
        # The card's face block and one ``_eyeLook`` dict per eye (the iris
        # component merged with the head-wide ``FaceBlendShape`` corrections).
        # Both halves of the gaze read these every frame — the iris ST from the
        # pupil sliders, the lids from the three corrections.
        self.eye_face: dict | None = None
        self.eye_look: list = [None, None]
        # Each eye's ST as the iris map was baked with it — the zero the
        # runtime gaze is a difference from (``body_head.eye_look_delta``).
        self.eye_st: list = [None, None]
        # Each iris's highlight shake (``body_head.HighlightShake``), ``None``
        # when the head's bake predates ``YureTime``; the gaze of the last
        # ``face_morph.look_at`` beside it, which the shake re-reads (§7.403).
        self.eye_shake: list = [None, None]
        self.eye_gaze: tuple = ((0.0, 0.0), 0.0)
        # The expression the card was saved wearing (``face_morph.CardFace``).
        # ``<ChangeHeadAsync>`` applies it as the last thing it does, so this
        # is the character's face in every scene until something asks for
        # another one — ``chara.expression`` owns it from there.
        self.card_face = None
        # ``ChaFileStatus.clothesState`` as the character is currently worn —
        # the card's until undress moves a garment between its states; the
        # state nodes and the skin mask both read it (§7.387).
        self.clothes_state: list[int] = []
        # ``enableShapeHand``: finger-bone pose entries laid over every frame's
        # pose, after the animation — ``hand_shape.hand_override`` (§7.389).
        self.hand_override: dict = {}
        # The worn garments' DynamicBone chains (``sway_spec``) and the
        # load-time settle written into the bind for them (``sway``), which
        # ``figure.sway_live`` takes back out to solve them per frame (§7.540).
        self.sway_specs: list = []
        self.sway_deltas: dict = {}
        # The graft's height factor (``prepare_for_assets``): the height
        # slider's ``cf_n_height`` scale, which the graft writes into the bone
        # positions and not into any matrix — a length read off a bone's 3×3
        # (a collider's radius) has to be multiplied by it.
        self.rig_scale: float = 1.0
        # The graft before the card's shape pass, and the rest that pass wrote:
        # what the maker's card shape sliders reshape from (``reshape_live``).
        self.bare_graft = None
        self.shaped_rest: list = []

    @property
    def height(self) -> float:
        lo, hi = self.model.bounds()
        return float(hi[1] - lo[1])

    def part(self, name: str):
        return self.model.get(name)

    def set_group_visible(self, group: str, visible: bool) -> None:
        for p in self.model.by_group(group):
            p.visible = visible


def build_character(params: dict, sex: int = 0, *,
                    with_genitals: bool = False) -> BuiltCharacter:
    """Build the rig a card is dressed onto: skeleton, proportions, empty body.

    A character's looks come from an Illusion card and nothing else, so this
    generates no body, head, hair, clothes or accessory geometry — only the
    skeleton, the measurements the rest of the engine reads off it, and an
    **empty** ``body`` part that :func:`figure.dress_character` fills in.  If
    the card cannot be read the caller reports it; there is no procedural
    stand-in to fall back to (see CLAUDE.md).

    ``with_genitals`` defaults **False**: genitals are Origin meshes on the body
    bake (``o_mnpa`` / ``o_mnpb`` / ``o_dankon``…), attached by ``apply_body``.
    The old procedural ``gen_*`` path is retired.
    """
    model, skel, prop, pieces = body_mod.build_body_model(params, sex)
    built = BuiltCharacter(model, skel, prop, params, sex, pieces)
    if with_genitals:
        # Kept only for tests that still pass True; production always False.
        from ..mesh import genitals as gen_mod
        gen_mod.attach_genitals(built)
    return built
