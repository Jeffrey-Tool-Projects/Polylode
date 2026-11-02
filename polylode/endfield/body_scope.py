"""Complete-body scope uses the original model and explicit local adult ages."""
from __future__ import annotations

from .model import folder_of

# The handoff's adult human rig families. Unknown/nonhuman models require
# separate verified support; moving a juvenile FBX cannot change its scope.
ADULT_HUMAN_TYPES = frozenset({"girl", "lady", "boy", "gentleman"})
JUVENILE_TYPES = frozenset({"loli", "best_girl", "kids", "littleboy"})
#: The scope of a converted source with no rig families of its own (``sources``): the owner's recorded
#: adult basis and the body's sex, declared in the item's table (``declared_scope``; CLAUDE.md, the age rule).
DECLARED_FEMALE = "declared_adult_female"
FEMALE_HUMAN_TYPES = frozenset({"girl", "lady", "best_girl", DECLARED_FEMALE})
#: The only man converted (CLAUDE.md, the owner 2026-10-13, §7.919): EndminM.
MEN = frozenset({"chr_0002_endminm_postmodel"})


def declared_scope(extra: dict, character: str) -> str | None:
    """:data:`DECLARED_FEMALE` when the table declares an adult woman with its basis; ``None`` without a declaration.

    A declaration that is not one — no recorded adult basis, another sex — is refused, never
    read as no declaration.
    """
    scope = extra.get("declared_scope")
    if scope is None:
        return None
    if (not isinstance(scope, dict) or scope.get("adult") is not True or not scope.get("age_basis")
            or scope.get("sex") != "female" or not scope.get("sex_basis")):
        raise ValueError(f"{character}: the declared scope has no recorded adult female basis")
    return DECLARED_FEMALE


def require_body_scope(extra: dict, character: str) -> str:
    """Return the supported rig family or reject complete-body processing."""
    declared = declared_scope(extra, character)
    if declared is not None:
        return declared
    model = str(extra.get("model") or "")
    folder = folder_of(model)
    family = folder.split("/", 1)[0] if folder else ""
    from .hscope import declared_age
    age = declared_age(extra)
    known_adult_female = family == "best_girl" and age is not None and age >= 18
    if (age is not None and age < 18) or (family not in ADULT_HUMAN_TYPES and not known_adult_female):
        raise ValueError(f"{character}: complete body is not permitted for model type "
                         f"{family or 'undeclared'} ({model or 'no model declaration'})")
    return family
