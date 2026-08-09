"""Frozen and named QA-SRL question-equivalence contracts."""

from __future__ import annotations

from .types import EvaluationQuestion


EXACT_SLOTS_V1 = "exact-slots-v1"
QASRL_GS_FIVE_FIELD_V1 = "qasrl-gs-five-field-v1"
QANOM_COARSE_ROLE_V1 = "qanom-coarse-role-v1"

QUESTION_EQUIVALENCE_MODES = {
    EXACT_SLOTS_V1,
    QASRL_GS_FIVE_FIELD_V1,
    QANOM_COARSE_ROLE_V1,
}


_CORE_WH = {"what", "who"}
_ADJUNCT_WH = {"how", "how long", "how much", "when", "where", "why"}
_QANOM_PREPOSITIONS = {
    "",
    "about",
    "across",
    "after",
    "against",
    "ahead",
    "ahead at",
    "along",
    "along with",
    "amid",
    "among",
    "around",
    "as",
    "as doing",
    "aside",
    "aside for",
    "at",
    "before",
    "behind",
    "between",
    "by",
    "by doing",
    "do",
    "doing",
    "down",
    "down from",
    "during",
    "for",
    "for doing",
    "from",
    "from among",
    "from doing",
    "in",
    "in doing",
    "in from",
    "in to",
    "into",
    "of",
    "of doing",
    "off",
    "off for",
    "off from",
    "off of",
    "off to",
    "on",
    "on doing",
    "on to",
    "on to do",
    "onto",
    "out",
    "out by",
    "out of",
    "out of doing",
    "out to",
    "out to do",
    "over",
    "over from",
    "through",
    "to",
    "to as",
    "to do",
    "to doing",
    "towards",
    "under",
    "up",
    "up doing",
    "up for",
    "up to",
    "up with",
    "upon",
    "with",
    "with doing",
    "without",
}


def questions_equivalent(
    left: EvaluationQuestion,
    right: EvaluationQuestion,
    *,
    mode: str,
) -> bool:
    if mode == EXACT_SLOTS_V1:
        return _exact_slots(left) == _exact_slots(right)
    if mode == QASRL_GS_FIVE_FIELD_V1:
        if left.surface_form.casefold() == right.surface_form.casefold():
            return True
        left_wh = _slot(left.wh)
        right_wh = _slot(right.wh)
        if not left_wh or not right_wh:
            return False
        return (
            left_wh,
            _slot(left.subj),
            _slot(left.obj),
            left.is_passive,
            left.is_negated,
        ) == (
            right_wh,
            _slot(right.subj),
            _slot(right.obj),
            right.is_passive,
            right.is_negated,
        )
    if mode == QANOM_COARSE_ROLE_V1:
        return qanom_coarse_role(left) == qanom_coarse_role(right)
    choices = ", ".join(sorted(QUESTION_EQUIVALENCE_MODES))
    raise ValueError(f"unknown question-equivalence mode {mode!r}; choose: {choices}")


def qanom_coarse_role(question: EvaluationQuestion) -> str:
    """Reproduce the He-style role mapping checked into QANom."""

    wh = _slot(question.wh)
    subj = _slot(question.subj)
    obj = _slot(question.obj)
    prep = _slot(question.prep)
    obj2 = _slot(question.obj2)

    def role_two() -> str:
        return f"R2_{_known_preposition(prep)}" if prep else "R2"

    if wh in _CORE_WH and not question.is_passive:
        if not subj:
            return "R0"
        if not obj:
            return "R1"
        if not obj2 or obj2 in {"do", "doing"}:
            return role_two()
        return "Ungrammatical-question"

    if wh in _CORE_WH and question.is_passive:
        if not subj:
            return "R1"
        if not obj:
            return "R0" if prep == "by" else role_two()
        return "R0" if prep == "by" else role_two()

    if wh in _ADJUNCT_WH:
        if not obj2 and prep:
            return f"{wh}_{_known_preposition(prep)}"
        return wh
    return "Ungrammatical-question"


def _exact_slots(question: EvaluationQuestion) -> tuple[object, ...]:
    return (
        _slot(question.wh),
        _slot(question.aux),
        _slot(question.subj),
        _slot(question.verb),
        _slot(question.obj),
        _slot(question.prep),
        _slot(question.obj2),
        question.is_passive,
        question.is_negated,
    )


def _slot(value: str) -> str:
    return "" if value == "_" else value.casefold()


def _known_preposition(value: str) -> str:
    return value if value in _QANOM_PREPOSITIONS else "UNK"
