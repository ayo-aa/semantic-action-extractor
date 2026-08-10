"""Evidence-preserving schema for QA-SRL and QANom research annotations.

This module intentionally stays separate from the compact inference schema.  It
preserves source boundaries, dataset lineage, alternative answers, and raw
annotation judgments that a training or evaluation adapter must not collapse.
Character boundaries use zero-based, end-exclusive Unicode code-point indices,
equivalent to Python string indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

from .schema import MENTION_QUALIFIER_KINDS, PREDICATE_TYPES, TextSpan


ANNOTATION_SCHEMA_VERSION = "0.3.0"


@dataclass(frozen=True, slots=True)
class AnnotationProvenance:
    """Dataset and source identifiers needed to trace an annotation record."""

    dataset: str
    release: str
    split: str
    source_id: str
    record_id: str
    document_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("dataset", "release", "split", "source_id", "record_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"provenance {field_name} cannot be empty")
        if self.document_id is not None:
            _require_nonempty(self.document_id, label="provenance document_id")
        _validate_metadata(self.metadata, label="provenance metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "release": self.release,
            "split": self.split,
            "source_id": self.source_id,
            "record_id": self.record_id,
            "document_id": self.document_id,
            "metadata": _json_copy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AnnotationToken:
    """One token and its zero-based, end-exclusive code-point boundary."""

    index: int
    span: TextSpan

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("token index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, **self.span.to_dict()}


@dataclass(frozen=True, slots=True)
class TokenAlignedSpan:
    """A contiguous source span aligned to an end-exclusive token range."""

    span: TextSpan
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if self.token_end <= self.token_start:
            raise ValueError("token_end must be greater than token_start")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.span.to_dict(),
            "token_start": self.token_start,
            "token_end": self.token_end,
        }


@dataclass(frozen=True, slots=True)
class VerbInflectionParadigm:
    """The five inflected forms stored in a QA-SRL Bank verb entry."""

    stem: str
    present_singular_3rd: str
    present_participle: str
    past: str
    past_participle: str

    def __post_init__(self) -> None:
        for field_name in (
            "stem",
            "present_singular_3rd",
            "present_participle",
            "past",
            "past_participle",
        ):
            _require_nonempty(
                getattr(self, field_name),
                label=f"verb inflection {field_name}",
            )

    def to_dict(self) -> dict[str, str]:
        """Use the field names from QA-SRL Bank's ``verbInflectedForms``."""

        return {
            "stem": self.stem,
            "presentSingular3rd": self.present_singular_3rd,
            "presentParticiple": self.present_participle,
            "past": self.past,
            "pastParticiple": self.past_participle,
        }


@dataclass(frozen=True, slots=True)
class AnswerAlternative:
    """One answer option, potentially composed of discontinuous source spans."""

    alternative_id: str
    spans: tuple[TokenAlignedSpan, ...]
    surface_form: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.alternative_id, label="answer alternative_id")
        if not self.spans:
            raise ValueError("an answer alternative must contain at least one span")
        if self.surface_form is not None and not self.surface_form.strip():
            raise ValueError("answer surface_form cannot be empty when provided")
        _validate_metadata(self.metadata, label="answer metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alternative_id": self.alternative_id,
            "spans": [span.to_dict() for span in self.spans],
            "surface_form": self.surface_form,
            "metadata": _json_copy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class QuestionJudgment:
    """One source's validity decision and zero or more answer alternatives.

    QA-SRL Bank 2.1 contains one retained annotation marked valid with no answer
    span.  The interchange layer preserves that upstream evidence instead of
    silently dropping the judgment or inventing an answer.
    """

    source_id: str
    is_valid: bool
    answers: tuple[AnswerAlternative, ...] = field(default_factory=tuple)
    confidence: float | None = None
    confidence_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.source_id, label="question judgment source_id")
        if not isinstance(self.is_valid, bool):
            raise ValueError("question is_valid must be a boolean")
        if (
            self.is_valid
            and not self.answers
            and self.metadata.get("upstream_empty_valid_spans") is not True
        ):
            raise ValueError(
                "a valid question judgment without an answer must be marked "
                "as an upstream_empty_valid_spans anomaly"
            )
        if not self.is_valid and self.answers:
            raise ValueError("an invalid question judgment cannot include answers")
        _require_unique_ids(
            (answer.alternative_id for answer in self.answers),
            label="answer alternative_id",
        )
        _validate_optional_confidence(
            self.confidence,
            self.confidence_type,
            label="question confidence",
        )
        _validate_metadata(self.metadata, label="question judgment metadata")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_id": self.source_id,
            "is_valid": self.is_valid,
            "answers": [answer.to_dict() for answer in self.answers],
            "metadata": _json_copy(self.metadata),
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
            payload["confidence_type"] = self.confidence_type
        return payload


@dataclass(frozen=True, slots=True)
class QASRLQuestionSlots:
    """Seven canonical slots plus the optional raw QANom verb decomposition."""

    wh: str
    aux: str
    subj: str
    verb: str
    obj: str
    prep: str
    obj2: str
    verb_prefix: str | None = None
    verb_slot_inflection: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.wh,
            self.aux,
            self.subj,
            self.verb,
            self.obj,
            self.prep,
            self.obj2,
        )
        if any(not isinstance(value, str) for value in values):
            raise ValueError("all seven QA-SRL slots must be strings")
        if any(not value.strip() for value in values):
            raise ValueError("QA-SRL slots use '_' rather than empty strings")
        abstract_forms = {
            "stem",
            "presentSingular3rd",
            "presentParticiple",
            "past",
            "pastParticiple",
        }
        if self.verb != "_" and abstract_forms.isdisjoint(self.verb.split()):
            raise ValueError(
                "the QA-SRL verb slot must preserve an abstract release form"
            )
        if (self.verb_prefix is None) != (self.verb_slot_inflection is None):
            raise ValueError(
                "QANom verb_prefix and verb_slot_inflection must be provided together"
            )
        if self.verb_slot_inflection is not None:
            if self.verb_slot_inflection == "":
                if self.verb != "_":
                    raise ValueError(
                        "a missing QANom inflection must remain unknown as '_'"
                    )
                return
            qanom_inflections = {
                "Stem",
                "PresentSingular3rd",
                "PresentParticiple",
                "Past",
                "PastParticiple",
            }
            if self.verb_slot_inflection not in qanom_inflections:
                raise ValueError("unsupported QANom verb_slot_inflection")
            prefix = self.verb_prefix.replace("~!~", " ").strip()
            inflection = (
                self.verb_slot_inflection[0].lower()
                + self.verb_slot_inflection[1:]
            )
            expected_verb = " ".join(part for part in (prefix, inflection) if part)
            if self.verb != expected_verb:
                raise ValueError(
                    "the canonical verb slot must match QANom prefix and inflection"
                )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "wh": self.wh,
            "aux": self.aux,
            "subj": self.subj,
            "verb": self.verb,
            "obj": self.obj,
            "prep": self.prep,
            "obj2": self.obj2,
            "verb_prefix": self.verb_prefix,
            "verb_slot_inflection": self.verb_slot_inflection,
        }


@dataclass(frozen=True, slots=True)
class QASRLQuestion:
    """One structured question with every retained annotation judgment."""

    question_id: str
    slots: QASRLQuestionSlots
    surface_form: str
    question_sources: tuple[str, ...]
    judgments: tuple[QuestionJudgment, ...]
    is_negated: bool
    is_passive: bool
    tense: str | None = None
    is_perfect: bool | None = None
    is_progressive: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.question_id, label="question_id")
        _require_nonempty(self.surface_form, label="question surface_form")
        if not self.question_sources:
            raise ValueError("a QA-SRL question must preserve its question sources")
        _require_unique_ids(self.question_sources, label="question source")
        if not self.judgments:
            raise ValueError("a QA-SRL question must preserve at least one judgment")
        for field_name in ("is_negated", "is_passive"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"question {field_name} must be a boolean")
        bank_fields = (self.tense, self.is_perfect, self.is_progressive)
        if any(value is not None for value in bank_fields) and not all(
            value is not None for value in bank_fields
        ):
            raise ValueError(
                "tense, is_perfect, and is_progressive must be provided together"
            )
        if self.tense is not None:
            _require_nonempty(self.tense, label="question tense")
            if not isinstance(self.is_perfect, bool):
                raise ValueError("question is_perfect must be a boolean")
            if not isinstance(self.is_progressive, bool):
                raise ValueError("question is_progressive must be a boolean")
        _validate_metadata(self.metadata, label="question metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "slots": self.slots.to_dict(),
            "surface_form": self.surface_form,
            "question_sources": list(self.question_sources),
            "judgments": [judgment.to_dict() for judgment in self.judgments],
            "tense": self.tense,
            "is_perfect": self.is_perfect,
            "is_progressive": self.is_progressive,
            "is_negated": self.is_negated,
            "is_passive": self.is_passive,
            "metadata": _json_copy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EventivityJudgment:
    """One QANom-style decision about whether a nominal candidate is eventive."""

    judgment_id: str
    is_eventive: bool
    annotator_id: str | None = None
    confidence: float | None = None
    confidence_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.judgment_id, label="eventivity judgment_id")
        if not isinstance(self.is_eventive, bool):
            raise ValueError("is_eventive must be a boolean")
        if self.annotator_id is not None and not self.annotator_id.strip():
            raise ValueError("annotator_id cannot be empty when provided")
        _validate_optional_confidence(
            self.confidence,
            self.confidence_type,
            label="eventivity confidence",
        )
        _validate_metadata(self.metadata, label="eventivity judgment metadata")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "judgment_id": self.judgment_id,
            "annotator_id": self.annotator_id,
            "is_eventive": self.is_eventive,
            "metadata": _json_copy(self.metadata),
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
            payload["confidence_type"] = self.confidence_type
        return payload


@dataclass(frozen=True, slots=True)
class AnnotationMentionQualifier:
    """One source-grounded qualifier on how an event is mentioned.

    Qualifiers describe wording such as negation, possibility, planning, or
    reporting. They do not assert whether the event occurred, was completed,
    or belongs to a workflow state.
    """

    kind: str
    evidence: tuple[TokenAlignedSpan, ...]

    def __post_init__(self) -> None:
        if self.kind not in MENTION_QUALIFIER_KINDS:
            choices = ", ".join(sorted(MENTION_QUALIFIER_KINDS))
            raise ValueError(f"mention qualifier kind must be one of: {choices}")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError(
                "mention qualifier evidence must be a non-empty tuple of spans"
            )
        previous_token_end = -1
        for span in self.evidence:
            if not isinstance(span, TokenAlignedSpan):
                raise TypeError(
                    "mention qualifier evidence must contain only "
                    "TokenAlignedSpan values"
                )
            if span.token_start < previous_token_end:
                raise ValueError(
                    "mention qualifier evidence must be ordered and non-overlapping"
                )
            previous_token_end = span.token_end

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "evidence": [span.to_dict() for span in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class PredicateCandidate:
    """A verbal or nominal predicate candidate and its complete annotations.

    A small number of QANom release rows attach questions to a candidate whose
    retained eventivity decision is negative. The schema permits that conflict
    so an adapter can preserve and flag it instead of changing upstream labels.
    """

    candidate_id: str
    span: TokenAlignedSpan
    lemma: str
    predicate_type: str
    related_verbal_form: str | None = None
    verb_inflected_forms: VerbInflectionParadigm | None = None
    eventivity_judgments: tuple[EventivityJudgment, ...] = field(
        default_factory=tuple
    )
    questions: tuple[QASRLQuestion, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mention_qualifiers: tuple[AnnotationMentionQualifier, ...] | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.candidate_id, label="candidate_id")
        _require_nonempty(self.lemma, label="candidate lemma")
        if self.predicate_type not in PREDICATE_TYPES:
            choices = ", ".join(sorted(PREDICATE_TYPES))
            raise ValueError(f"predicate_type must be one of: {choices}")
        if self.related_verbal_form is not None:
            _require_nonempty(
                self.related_verbal_form,
                label="related_verbal_form",
            )
        if self.predicate_type == "verbal":
            if self.verb_inflected_forms is None:
                raise ValueError(
                    "a verbal candidate must preserve verb_inflected_forms"
                )
            if self.lemma != self.verb_inflected_forms.stem:
                raise ValueError(
                    "a verbal candidate lemma must match the inflection stem"
                )
            for question in self.questions:
                if (
                    question.tense is None
                    or question.is_perfect is None
                    or question.is_progressive is None
                ):
                    raise ValueError(
                        "verbal QA-SRL questions must preserve Bank grammar fields"
                    )
        _require_unique_ids(
            (judgment.judgment_id for judgment in self.eventivity_judgments),
            label="eventivity judgment_id",
        )
        _require_unique_ids(
            (question.question_id for question in self.questions),
            label="question_id",
        )
        if self.mention_qualifiers is not None:
            if not isinstance(self.mention_qualifiers, tuple):
                raise TypeError(
                    "mention_qualifiers must be a tuple or None when not assessed"
                )
            if any(
                not isinstance(qualifier, AnnotationMentionQualifier)
                for qualifier in self.mention_qualifiers
            ):
                raise TypeError(
                    "mention_qualifiers must contain only "
                    "AnnotationMentionQualifier values"
                )
            _require_unique_ids(
                (qualifier.kind for qualifier in self.mention_qualifiers),
                label="mention qualifier kind",
            )
            if (
                self.eventivity_judgments
                and not any(
                    judgment.is_eventive
                    for judgment in self.eventivity_judgments
                )
            ):
                raise ValueError(
                    "a non-eventive candidate cannot contain mention qualifiers"
                )
        if (
            self.questions
            and self.eventivity_judgments
            and not any(item.is_eventive for item in self.eventivity_judgments)
            and self.metadata.get("upstream_eventivity_question_conflict") is not True
        ):
            raise ValueError(
                "a non-eventive candidate with questions must be marked as an "
                "upstream_eventivity_question_conflict anomaly"
            )
        _validate_metadata(self.metadata, label="candidate metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "span": self.span.to_dict(),
            "lemma": self.lemma,
            "predicate_type": self.predicate_type,
            "related_verbal_form": self.related_verbal_form,
            "verb_inflected_forms": (
                None
                if self.verb_inflected_forms is None
                else self.verb_inflected_forms.to_dict()
            ),
            "eventivity_judgments": [
                judgment.to_dict() for judgment in self.eventivity_judgments
            ],
            "questions": [question.to_dict() for question in self.questions],
            "mention_qualifiers": (
                None
                if self.mention_qualifiers is None
                else [qualifier.to_dict() for qualifier in self.mention_qualifiers]
            ),
            "metadata": _json_copy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """One fully traceable, tokenized source record for training or evaluation."""

    text: str
    tokens: tuple[AnnotationToken, ...]
    provenance: AnnotationProvenance
    candidates: tuple[PredicateCandidate, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ANNOTATION_SCHEMA_VERSION}"
            )
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("annotation text cannot be empty")
        if not self.tokens:
            raise ValueError("an annotation record must include source tokens")
        _validate_metadata(self.metadata, label="record metadata")
        _validate_tokens(self.text, self.tokens)
        _require_unique_ids(
            (candidate.candidate_id for candidate in self.candidates),
            label="candidate_id",
        )
        for candidate in self.candidates:
            _validate_aligned_span(
                candidate.span,
                text=self.text,
                tokens=self.tokens,
                label=f"candidate {candidate.candidate_id}",
            )
            if candidate.mention_qualifiers is not None:
                for qualifier in candidate.mention_qualifiers:
                    _validate_answer_spans(
                        qualifier.evidence,
                        text=self.text,
                        tokens=self.tokens,
                        label=(
                            f"candidate {candidate.candidate_id} mention qualifier "
                            f"{qualifier.kind}"
                        ),
                    )
            for question in candidate.questions:
                for judgment in question.judgments:
                    for answer in judgment.answers:
                        _validate_answer_spans(
                            answer.spans,
                            text=self.text,
                            tokens=self.tokens,
                            label=f"answer {answer.alternative_id}",
                        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "text": self.text,
            "tokens": [token.to_dict() for token in self.tokens],
            "provenance": self.provenance.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": _json_copy(self.metadata),
        }


def _validate_tokens(text: str, tokens: Sequence[AnnotationToken]) -> None:
    previous_end = -1
    for expected_index, token in enumerate(tokens):
        if token.index != expected_index:
            raise ValueError("token indexes must be contiguous and start at zero")
        span = token.span
        if span.start < previous_end:
            raise ValueError("source token spans cannot overlap")
        if span.end > len(text) or text[span.start : span.end] != span.text:
            raise ValueError("every token span must map exactly to annotation text")
        previous_end = span.end


def _validate_answer_spans(
    spans: Sequence[TokenAlignedSpan],
    *,
    text: str,
    tokens: Sequence[AnnotationToken],
    label: str,
) -> None:
    previous_token_end = -1
    for span in spans:
        _validate_aligned_span(span, text=text, tokens=tokens, label=label)
        if span.token_start < previous_token_end:
            raise ValueError(f"{label} spans must be ordered and non-overlapping")
        previous_token_end = span.token_end


def _validate_aligned_span(
    aligned_span: TokenAlignedSpan,
    *,
    text: str,
    tokens: Sequence[AnnotationToken],
    label: str,
) -> None:
    if aligned_span.token_end > len(tokens):
        raise ValueError(f"{label} token range falls outside the source tokens")
    span = aligned_span.span
    if span.end > len(text) or text[span.start : span.end] != span.text:
        raise ValueError(f"{label} character span does not match annotation text")
    expected_start = tokens[aligned_span.token_start].span.start
    expected_end = tokens[aligned_span.token_end - 1].span.end
    if span.start != expected_start or span.end != expected_end:
        raise ValueError(f"{label} character and token boundaries do not align")


def _validate_optional_confidence(
    value: float | None,
    value_type: str | None,
    *,
    label: str,
) -> None:
    if (value is None) != (value_type is None):
        raise ValueError(f"{label} and its type must be provided together")
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{label} must be between 0 and 1")
    if not isinstance(value_type, str) or not value_type.strip():
        raise ValueError(f"{label} type cannot be empty")


def _require_nonempty(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")


def _require_unique_ids(values: Iterable[str], *, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        _require_nonempty(value, label=label)
        if value in seen:
            raise ValueError(f"{label} values must be unique within their parent")
        seen.add(value)


def _validate_metadata(metadata: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label} must be a mapping")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        _validate_json_value(value, label=label)


def _validate_json_value(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return
    if isinstance(value, Mapping):
        _validate_metadata(value, label=label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item, label=label)
        return
    raise ValueError(f"{label} values must be JSON-serializable")


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    return value
