"""Fail-closed primitives for classic PropBank pointers and Penn trees.

This module is corpus agnostic.  It parses the two documented English
PropBank record dialects, resolves terminal-and-height pointers against Penn
Treebank trees, and converts one record into the source-neutral word-level SRL
contract.  Archive discovery, MASC-specific filename joins, provenance
allowlists, and split generation deliberately remain outside this boundary.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath

from .example import PreparedWordLevelSRLExample


VERBAL_PENN_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})
SUPPORTED_LINK_LABELS = frozenset(
    {"LINK-PCR", "LINK-PRO", "LINK-PSV", "LINK-SLC"}
)

_TREE_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
_POINTER_FIELD_RE = re.compile(
    r"^(?P<pointer>[0-9]+:[0-9]+(?:[,;*][0-9]+:[0-9]+)*)-"
    r"(?P<label>[^\s]+)$"
)
_TYPED_LEMMA_RE = re.compile(r"^(?P<lemma>.+)-(?P<kind>[anv])$")
_CORE_ROLE_RE = re.compile(
    r"^(?P<prefix>(?:C-|R-)?)"
    r"(?P<base>ARG(?:[0-9]+|A))"
    r"(?:-(?P<feature>[A-Za-z0-9][A-Za-z0-9_.-]*))?$"
)
_MODIFIER_ROLE_RE = re.compile(
    r"^(?P<prefix>(?:C-|R-)?)ARGM-"
    r"(?P<feature>[A-Za-z0-9][A-Za-z0-9_.-]*)$"
)


class PropBankAdapterError(ValueError):
    """An auditable, fail-closed parsing or conversion failure."""

    def __init__(self, reason_code: str, message: str) -> None:
        if not reason_code or not re.fullmatch(r"[a-z0-9_]+", reason_code):
            raise ValueError("reason_code must be nonempty snake_case")
        super().__init__(message)
        self.reason_code = reason_code


def _error(reason_code: str, message: str) -> PropBankAdapterError:
    return PropBankAdapterError(reason_code, message)


@dataclass(frozen=True, slots=True)
class PennTerminal:
    """One indexed Penn Treebank leaf and its preterminal metadata."""

    index: int
    word: str
    pos: str
    preterminal_path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PennTree:
    """A minimal immutable Penn Treebank constituent tree."""

    label: str
    children: tuple[PennTree | str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("tree label must be a string")
        if not self.label:
            raise ValueError("tree label cannot be empty")
        if not isinstance(self.children, tuple):
            raise TypeError("tree children must be a tuple")
        if not self.children:
            raise ValueError("tree node must have at least one child")
        has_words = any(isinstance(child, str) for child in self.children)
        has_nodes = any(isinstance(child, PennTree) for child in self.children)
        if has_words and has_nodes:
            raise ValueError("tree node cannot mix terminal and constituent children")
        if has_words:
            if len(self.children) != 1:
                raise ValueError("a preterminal must contain exactly one word")
            word = self.children[0]
            if not isinstance(word, str) or not word:
                raise ValueError("tree terminal cannot be empty")
        elif not has_nodes:
            raise TypeError("tree children must be PennTree nodes or strings")

    @property
    def is_preterminal(self) -> bool:
        """Whether this node directly contains one terminal word."""

        return len(self.children) == 1 and isinstance(self.children[0], str)

    def terminals(self) -> tuple[PennTerminal, ...]:
        """Return leaves in pointer-counting order, including ``-NONE-``."""

        terminals: list[PennTerminal] = []

        def visit(node: PennTree, path: tuple[int, ...]) -> None:
            if node.is_preterminal:
                word = node.children[0]
                assert isinstance(word, str)
                terminals.append(
                    PennTerminal(len(terminals), word, node.label, path)
                )
                return
            for child_index, child in enumerate(node.children):
                if not isinstance(child, PennTree):
                    raise TypeError("non-preterminal tree node contains a word")
                visit(child, path + (child_index,))

        visit(self, ())
        return tuple(terminals)

    def node_at(self, path: tuple[int, ...]) -> PennTree:
        """Return the constituent node at a child-index path."""

        node = self
        for child_index in path:
            if child_index < 0 or child_index >= len(node.children):
                raise _error("tree_path_invalid", f"invalid tree path: {path!r}")
            child = node.children[child_index]
            if not isinstance(child, PennTree):
                raise _error("tree_path_invalid", f"tree path enters a word: {path!r}")
            node = child
        return node

    def surface_words(self) -> tuple[str, ...]:
        """Return PTB words after omitting empty-element preterminals."""

        return tuple(
            terminal.word
            for terminal in self.terminals()
            if terminal.pos != "-NONE-"
        )

    def terminal_to_surface(self) -> tuple[int | None, ...]:
        """Map parse-terminal indexes to model-facing word indexes."""

        mapping: list[int | None] = []
        next_surface = 0
        for terminal in self.terminals():
            if terminal.pos == "-NONE-":
                mapping.append(None)
            else:
                mapping.append(next_surface)
                next_surface += 1
        return tuple(mapping)

    def resolve(self, pointer: TreePointer) -> ResolvedTreePointer:
        """Resolve one terminal-and-height pointer against this tree."""

        terminals = self.terminals()
        if pointer.terminal >= len(terminals):
            raise _error(
                "pointer_terminal_out_of_range",
                f"terminal {pointer.terminal} is outside a "
                f"{len(terminals)}-terminal tree",
            )
        terminal = terminals[pointer.terminal]
        preterminal_path = terminal.preterminal_path
        if pointer.height > len(preterminal_path):
            raise _error(
                "pointer_height_out_of_range",
                f"height {pointer.height} ascends above the tree root",
            )
        if pointer.height:
            target_path = preterminal_path[: -pointer.height]
        else:
            target_path = preterminal_path
        target = self.node_at(target_path)

        selected_terminal_indexes = tuple(
            item.index
            for item in terminals
            if item.preterminal_path[: len(target_path)] == target_path
        )
        if not selected_terminal_indexes:
            raise _error("pointer_selects_no_terminal", "pointer selected no terminals")
        if selected_terminal_indexes[0] != pointer.terminal:
            raise _error(
                "pointer_leftmost_terminal_mismatch",
                "the pointer terminal is not the selected node's leftmost terminal",
            )

        mapping = self.terminal_to_surface()
        surface_indexes = tuple(
            surface_index
            for terminal_index in selected_terminal_indexes
            if (surface_index := mapping[terminal_index]) is not None
        )
        return ResolvedTreePointer(
            pointer=pointer,
            node_label=target.label,
            terminal_indexes=selected_terminal_indexes,
            surface_indexes=surface_indexes,
        )


def _parse_tree_at(
    tokens: tuple[str, ...],
    start: int,
    *,
    allow_unlabeled_wrapper: bool = False,
) -> tuple[PennTree, int]:
    if start >= len(tokens) or tokens[start] != "(":
        raise _error("tree_syntax_invalid", "expected '(' at tree start")
    cursor = start + 1
    if cursor >= len(tokens):
        raise _error("tree_syntax_invalid", "unterminated tree")

    # Some PTB files wrap a tree in an unlabeled pair of parentheses.  Collapse
    # that serialization wrapper so it cannot alter pointer heights.
    if tokens[cursor] == "(":
        if not allow_unlabeled_wrapper:
            raise _error(
                "nested_unlabeled_wrapper_invalid",
                "an unlabeled wrapper is allowed only at the serialized tree root",
            )
        child, cursor = _parse_tree_at(tokens, cursor)
        if cursor >= len(tokens) or tokens[cursor] != ")":
            raise _error(
                "unlabeled_tree_wrapper_invalid",
                "an unlabeled wrapper must contain exactly one tree",
            )
        return child, cursor + 1

    label = tokens[cursor]
    if label == ")":
        raise _error("tree_syntax_invalid", "tree node is missing a label")
    cursor += 1
    children: list[PennTree | str] = []
    while cursor < len(tokens) and tokens[cursor] != ")":
        if tokens[cursor] == "(":
            child, cursor = _parse_tree_at(tokens, cursor)
            children.append(child)
        else:
            children.append(tokens[cursor])
            cursor += 1
    if cursor >= len(tokens):
        raise _error("tree_syntax_invalid", f"unterminated node {label!r}")
    try:
        node = PennTree(label, tuple(children))
    except (TypeError, ValueError) as error:
        raise _error("tree_structure_invalid", str(error)) from error
    return node, cursor + 1


def parse_penn_trees(source: str) -> tuple[PennTree, ...]:
    """Parse one or more bracketed Penn trees without external dependencies."""

    if not isinstance(source, str):
        raise TypeError("Penn tree source must be a string")
    tokens = tuple(_TREE_TOKEN_RE.findall(source))
    if not tokens:
        raise _error("tree_source_empty", "Penn tree source is empty")

    trees: list[PennTree] = []
    cursor = 0
    while cursor < len(tokens):
        tree, cursor = _parse_tree_at(
            tokens, cursor, allow_unlabeled_wrapper=True
        )
        trees.append(tree)
    return tuple(trees)


def parse_penn_tree(source: str) -> PennTree:
    """Parse exactly one bracketed Penn tree."""

    trees = parse_penn_trees(source)
    if len(trees) != 1:
        raise _error(
            "tree_count_invalid", f"expected one tree, found {len(trees)}"
        )
    return trees[0]


@dataclass(frozen=True, order=True, slots=True)
class TreePointer:
    """One zero-based ``terminal:height`` constituent address."""

    terminal: int
    height: int

    def __post_init__(self) -> None:
        if type(self.terminal) is not int or type(self.height) is not int:
            raise TypeError("pointer terminal and height must be integers")
        if self.terminal < 0 or self.height < 0:
            raise ValueError("pointer terminal and height must be non-negative")

    def __str__(self) -> str:
        return f"{self.terminal}:{self.height}"


@dataclass(frozen=True, slots=True)
class ConcatenatedPointer:
    """One chain member containing one or more concatenated nodes."""

    nodes: tuple[TreePointer, ...]
    operator: str | None = None

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("pointer member cannot be empty")
        if len(self.nodes) == 1 and self.operator is not None:
            raise ValueError("a single-node pointer member has no operator")
        if len(self.nodes) > 1 and self.operator not in {",", ";"}:
            raise ValueError("concatenated pointers require ',' or ';'")


@dataclass(frozen=True, slots=True)
class PointerExpression:
    """A pointer chain whose members preserve concatenation boundaries."""

    members: tuple[ConcatenatedPointer, ...]
    raw: str

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("pointer expression cannot be empty")
        if not isinstance(self.raw, str) or not self.raw:
            raise ValueError("raw pointer cannot be empty")

    @property
    def nodes(self) -> tuple[TreePointer, ...]:
        return tuple(node for member in self.members for node in member.nodes)


def parse_pointer(source: str) -> PointerExpression:
    """Parse documented comma/semicolon concatenation and ``*`` chains."""

    if not isinstance(source, str):
        raise TypeError("pointer source must be a string")
    if not source:
        raise _error("pointer_empty", "pointer cannot be empty")
    if ";" in source and "*" in source:
        raise _error(
            "mixed_semicolon_chain_unsupported",
            "mixed ';' and '*' grouping is not documented precisely enough",
        )
    if ";" in source and "," in source:
        raise _error(
            "mixed_concatenation_unsupported",
            "mixed ',' and ';' concatenation is unsupported",
        )
    if not re.fullmatch(r"[0-9]+:[0-9]+(?:[,;*][0-9]+:[0-9]+)*", source):
        raise _error("pointer_syntax_invalid", f"invalid pointer: {source!r}")

    members: list[ConcatenatedPointer] = []
    for raw_member in source.split("*"):
        operator: str | None = None
        if "," in raw_member:
            operator = ","
        elif ";" in raw_member:
            operator = ";"
        raw_nodes = raw_member.split(operator) if operator else [raw_member]
        nodes: list[TreePointer] = []
        for raw_node in raw_nodes:
            terminal_text, height_text = raw_node.split(":", 1)
            nodes.append(TreePointer(int(terminal_text), int(height_text)))
        members.append(ConcatenatedPointer(tuple(nodes), operator))
    return PointerExpression(tuple(members), source)


@dataclass(frozen=True, slots=True)
class ResolvedTreePointer:
    """One pointer resolved to parse terminals and model-facing words."""

    pointer: TreePointer
    node_label: str
    terminal_indexes: tuple[int, ...]
    surface_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResolvedPointerMember:
    """One resolved chain member, retaining concatenated pieces."""

    pieces: tuple[ResolvedTreePointer, ...]
    operator: str | None

    @property
    def surface_indexes(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    index
                    for piece in self.pieces
                    for index in piece.surface_indexes
                }
            )
        )


@dataclass(frozen=True, slots=True)
class ResolvedPointer:
    """A resolved expression whose members still encode trace chains."""

    pointer: PointerExpression
    members: tuple[ResolvedPointerMember, ...]


def resolve_pointer(pointer: PointerExpression, tree: PennTree) -> ResolvedPointer:
    """Resolve every node while preserving chain and piece structure."""

    if not isinstance(pointer, PointerExpression):
        raise TypeError("pointer must be a PointerExpression")
    if not isinstance(tree, PennTree):
        raise TypeError("tree must be a PennTree")
    members = tuple(
        ResolvedPointerMember(
            pieces=tuple(tree.resolve(node) for node in member.nodes),
            operator=member.operator,
        )
        for member in pointer.members
    )
    return ResolvedPointer(pointer, members)


class PropBankDialect(str, Enum):
    """The two documented English ``.prop`` fixed-field layouts."""

    CLASSIC = "classic"
    MODERN = "modern"


@dataclass(frozen=True, slots=True)
class PropBankLabel:
    """One pointer plus its unmodified source label."""

    pointer: PointerExpression
    label: str


@dataclass(frozen=True, slots=True)
class PropBankRecord:
    """One parsed supplied-predicate proposition."""

    document_id: str
    sentence_index: int
    predicate_terminal_index: int
    annotator: str
    roleset: str
    dialect: PropBankDialect
    predicate: PropBankLabel
    arguments: tuple[PropBankLabel, ...]
    links: tuple[PropBankLabel, ...]
    raw_line: str
    inflection_raw: str | None = None
    typed_lemma: str | None = None
    predicate_type: str | None = None
    unused_field: str | None = None

    @property
    def record_id(self) -> str:
        return (
            f"{self.document_id}:{self.sentence_index}:"
            f"{self.predicate_terminal_index}"
        )


def _parse_labeled_pointer(field: str) -> PropBankLabel:
    match = _POINTER_FIELD_RE.fullmatch(field)
    if match is None:
        raise _error(
            "proposition_label_invalid", f"invalid proposition label: {field!r}"
        )
    return PropBankLabel(
        parse_pointer(match.group("pointer")), match.group("label")
    )


def _nonnegative_integer(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise _error("record_index_invalid", f"{field} must be an integer") from error
    if parsed < 0:
        raise _error("record_index_invalid", f"{field} must be non-negative")
    return parsed


def parse_propbank_record(line: str) -> PropBankRecord:
    """Parse a classic or modern English PropBank stand-off record."""

    if not isinstance(line, str):
        raise TypeError("PropBank record must be a string")
    raw_line = line.strip()
    if not raw_line:
        raise _error("record_empty", "PropBank record is empty")
    fields = raw_line.split()
    if len(fields) < 7:
        raise _error("record_field_count_invalid", "record has too few fields")

    document_id = fields[0]
    sentence_index = _nonnegative_integer(fields[1], field="sentence index")
    predicate_terminal_index = _nonnegative_integer(
        fields[2], field="predicate terminal index"
    )
    annotator = fields[3]

    typed_match = _TYPED_LEMMA_RE.fullmatch(fields[4])
    if typed_match is not None:
        if len(fields) < 8 or fields[6] != "-----":
            raise _error(
                "record_dialect_invalid",
                "modern records require typed lemma, roleset, and '-----'",
            )
        dialect = PropBankDialect.MODERN
        typed_lemma = typed_match.group("lemma")
        predicate_type = typed_match.group("kind")
        roleset = fields[5]
        unused_field = fields[6]
        inflection_raw = None
        proposition_fields = fields[7:]
    else:
        if len(fields[5]) != 5:
            raise _error(
                "record_dialect_invalid",
                "classic records require a five-character inflection field",
            )
        dialect = PropBankDialect.CLASSIC
        typed_lemma = None
        predicate_type = None
        roleset = fields[4]
        inflection_raw = fields[5]
        unused_field = None
        proposition_fields = fields[6:]

    if not roleset or "." not in roleset:
        raise _error("roleset_invalid", f"invalid roleset: {roleset!r}")
    labels = tuple(_parse_labeled_pointer(field) for field in proposition_fields)
    if dialect is PropBankDialect.CLASSIC:
        if any(";" in label.pointer.raw for label in labels):
            raise _error(
                "classic_pointer_operator_unsupported",
                "classic records do not define semicolon pointers",
            )
        if any(
            label.label.startswith(("LINK-", "C-", "R-")) for label in labels
        ):
            raise _error(
                "classic_label_unsupported",
                "classic records do not define LINK-, C-, or R- labels",
            )
    predicates = tuple(label for label in labels if label.label == "rel")
    if len(predicates) != 1:
        raise _error(
            "predicate_label_count_invalid",
            f"expected exactly one rel label, found {len(predicates)}",
        )
    links = tuple(label for label in labels if label.label.startswith("LINK-"))
    arguments = tuple(
        label
        for label in labels
        if label.label != "rel" and not label.label.startswith("LINK-")
    )
    return PropBankRecord(
        document_id=document_id,
        sentence_index=sentence_index,
        predicate_terminal_index=predicate_terminal_index,
        annotator=annotator,
        roleset=roleset,
        dialect=dialect,
        predicate=predicates[0],
        arguments=arguments,
        links=links,
        raw_line=raw_line,
        inflection_raw=inflection_raw,
        typed_lemma=typed_lemma,
        predicate_type=predicate_type,
        unused_field=unused_field,
    )


@dataclass(frozen=True, slots=True)
class NormalizedRole:
    """A lossless model label plus any classic numbered-role feature."""

    raw_label: str
    model_label: str
    feature: str | None


def normalize_role_label(label: str) -> NormalizedRole:
    """Validate a semantic label without inventing continuation/reference tags."""

    if not isinstance(label, str):
        raise TypeError("role label must be a string")
    core_match = _CORE_ROLE_RE.fullmatch(label)
    if core_match is not None:
        return NormalizedRole(
            raw_label=label,
            model_label=core_match.group("prefix") + core_match.group("base"),
            feature=core_match.group("feature"),
        )
    modifier_match = _MODIFIER_ROLE_RE.fullmatch(label)
    if modifier_match is not None:
        feature = modifier_match.group("feature")
        return NormalizedRole(
            raw_label=label,
            model_label=modifier_match.group("prefix") + "ARGM-" + feature,
            feature=feature,
        )
    if label == "ARGM" or label in {"C-ARGM", "R-ARGM"}:
        raise _error("argm_feature_missing", "ARGM requires a feature")
    raise _error("unsupported_role", f"unsupported role label: {label!r}")


@dataclass(frozen=True, slots=True)
class ResolvedRole:
    """One semantic role projected to one or more surface pieces."""

    role: NormalizedRole
    raw_pointer: str
    pieces: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class PropBankConversionProvenance:
    """Adapter details retained outside the neural example."""

    source_record_id: str
    source_record: PropBankRecord
    raw_document_id: str
    dialect: PropBankDialect
    predicate_terminal_index: int
    predicate_pos: str
    parse_terminal_to_word: tuple[int | None, ...]
    predicate_surface_indexes: tuple[int, ...]
    roles: tuple[ResolvedRole, ...]
    link_labels: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PropBankConversion:
    """A model-facing example paired with its audit provenance."""

    example: PreparedWordLevelSRLExample
    provenance: PropBankConversionProvenance


SourcePolicy = Callable[[PropBankRecord], str | None]


def deny_wsj_prefixed_document(record: PropBankRecord) -> str | None:
    """Conservatively deny a ``wsj_`` basename pending rights confirmation.

    This is a backstop, not a complete MASC provenance policy.  A future
    archive adapter must also require a pinned, source-family-reviewed
    allowlist.
    """

    basename = PurePosixPath(record.document_id.replace("\\", "/")).name.casefold()
    if basename.startswith("wsj_"):
        return "excluded_source_wsj"
    return None


def allow_all_sources(record: PropBankRecord) -> str | None:
    """Explicitly opt out of source denial for an independently approved source."""

    del record
    return None


def _surface_pieces(
    resolved: ResolvedPointer,
    *,
    no_surface_reason: str,
    ambiguous_reason: str,
) -> tuple[tuple[int, ...], ...]:
    surface_members = tuple(
        member for member in resolved.members if member.surface_indexes
    )
    if not surface_members:
        raise _error(no_surface_reason, "pointer has no surface realization")
    if len(surface_members) > 1:
        raise _error(
            ambiguous_reason,
            "trace chain has more than one surface-bearing member",
        )
    member = surface_members[0]
    pieces: list[tuple[int, ...]] = []
    for piece in member.pieces:
        if not piece.surface_indexes:
            continue
        indexes = tuple(sorted(set(piece.surface_indexes)))
        if indexes in pieces:
            pieces.append(indexes)
            continue
        if any(set(indexes) & set(existing) for existing in pieces):
            raise _error(
                "pointer_piece_overlap",
                "concatenated pointer pieces overlap on the surface",
            )
        pieces.append(indexes)
    if not pieces:
        raise _error(no_surface_reason, "pointer has no surface realization")
    return tuple(pieces)


def _validate_links(record: PropBankRecord, tree: PennTree) -> None:
    """Validate LINK syntax while keeping its semantics metadata-only.

    Published English PropBank examples permit a LINK to share multiple nodes
    with one argument.  The source-neutral core therefore validates the label,
    chain shape, tree addresses, and association with exactly one argument,
    but does not infer a replacement span or a reference direction.  A
    corpus-specific rule may be added only after the real archive audit
    verifies it.
    """

    for link in record.links:
        if link.label not in SUPPORTED_LINK_LABELS:
            raise _error(
                "unsupported_link_type", f"unsupported link: {link.label!r}"
            )
        if len(link.pointer.members) < 2:
            raise _error("link_chain_invalid", "LINK pointer must contain a chain")
        link_nodes = frozenset(link.pointer.nodes)
        linked_arguments = tuple(
            argument
            for argument in record.arguments
            if link_nodes.intersection(argument.pointer.nodes)
        )
        if not linked_arguments:
            raise _error(
                "link_argument_missing",
                "LINK pointer does not share a node with a semantic argument",
            )
        if len(linked_arguments) > 1:
            raise _error(
                "link_argument_ambiguous",
                "LINK pointer shares nodes with more than one semantic argument",
            )
        resolve_pointer(link.pointer, tree)


def _iter_runs(indexes: Collection[int]) -> Iterator[tuple[int, ...]]:
    ordered = sorted(set(indexes))
    if not ordered:
        return
    run = [ordered[0]]
    for index in ordered[1:]:
        if index == run[-1] + 1:
            run.append(index)
        else:
            yield tuple(run)
            run = [index]
    yield tuple(run)


def _write_piece(
    tags: list[str], indexes: Collection[int], label: str, *, overlap_reason: str
) -> None:
    for run in _iter_runs(indexes):
        for offset, index in enumerate(run):
            if index < 0 or index >= len(tags):
                raise _error(
                    "surface_index_out_of_range", "resolved surface index is invalid"
                )
            if tags[index] != "O":
                raise _error(overlap_reason, "two labels overlap on one surface word")
            prefix = "B" if offset == 0 else "I"
            tags[index] = f"{prefix}-{label}"


def convert_propbank_record(
    record: PropBankRecord,
    tree: PennTree,
    *,
    allowed_predicate_tags: Collection[str] = VERBAL_PENN_TAGS,
    source_policy: SourcePolicy = deny_wsj_prefixed_document,
) -> PropBankConversion:
    """Convert one proposition to strict word-level BIO or reject atomically.

    The default source policy denies ``wsj_`` basenames.  Passing
    :func:`allow_all_sources` is an explicit opt-out for a separately approved
    source; a MASC adapter must instead supply its pinned provenance allowlist.
    """

    if not isinstance(record, PropBankRecord):
        raise TypeError("record must be a PropBankRecord")
    if not isinstance(tree, PennTree):
        raise TypeError("tree must be a PennTree")
    if not callable(source_policy):
        raise TypeError("source_policy must be callable")
    denial_reason = source_policy(record)
    if denial_reason is not None:
        raise _error(denial_reason, "source policy denied this document")

    terminals = tree.terminals()
    if record.predicate_terminal_index >= len(terminals):
        raise _error(
            "predicate_terminal_out_of_range",
            "fixed predicate terminal is outside the parse tree",
        )
    predicate_terminal = terminals[record.predicate_terminal_index]
    terminal_to_surface = tree.terminal_to_surface()
    predicate_surface_index = terminal_to_surface[record.predicate_terminal_index]
    if predicate_surface_index is None:
        raise _error("predicate_has_no_surface", "predicate terminal is empty")
    if record.dialect is PropBankDialect.MODERN and record.predicate_type != "v":
        raise _error("nonverbal_type", "modern record is not typed as verbal")
    allowed_tags = frozenset(allowed_predicate_tags)
    if predicate_terminal.pos not in allowed_tags:
        raise _error(
            "predicate_pos_mismatch",
            f"predicate POS {predicate_terminal.pos!r} is not eligible",
        )

    resolved_predicate = resolve_pointer(record.predicate.pointer, tree)
    predicate_pieces = _surface_pieces(
        resolved_predicate,
        no_surface_reason="predicate_has_no_surface",
        ambiguous_reason="predicate_chain_ambiguous",
    )
    predicate_indexes = tuple(
        sorted({index for piece in predicate_pieces for index in piece})
    )
    if predicate_surface_index not in predicate_indexes:
        raise _error(
            "rel_does_not_cover_predicate",
            "rel pointer does not cover the fixed predicate terminal",
        )

    _validate_links(record, tree)
    normalized_roles = tuple(
        normalize_role_label(argument.label) for argument in record.arguments
    )
    model_labels = {role.model_label for role in normalized_roles}
    for role in normalized_roles:
        if role.model_label.startswith("C-"):
            base_label = role.model_label[2:]
            if base_label not in model_labels:
                raise _error(
                    "continuation_role_without_base",
                    f"{role.raw_label!r} has no corresponding base role",
                )
        elif role.model_label.startswith("R-"):
            base_label = role.model_label[2:]
            if base_label not in model_labels:
                raise _error(
                    "reference_role_without_base",
                    f"{role.raw_label!r} has no corresponding base role",
                )

    resolved_roles: list[ResolvedRole] = []
    warnings: list[str] = []
    for argument_index, (argument, normalized) in enumerate(
        zip(record.arguments, normalized_roles, strict=True)
    ):
        resolved_argument = resolve_pointer(argument.pointer, tree)
        pieces = _surface_pieces(
            resolved_argument,
            no_surface_reason="argument_has_no_surface_realization",
            ambiguous_reason="argument_trace_chain_ambiguous",
        )

        unique_pieces: list[tuple[int, ...]] = []
        for piece in pieces:
            if piece in unique_pieces:
                warnings.append(f"duplicate_pointer_deduplicated:{argument_index}")
                continue
            if any(set(piece) & set(existing) for existing in unique_pieces):
                raise _error(
                    "argument_piece_overlap",
                    "one argument contains overlapping pointer pieces",
                )
            unique_pieces.append(piece)
        resolved_roles.append(
            ResolvedRole(
                role=normalized,
                raw_pointer=argument.pointer.raw,
                pieces=tuple(unique_pieces),
            )
        )

    for continuation in resolved_roles:
        if not continuation.role.model_label.startswith("C-"):
            continue
        base_label = continuation.role.model_label[2:]
        base_indexes = tuple(
            index
            for candidate in resolved_roles
            if candidate.role.model_label == base_label
            for piece in candidate.pieces
            for index in piece
        )
        continuation_indexes = tuple(
            index for piece in continuation.pieces for index in piece
        )
        if min(continuation_indexes) <= max(base_indexes):
            raise _error(
                "continuation_role_precedes_base",
                f"{continuation.role.raw_label!r} must follow its base role",
            )

    words = tree.surface_words()
    tags = ["O"] * len(words)
    for resolved_role in resolved_roles:
        for piece in resolved_role.pieces:
            _write_piece(
                tags,
                piece,
                resolved_role.role.model_label,
                overlap_reason="semantic_role_overlap",
            )

    if tags[predicate_surface_index] != "O":
        raise _error(
            "predicate_argument_overlap", "predicate overlaps a semantic argument"
        )
    tags[predicate_surface_index] = "B-V"
    for piece in predicate_pieces:
        extra_indexes = tuple(
            index for index in piece if index != predicate_surface_index
        )
        if extra_indexes:
            _write_piece(
                tags,
                extra_indexes,
                "C-V",
                overlap_reason="predicate_piece_overlap",
            )

    example = PreparedWordLevelSRLExample(
        example_id=record.record_id,
        document_id=record.document_id,
        sentence_id=f"{record.document_id}:{record.sentence_index}",
        words=words,
        predicate_index=predicate_surface_index,
        tags=tuple(tags),
        predicate_roleset=record.roleset,
    )
    provenance = PropBankConversionProvenance(
        source_record_id=record.record_id,
        source_record=record,
        raw_document_id=record.document_id,
        dialect=record.dialect,
        predicate_terminal_index=record.predicate_terminal_index,
        predicate_pos=predicate_terminal.pos,
        parse_terminal_to_word=terminal_to_surface,
        predicate_surface_indexes=predicate_indexes,
        roles=tuple(resolved_roles),
        link_labels=tuple(link.label for link in record.links),
        warnings=tuple(warnings),
    )
    return PropBankConversion(example=example, provenance=provenance)
