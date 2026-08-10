import unittest

from semantic_action_extractor.srl.alignment import align_word_labels
from semantic_action_extractor.srl.propbank import (
    PropBankAdapterError,
    PropBankDialect,
    allow_all_sources,
    convert_propbank_record,
    normalize_role_label,
    parse_penn_tree,
    parse_penn_trees,
    parse_pointer,
    parse_propbank_record,
    resolve_pointer,
)


class FakeEncoding(dict):
    def __init__(self, word_ids, **values):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    def __init__(self, encoding):
        self.encoding = encoding

    def __call__(self, words, **kwargs):
        del words, kwargs
        return self.encoding


TREE_A = """
(TOP
  (S
    (NP-SBJ (NNP Mira))
    (VP (VBD mailed)
        (NP (DT a) (NN parcel))
        (PP-TMP (IN on) (NNP Tuesday)))
    (. .)))
"""

TREE_B = """
(TOP
  (S
    (NP-SBJ (NNP Kai))
    (VP (VBD promised)
        (S (NP-SBJ (-NONE- *-1))
           (VP (TO to) (VB return))))
    (. .)))
"""

TREE_C = """
(TOP
  (S
    (NP-SBJ (NNP Lena))
    (VP (VBD described)
        (NP (NP (DT a) (NN plan))
            (CC and)
            (NP (DT a) (NN budget)))
        (NP (-NONE- *T*-7)))
    (. .)))
"""

TREE_D = """
(TOP
  (S
    (NP-SBJ
      (NP (DT The) (NN artist))
      (SBAR
        (WHNP-2 (WP who))
        (S (NP-SBJ (-NONE- *T*-2))
           (VP (VBD smiled)))))
    (. .)))
"""

TREE_E = """
(TOP
  (S
    (NP-SBJ (PRP They))
    (VP (VBD compared)
        (ADVP (RB more))
        (PP (-NONE- *ICH*-4))
        (SBAR-4 (IN than)
          (S (NP-SBJ (PRP we))
             (VP (VBD expected)))))
    (. .)))
"""


class PennTreeTests(unittest.TestCase):
    def test_parses_surface_words_and_retains_empty_terminal_indexes(self) -> None:
        tree = parse_penn_tree(TREE_B)

        self.assertEqual(
            tuple(terminal.word for terminal in tree.terminals()),
            ("Kai", "promised", "*-1", "to", "return", "."),
        )
        self.assertEqual(
            tree.surface_words(), ("Kai", "promised", "to", "return", ".")
        )
        self.assertEqual(tree.terminal_to_surface(), (0, 1, None, 2, 3, 4))

    def test_collapses_only_an_unlabeled_serialization_wrapper(self) -> None:
        tree = parse_penn_tree("((S (NP (NNP Nia)) (VP (VBD waved))))")

        self.assertEqual(tree.label, "S")
        self.assertEqual(tree.resolve(parse_pointer("0:2").nodes[0]).node_label, "S")

        with self.assertRaises(PropBankAdapterError) as raised:
            parse_penn_tree("(TOP ((S (NP (NNP Nia)) (VP (VBD waved)))) )")
        self.assertEqual(
            raised.exception.reason_code, "nested_unlabeled_wrapper_invalid"
        )

    def test_parses_multiple_trees_and_rejects_malformed_structure(self) -> None:
        trees = parse_penn_trees(
            "(TOP (S (NP (NNP One)))) (TOP (S (NP (NNP Two))))"
        )
        self.assertEqual(len(trees), 2)

        with self.assertRaises(PropBankAdapterError) as raised:
            parse_penn_tree("(TOP (S (NP (NNP unfinished)))) trailing")
        self.assertEqual(raised.exception.reason_code, "tree_syntax_invalid")


class PointerTests(unittest.TestCase):
    def test_preserves_comma_precedence_inside_trace_chain(self) -> None:
        pointer = parse_pointer("28:1,30:1*32:1*33:0")

        self.assertEqual(len(pointer.members), 3)
        self.assertEqual(pointer.members[0].operator, ",")
        self.assertEqual(
            tuple(node.terminal for node in pointer.members[0].nodes), (28, 30)
        )

    def test_supports_semicolon_but_rejects_ambiguous_mixed_grouping(self) -> None:
        pointer = parse_pointer("2:1;4:1")
        self.assertEqual(pointer.members[0].operator, ";")

        with self.assertRaises(PropBankAdapterError) as raised:
            parse_pointer("2:1;4:1*3:1")
        self.assertEqual(
            raised.exception.reason_code, "mixed_semicolon_chain_unsupported"
        )

    def test_rejects_unknown_pointer_operator(self) -> None:
        with self.assertRaises(PropBankAdapterError) as raised:
            parse_pointer("2:1+4:1")
        self.assertEqual(raised.exception.reason_code, "pointer_syntax_invalid")

    def test_enforces_leftmost_terminal_address_invariant(self) -> None:
        tree = parse_penn_tree(TREE_A)

        with self.assertRaises(PropBankAdapterError) as raised:
            resolve_pointer(parse_pointer("1:2"), tree)
        self.assertEqual(
            raised.exception.reason_code, "pointer_leftmost_terminal_mismatch"
        )


class PropBankRecordTests(unittest.TestCase):
    def test_parses_classic_record_and_preserves_inflection(self) -> None:
        record = parse_propbank_record(
            "invented/a.mrg 0 1 gold mail.01 vp--a "
            "0:1-ARG0 1:0-rel 2:1-ARG1"
        )

        self.assertEqual(record.dialect, PropBankDialect.CLASSIC)
        self.assertEqual(record.inflection_raw, "vp--a")
        self.assertIsNone(record.typed_lemma)

    def test_parses_modern_record_and_typed_lemma(self) -> None:
        record = parse_propbank_record(
            "invented/a.parse 0 1 gold mail-v mail.01 ----- "
            "0:1-ARG0 1:0-rel 2:1-ARG1"
        )

        self.assertEqual(record.dialect, PropBankDialect.MODERN)
        self.assertEqual(record.typed_lemma, "mail")
        self.assertEqual(record.predicate_type, "v")

    def test_rejects_hybrid_record_dialect(self) -> None:
        with self.assertRaises(PropBankAdapterError) as raised:
            parse_propbank_record(
                "invented/a.parse 0 1 gold mail-v mail.01 bad "
                "0:1-ARG0 1:0-rel"
            )
        self.assertEqual(raised.exception.reason_code, "record_dialect_invalid")

    def test_rejects_modern_only_syntax_in_classic_record(self) -> None:
        for field, reason in (
            ("0:0;1:0-ARG0", "classic_pointer_operator_unsupported"),
            ("0:0*1:0-LINK-SLC", "classic_label_unsupported"),
            ("0:0-C-ARG0", "classic_label_unsupported"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(PropBankAdapterError) as raised:
                    parse_propbank_record(
                        "invented/a.mrg 0 1 gold mail.01 vp--a "
                        f"1:0-rel {field}"
                    )
                self.assertEqual(raised.exception.reason_code, reason)

    def test_requires_exactly_one_predicate_label(self) -> None:
        with self.assertRaises(PropBankAdapterError) as raised:
            parse_propbank_record(
                "invented/a.mrg 0 1 gold mail.01 vp--a 0:1-ARG0"
            )
        self.assertEqual(
            raised.exception.reason_code, "predicate_label_count_invalid"
        )

    def test_normalizes_numbered_feature_without_losing_raw_label(self) -> None:
        role = normalize_role_label("ARG2-on")

        self.assertEqual(role.raw_label, "ARG2-on")
        self.assertEqual(role.model_label, "ARG2")
        self.assertEqual(role.feature, "on")

    def test_rejects_incomplete_or_unknown_role(self) -> None:
        for label, reason in (
            ("ARGM", "argm_feature_missing"),
            ("ARGX", "unsupported_role"),
        ):
            with self.subTest(label=label):
                with self.assertRaises(PropBankAdapterError) as raised:
                    normalize_role_label(label)
                self.assertEqual(raised.exception.reason_code, reason)


class PropBankConversionTests(unittest.TestCase):
    def test_converts_classic_constituent_pointers_to_gold_bio(self) -> None:
        conversion = self._convert(
            TREE_A,
            "invented/a.mrg 0 1 gold mail.01 vp--a "
            "0:1-ARG0 1:0-rel 2:1-ARG1 4:1-ARGM-TMP",
        )

        self.assertEqual(
            conversion.example.words,
            ("Mira", "mailed", "a", "parcel", "on", "Tuesday", "."),
        )
        self.assertEqual(
            conversion.example.tags,
            (
                "B-ARG0",
                "B-V",
                "B-ARG1",
                "I-ARG1",
                "B-ARGM-TMP",
                "I-ARGM-TMP",
                "O",
            ),
        )
        self.assertEqual(conversion.example.predicate_index, 1)
        self.assertFalse(hasattr(conversion.example, "split"))
        self.assertEqual(
            conversion.provenance.source_record.raw_line,
            "invented/a.mrg 0 1 gold mail.01 vp--a "
            "0:1-ARG0 1:0-rel 2:1-ARG1 4:1-ARGM-TMP",
        )

    def test_converts_modern_record_to_the_same_word_contract(self) -> None:
        conversion = self._convert(
            TREE_A,
            "invented/a.parse 0 1 gold mail-v mail.01 ----- "
            "0:1-ARG0 1:0-rel 2:1-ARG1",
        )

        self.assertEqual(conversion.example.tags[:4], (
            "B-ARG0",
            "B-V",
            "B-ARG1",
            "I-ARG1",
        ))
        self.assertEqual(conversion.provenance.dialect, PropBankDialect.MODERN)

    def test_resolves_empty_terminal_chain_before_surface_projection(self) -> None:
        conversion = self._convert(
            TREE_B,
            "invented/b.parse 0 4 gold return-v return.01 ----- "
            "2:1*0:1-ARG0 4:0-rel",
        )

        self.assertEqual(conversion.example.words, (
            "Kai",
            "promised",
            "to",
            "return",
            ".",
        ))
        self.assertEqual(conversion.example.predicate_index, 3)
        self.assertEqual(
            conversion.provenance.parse_terminal_to_word,
            (0, 1, None, 2, 3, 4),
        )
        self.assertEqual(conversion.example.tags[0], "B-ARG0")

    def test_rejects_argument_with_no_surface_realization(self) -> None:
        self._assert_conversion_reason(
            TREE_B,
            "invented/b.parse 0 4 gold return-v return.01 ----- "
            "2:1-ARG0 4:0-rel",
            "argument_has_no_surface_realization",
        )

    def test_keeps_comma_pieces_discontinuous_and_does_not_fill_gap(self) -> None:
        conversion = self._convert(
            TREE_C,
            "invented/c.parse 0 1 gold describe-v describe.01 ----- "
            "0:1-ARG0 1:0-rel 2:1,5:1*7:1-ARG1",
        )

        self.assertEqual(
            conversion.example.tags,
            (
                "B-ARG0",
                "B-V",
                "B-ARG1",
                "I-ARG1",
                "O",
                "B-ARG1",
                "I-ARG1",
                "O",
            ),
        )
        self.assertEqual(
            conversion.provenance.roles[1].pieces, ((2, 3), (5, 6))
        )

    def test_preserves_semicolon_piece_boundary_after_empty_removal(self) -> None:
        conversion = self._convert(
            TREE_E,
            "invented/e.parse 0 1 gold compare-v compare.01 ----- "
            "0:1-ARG0 1:0-rel 2:1;4:1-ARGM-MNR",
        )

        self.assertEqual(
            conversion.example.tags,
            (
                "B-ARG0",
                "B-V",
                "B-ARGM-MNR",
                "B-ARGM-MNR",
                "I-ARGM-MNR",
                "I-ARGM-MNR",
                "O",
            ),
        )

    def test_rejects_ambiguous_trace_chain(self) -> None:
        self._assert_conversion_reason(
            TREE_D,
            "invented/d.parse 0 4 gold smile-v smile.01 ----- "
            "3:1*2:1*0:1-ARG0 4:0-rel",
            "argument_trace_chain_ambiguous",
        )

    def test_keeps_multi_node_link_metadata_without_inferring_a_span(self) -> None:
        tree = """
        (TOP
          (S (NP-SBJ (NNP Noor))
             (VP (VBD worked)
                 (NP (-NONE- *-1))
                 (PP (-NONE- *ICH*-2)))
             (. .)))
        """
        conversion = self._convert(
            tree,
            "invented/link.parse 0 1 gold work-v work.01 ----- "
            "2:1*3:1*0:1-ARG0 1:0-rel 2:1*3:1-LINK-SLC",
        )

        self.assertEqual(conversion.example.tags[0], "B-ARG0")
        self.assertEqual(conversion.provenance.link_labels, ("LINK-SLC",))
        self.assertEqual(conversion.provenance.warnings, ())

    def test_link_metadata_does_not_invent_an_argument_surface(self) -> None:
        self._assert_conversion_reason(
            TREE_D,
            "invented/d.parse 0 4 gold smile-v smile.01 ----- "
            "3:1-ARG0 4:0-rel 3:1*2:1-LINK-SLC",
            "argument_has_no_surface_realization",
        )

    def test_rejects_malformed_or_unknown_link(self) -> None:
        self._assert_conversion_reason(
            TREE_D,
            "invented/d.parse 0 4 gold smile-v smile.01 ----- "
            "0:1-ARG0 4:0-rel 3:1-LINK-SLC",
            "link_chain_invalid",
        )
        self._assert_conversion_reason(
            TREE_D,
            "invented/d.parse 0 4 gold smile-v smile.01 ----- "
            "3:1-ARG0 4:0-rel 3:1*2:1-LINK-ZZZ",
            "unsupported_link_type",
        )

    def test_rejects_orphaned_or_ambiguously_anchored_link(self) -> None:
        self._assert_conversion_reason(
            TREE_D,
            "invented/d.parse 0 4 gold smile-v smile.01 ----- "
            "0:1-ARG0 4:0-rel 3:1*2:1-LINK-SLC",
            "link_argument_missing",
        )
        self._assert_conversion_reason(
            TREE_D,
            "invented/d.parse 0 4 gold smile-v smile.01 ----- "
            "3:1-ARG0 2:1-ARGM-ADV 4:0-rel 3:1*2:1-LINK-SLC",
            "link_argument_ambiguous",
        )

    def test_rejects_role_overlap_atomically(self) -> None:
        self._assert_conversion_reason(
            TREE_A,
            "invented/a.parse 0 1 gold mail-v mail.01 ----- "
            "0:1-ARG0 0:2-ARGM-TMP 1:0-rel",
            "semantic_role_overlap",
        )

    def test_rejects_predicate_pointer_mismatch(self) -> None:
        self._assert_conversion_reason(
            TREE_A,
            "invented/a.parse 0 1 gold mail-v mail.01 ----- "
            "0:1-ARG0 2:0-rel",
            "rel_does_not_cover_predicate",
        )

    def test_requires_verbal_type_and_penn_pos(self) -> None:
        self._assert_conversion_reason(
            TREE_A,
            "invented/a.parse 0 1 gold mail-n mail.01 ----- "
            "0:1-ARG0 1:0-rel",
            "nonverbal_type",
        )
        self._assert_conversion_reason(
            TREE_A,
            "invented/a.mrg 0 0 gold mira.01 ----- "
            "0:0-rel",
            "predicate_pos_mismatch",
        )

    def test_marks_extra_multiword_predicate_piece_as_continuation(self) -> None:
        tree = """
        (TOP (S (NP-SBJ (PRP They))
                (VP (VBD signed) (PRT (RP off)) (NP (NNS forms)))
                (. .)))
        """
        conversion = self._convert(
            tree,
            "invented/f.parse 0 1 gold sign-v sign_off.01 ----- "
            "0:1-ARG0 1:0,2:0-rel 3:1-ARG1",
        )

        self.assertEqual(
            conversion.example.tags,
            ("B-ARG0", "B-V", "B-C-V", "B-ARG1", "O"),
        )

    def test_converted_example_aligns_with_continuation_closed_labels(self) -> None:
        tree = """
        (TOP (S (NP-SBJ (PRP They))
                (VP (VBD signed) (PRT (RP off)) (NP (NNS forms)))
                (. .)))
        """
        conversion = self._convert(
            tree,
            "invented/f.parse 0 1 gold sign-v sign_off.01 ----- "
            "0:1-ARG0 1:0,2:0-rel 3:1-ARG1",
        )
        tokenizer = FakeTokenizer(
            FakeEncoding(
                [None, 0, 1, 2, 2, 3, 4, None],
                input_ids=[101, 10, 20, 30, 31, 40, 50, 102],
                attention_mask=[1] * 8,
            )
        )
        labels = {
            "O": 0,
            "B-ARG0": 1,
            "B-V": 2,
            "I-V": 3,
            "B-C-V": 4,
            "I-C-V": 5,
            "B-ARG1": 6,
            "I-ARG1": 7,
        }

        aligned = align_word_labels(
            tokenizer,
            conversion.example.words,
            conversion.example.tags,
            conversion.example.predicate_index,
            labels,
        )

        self.assertEqual(aligned.labels[3:5], (4, 5))
        self.assertEqual(sum(aligned.token_type_ids), 1)

    def test_preserves_numbered_role_feature_in_provenance(self) -> None:
        conversion = self._convert(
            TREE_A,
            "invented/a.mrg 0 1 gold mail.01 vp--a "
            "0:1-ARG0 1:0-rel 4:1-ARG2-on",
        )

        role = conversion.provenance.roles[1].role
        self.assertEqual(role.raw_label, "ARG2-on")
        self.assertEqual(role.model_label, "ARG2")
        self.assertEqual(role.feature, "on")
        self.assertEqual(conversion.example.tags[4:6], ("B-ARG2", "I-ARG2"))

    def test_requires_base_role_for_modern_continuation_or_reference(self) -> None:
        conversion = self._convert(
            TREE_C,
            "invented/c.parse 0 1 gold describe-v describe.01 ----- "
            "0:1-ARG0 1:0-rel 2:1-ARG1 5:1-C-ARG1",
        )
        self.assertEqual(
            conversion.example.tags[5:7], ("B-C-ARG1", "I-C-ARG1")
        )

        for label, reason in (
            ("C-ARG1", "continuation_role_without_base"),
            ("R-ARG1", "reference_role_without_base"),
        ):
            with self.subTest(label=label):
                self._assert_conversion_reason(
                    TREE_A,
                    "invented/a.parse 0 1 gold mail-v mail.01 ----- "
                    f"0:1-ARG0 1:0-rel 2:1-{label}",
                    reason,
                )

    def test_requires_continuation_to_follow_its_base_role(self) -> None:
        self._assert_conversion_reason(
            TREE_C,
            "invented/c.parse 0 1 gold describe-v describe.01 ----- "
            "0:1-ARG0 1:0-rel 2:1-C-ARG1 5:1-ARG1",
            "continuation_role_precedes_base",
        )

    def test_deduplicates_exact_argument_pointer_with_warning(self) -> None:
        conversion = self._convert(
            TREE_A,
            "invented/a.mrg 0 1 gold mail.01 vp--a "
            "0:1-ARG0 1:0-rel 2:1,2:1-ARG1",
        )

        self.assertEqual(
            conversion.provenance.warnings,
            ("duplicate_pointer_deduplicated:1",),
        )

    def test_denies_wsj_prefix_by_default_before_pointer_resolution(self) -> None:
        record = parse_propbank_record(
            "invented/wsj_synthetic_0001.mrg 0 999 gold mail.01 vp--a "
            "999:0-rel"
        )

        with self.assertRaises(PropBankAdapterError) as raised:
            convert_propbank_record(record, parse_penn_tree(TREE_A))
        self.assertEqual(raised.exception.reason_code, "excluded_source_wsj")

        approved_record = parse_propbank_record(
            "invented/wsj_synthetic_0001.mrg 0 1 gold mail.01 vp--a "
            "0:1-ARG0 1:0-rel"
        )
        conversion = convert_propbank_record(
            approved_record,
            parse_penn_tree(TREE_A),
            source_policy=allow_all_sources,
        )
        self.assertEqual(conversion.example.predicate_index, 1)

    @staticmethod
    def _convert(tree_source, record_source):
        return convert_propbank_record(
            parse_propbank_record(record_source),
            parse_penn_tree(tree_source),
        )

    def _assert_conversion_reason(self, tree_source, record_source, reason):
        with self.assertRaises(PropBankAdapterError) as raised:
            self._convert(tree_source, record_source)
        self.assertEqual(raised.exception.reason_code, reason)


if __name__ == "__main__":
    unittest.main()
