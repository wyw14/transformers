# Copyright 2026 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import tempfile
import unittest

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers.models.bert.tokenization_bert_legacy import BertTokenizerLegacy
from transformers.testing_utils import is_tokenizers_available, require_tokenizers
from transformers.tokenization_utils_base import _ADDED_TOKEN_DIFF_ATTRIBUTES


if is_tokenizers_available():
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel


def make_wordlevel_tokenizer(vocab, **special_token_kwargs):
    """Build an offline fast tokenizer from an explicit `token -> id` map (ids may be discontinuous)."""
    backend = Tokenizer(WordLevel(dict(vocab), unk_token="[UNK]"))
    special_token_kwargs.setdefault("unk_token", "[UNK]")
    return PreTrainedTokenizerFast(tokenizer_object=backend, **special_token_kwargs)


class StubVocabTokenizer(PreTrainedTokenizerBase):
    """Read-only test double exposing exactly the data `diff_vocab` consumes.

    Unlike concrete backends, it accepts arbitrary `token -> id` maps, including discontinuous ids and several
    tokens sharing the same id.
    """

    def __init__(self, vocab, roles=None, extras=None, added_tokens_decoder=None):
        # Intentionally bypass PreTrainedTokenizerBase.__init__: no backend, encoding or downloads.
        self.SPECIAL_TOKENS_ATTRIBUTES = list(roles or {})
        self._special_tokens_map = dict(roles or {})
        self._extra_special_tokens = list(extras or [])
        self._vocab = dict(vocab)
        self._decoder = dict(added_tokens_decoder or {})

    def get_vocab(self):
        return dict(self._vocab)

    @property
    def added_tokens_decoder(self):
        return dict(self._decoder)


def assert_empty_vocab_diff(test_case, report, extra_old=None, extra_new=None):
    """A report describes no difference (the extra lists are equal to the given snapshots)."""
    test_case.assertEqual(report["vocab"], {"added": [], "removed": [], "changed": []})
    test_case.assertEqual(report["special_token_roles"], [])
    test_case.assertEqual(report["added_tokens"], {"added": [], "removed": [], "changed": []})
    extra = report["extra_special_tokens"]
    test_case.assertEqual(extra["added"], [])
    test_case.assertEqual(extra["removed"], [])
    test_case.assertFalse(extra["reordered"])
    if extra_old is not None:
        test_case.assertEqual(extra["old"], extra_old)
        test_case.assertEqual(extra["new"], extra_new if extra_new is not None else extra_old)


@require_tokenizers
class TokenizerVocabDiffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        # Minimal WordPiece vocab file for the slow backend.
        cls.slow_vocab_tokens = [
            "[PAD]",
            "[UNK]",
            "<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            "[SEP]",
            "[MASK]",
            "hello",
            "world",
        ]
        with open(os.path.join(cls.tmpdirname, "vocab.txt"), "w", encoding="utf-8") as vocab_file:
            vocab_file.write("\n".join(cls.slow_vocab_tokens))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def _make_slow_tokenizer(self):
        # A real PythonBackend (slow, dict-based) tokenizer, as opposed to the Rust-backed fast tokenizer.
        return BertTokenizerLegacy(
            vocab_file=os.path.join(self.tmpdirname, "vocab.txt"),
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

    def _make_fast_tokenizer(self, vocab, insertion_order=None, **special_token_kwargs):
        # `insertion_order` lets us build the same token->id map with a different dict traversal order.
        ordered_vocab = {token: vocab[token] for token in insertion_order or list(vocab)}
        return make_wordlevel_tokenizer(ordered_vocab, **special_token_kwargs)

    def test_self_comparison_is_empty(self):
        tokenizer = make_wordlevel_tokenizer(
            {"[UNK]": 0, "<s>": 1, "</s>": 2, "<|a|>": 3, "<|b|>": 4},
            bos_token="<s>",
            eos_token="</s>",
            extra_special_tokens=["<|a|>", "<|b|>"],
        )
        tokenizer.add_tokens([AddedToken("new-token", lstrip=True)])

        report = tokenizer.diff_vocab(tokenizer)

        assert_empty_vocab_diff(self, report, extra_old=["<|a|>", "<|b|>"], extra_new=["<|a|>", "<|b|>"])
        # The report must be directly JSON serializable.
        self.assertEqual(json.loads(json.dumps(report, ensure_ascii=False)), report)

    def test_added_removed_and_changed_token_ids(self):
        old_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0, "a": 10, "b": 11})
        new_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0, "a": 10, "b": 12, "c": 13})

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(report["vocab"]["added"], [{"token": "c", "new_id": 13}])
        self.assertEqual(report["vocab"]["removed"], [])
        self.assertEqual(report["vocab"]["changed"], [{"token": "b", "old_id": 11, "new_id": 12}])
        self.assertEqual(report["special_token_roles"], [])
        self.assertEqual(report["added_tokens"]["changed"], [])

        # Reversing the direction swaps added/removed.
        reverse_report = new_tokenizer.diff_vocab(old_tokenizer)
        self.assertEqual(reverse_report["vocab"]["removed"], [{"token": "c", "old_id": 13}])
        self.assertEqual(reverse_report["vocab"]["added"], [])
        self.assertEqual(reverse_report["vocab"]["changed"], [{"token": "b", "old_id": 12, "new_id": 11}])

    def test_id_swap_only_reports_vocab_changes(self):
        old_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0, "a": 10, "b": 11})
        new_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0, "a": 11, "b": 10})

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(
            report["vocab"]["changed"],
            [
                {"token": "a", "old_id": 10, "new_id": 11},
                {"token": "b", "old_id": 11, "new_id": 10},
            ],
        )
        self.assertEqual(report["vocab"]["added"], [])
        self.assertEqual(report["vocab"]["removed"], [])
        # Nothing else moved: roles and AddedToken records are untouched by a pure id swap.
        self.assertEqual(report["special_token_roles"], [])
        self.assertEqual(report["added_tokens"], {"added": [], "removed": [], "changed": []})

    def test_role_swap_only_reports_role_changes(self):
        vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2}
        old_tokenizer = make_wordlevel_tokenizer(vocab, bos_token="<s>", eos_token="</s>")
        new_tokenizer = make_wordlevel_tokenizer(vocab, bos_token="</s>", eos_token="<s>")

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(report["vocab"], {"added": [], "removed": [], "changed": []})
        self.assertEqual(report["added_tokens"], {"added": [], "removed": [], "changed": []})
        roles = {entry["role"]: entry for entry in report["special_token_roles"]}
        self.assertEqual(sorted(roles), ["bos_token", "eos_token"])
        self.assertEqual(
            roles["bos_token"],
            {
                "role": "bos_token",
                "old_token": "<s>",
                "old_id": 1,
                "new_token": "</s>",
                "new_id": 2,
            },
        )
        self.assertEqual(
            roles["eos_token"],
            {
                "role": "eos_token",
                "old_token": "</s>",
                "old_id": 2,
                "new_token": "<s>",
                "new_id": 1,
            },
        )
        # Entries are sorted by role name.
        self.assertEqual([entry["role"] for entry in report["special_token_roles"]], ["bos_token", "eos_token"])

    def test_role_added_on_one_side_uses_null_values(self):
        vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2}
        old_tokenizer = make_wordlevel_tokenizer(vocab, eos_token="</s>")
        new_tokenizer = make_wordlevel_tokenizer(vocab, bos_token="<s>", eos_token="</s>")

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(
            report["special_token_roles"],
            [
                {
                    "role": "bos_token",
                    "old_token": None,
                    "old_id": None,
                    "new_token": "<s>",
                    "new_id": 1,
                }
            ],
        )

    def test_custom_model_specific_role_is_reported(self):
        # "<image>" lives in the base vocab on both sides, only the named role assignment differs.
        vocab = {"[UNK]": 0, "<|a|>": 1, "<image>": 2}
        old_tokenizer = make_wordlevel_tokenizer(vocab)
        new_tokenizer = make_wordlevel_tokenizer(vocab, model_specific_special_tokens={"image_token": "<image>"})

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(report["vocab"], {"added": [], "removed": [], "changed": []})
        self.assertEqual(
            report["special_token_roles"],
            [
                {
                    "role": "image_token",
                    "old_token": None,
                    "old_id": None,
                    "new_token": "<image>",
                    "new_id": 2,
                }
            ],
        )

    def test_added_token_attribute_changes_are_isolated(self):
        base_attributes = {
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        }
        for attribute in _ADDED_TOKEN_DIFF_ATTRIBUTES:
            with self.subTest(attribute=attribute):
                old_kwargs = dict(base_attributes)
                new_kwargs = dict(base_attributes)
                new_kwargs[attribute] = True
                old_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0})
                new_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0})
                old_tokenizer.add_tokens([AddedToken("<meta>", **old_kwargs)])
                new_tokenizer.add_tokens([AddedToken("<meta>", **new_kwargs)])

                report = old_tokenizer.diff_vocab(new_tokenizer)

                # The token keeps the same id: the vocab maps are identical.
                self.assertEqual(report["vocab"], {"added": [], "removed": [], "changed": []})
                self.assertEqual(report["special_token_roles"], [])
                self.assertEqual(len(report["added_tokens"]["changed"]), 1)
                entry = report["added_tokens"]["changed"][0]
                self.assertEqual(entry["token"], "<meta>")
                self.assertEqual(entry["old"]["id"], entry["new"]["id"])
                self.assertEqual(entry["attribute_changes"], {attribute: {"old": False, "new": True}})
                # The full records of both sides are present.
                self.assertEqual(entry["old"][attribute], False)
                self.assertEqual(entry["new"][attribute], True)

    def test_added_token_records_added_and_removed(self):
        old_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0})
        old_tokenizer.add_tokens([AddedToken("<old>", lstrip=True, special=False)])
        new_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0})
        new_tokenizer.add_tokens([AddedToken("<new>", rstrip=True, special=False)])

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(len(report["added_tokens"]["added"]), 1)
        added_entry = report["added_tokens"]["added"][0]
        self.assertEqual(added_entry["token"], "<new>")
        self.assertIsNone(added_entry["old"])
        self.assertEqual(
            added_entry["new"],
            {
                "id": 1,
                "single_word": False,
                "lstrip": False,
                "rstrip": True,
                "normalized": True,
                "special": False,
            },
        )
        removed_entry = report["added_tokens"]["removed"][0]
        self.assertEqual(removed_entry["token"], "<old>")
        self.assertIsNone(removed_entry["new"])
        self.assertEqual(removed_entry["old"]["id"], 1)
        self.assertTrue(removed_entry["old"]["lstrip"])
        self.assertEqual(report["added_tokens"]["changed"], [])

    def test_extra_special_tokens_reorder(self):
        vocab = {"[UNK]": 0, "<|a|>": 1, "<|b|>": 2, "<|c|>": 3}
        old_tokenizer = make_wordlevel_tokenizer(vocab, extra_special_tokens=["<|a|>", "<|b|>", "<|c|>"])
        new_tokenizer = make_wordlevel_tokenizer(vocab, extra_special_tokens=["<|b|>", "<|a|>", "<|c|>"])

        report = old_tokenizer.diff_vocab(new_tokenizer)

        # Pure reorder: nothing else changes.
        self.assertEqual(report["vocab"], {"added": [], "removed": [], "changed": []})
        self.assertEqual(report["special_token_roles"], [])
        self.assertEqual(report["added_tokens"], {"added": [], "removed": [], "changed": []})
        self.assertEqual(report["extra_special_tokens"]["old"], ["<|a|>", "<|b|>", "<|c|>"])
        self.assertEqual(report["extra_special_tokens"]["new"], ["<|b|>", "<|a|>", "<|c|>"])
        self.assertEqual(report["extra_special_tokens"]["added"], [])
        self.assertEqual(report["extra_special_tokens"]["removed"], [])
        self.assertTrue(report["extra_special_tokens"]["reordered"])

    def test_extra_special_tokens_content_change(self):
        vocab = {"[UNK]": 0, "<|a|>": 1, "<|b|>": 2, "<|c|>": 3}
        old_tokenizer = make_wordlevel_tokenizer(vocab, extra_special_tokens=["<|a|>", "<|b|>", "<|c|>"])
        new_tokenizer = make_wordlevel_tokenizer(vocab, extra_special_tokens=["<|a|>", "<|b|>"])

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(report["extra_special_tokens"]["added"], [])
        self.assertEqual(report["extra_special_tokens"]["removed"], ["<|c|>"])
        self.assertFalse(report["extra_special_tokens"]["reordered"])

    def test_tokens_match_by_raw_string_only(self):
        # Composed vs decomposed é are different raw strings, and BPE whitespace markers are not rewritten.
        old_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0, "caf\u00e9": 5, "\u0120hello": 6})
        new_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0, "cafe\u0301": 5, " hello": 6})

        report = old_tokenizer.diff_vocab(new_tokenizer)

        # Entries are sorted by raw token string (Unicode code point order).
        self.assertEqual(
            report["vocab"]["removed"],
            [{"token": "caf\u00e9", "old_id": 5}, {"token": "\u0120hello", "old_id": 6}],
        )
        self.assertEqual(
            report["vocab"]["added"],
            [{"token": " hello", "new_id": 6}, {"token": "cafe\u0301", "new_id": 5}],
        )
        self.assertEqual(report["vocab"]["changed"], [])

    def test_discontinuous_ids_and_shared_ids(self):
        # Concrete backends renumber away such maps, so the comparison engine is exercised directly with a double.
        old_tokenizer = StubVocabTokenizer({"[UNK]": 0, "x": 5, "y": 5, "z": 100})
        new_tokenizer = StubVocabTokenizer({"[UNK]": 0, "x": 5, "y": 5, "z": 100, "w": 100})

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(report["vocab"]["added"], [{"token": "w", "new_id": 100}])
        self.assertEqual(report["vocab"]["removed"], [])
        self.assertEqual(report["vocab"]["changed"], [])

    def test_role_token_absent_from_vocab_reports_null_id(self):
        old_tokenizer = StubVocabTokenizer({}, roles={"bos_token": "<s>"})
        new_tokenizer = StubVocabTokenizer({"<s>": 0}, roles={"bos_token": "<s>"})

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(
            report["special_token_roles"],
            [
                {
                    "role": "bos_token",
                    "old_token": "<s>",
                    "old_id": None,
                    "new_token": "<s>",
                    "new_id": 0,
                }
            ],
        )

    def test_vocab_dict_traversal_order_differs_across_backends(self):
        vocab = {token: index for index, token in enumerate(self.slow_vocab_tokens)}
        slow_tokenizer = self._make_slow_tokenizer()
        fast_tokenizer = self._make_fast_tokenizer(
            vocab,
            insertion_order=list(reversed(self.slow_vocab_tokens)),
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        # The two tokenizers really use different backends and iterate the same mapping in different orders.
        self.assertFalse(slow_tokenizer.is_fast)
        self.assertTrue(fast_tokenizer.is_fast)
        self.assertNotEqual(list(slow_tokenizer.get_vocab()), list(fast_tokenizer.get_vocab()))
        self.assertEqual(slow_tokenizer.get_vocab(), fast_tokenizer.get_vocab())

        report = slow_tokenizer.diff_vocab(fast_tokenizer)

        assert_empty_vocab_diff(self, report, extra_old=[], extra_new=[])

    def test_comparison_does_not_mutate_either_tokenizer(self):
        old_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0, "<s>": 1}, bos_token="<s>", extra_special_tokens=[])
        new_tokenizer = make_wordlevel_tokenizer({"[UNK]": 0, "<s>": 2}, bos_token="<s>", extra_special_tokens=[])
        old_snapshot = (
            dict(old_tokenizer.get_vocab()),
            dict(old_tokenizer.added_tokens_decoder),
            dict(old_tokenizer._special_tokens_map),
            list(old_tokenizer._extra_special_tokens),
        )
        new_snapshot = (
            dict(new_tokenizer.get_vocab()),
            dict(new_tokenizer.added_tokens_decoder),
            dict(new_tokenizer._special_tokens_map),
            list(new_tokenizer._extra_special_tokens),
        )

        report = old_tokenizer.diff_vocab(new_tokenizer)

        self.assertEqual(
            (
                dict(old_tokenizer.get_vocab()),
                dict(old_tokenizer.added_tokens_decoder),
                dict(old_tokenizer._special_tokens_map),
                list(old_tokenizer._extra_special_tokens),
            ),
            old_snapshot,
        )
        self.assertEqual(
            (
                dict(new_tokenizer.get_vocab()),
                dict(new_tokenizer.added_tokens_decoder),
                dict(new_tokenizer._special_tokens_map),
                list(new_tokenizer._extra_special_tokens),
            ),
            new_snapshot,
        )
        self.assertEqual(report["vocab"]["changed"], [{"token": "<s>", "old_id": 1, "new_id": 2}])

    def test_invalid_input_type(self):
        tokenizer = make_wordlevel_tokenizer({"[UNK]": 0})
        with self.assertRaises(TypeError):
            tokenizer.diff_vocab({"[UNK]": 0})
