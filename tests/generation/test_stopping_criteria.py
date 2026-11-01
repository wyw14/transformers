# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ..test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers.generation import (
        ConfidenceCriteria,
        EosTokenCriteria,
        MaxLengthCriteria,
        MaxTimeCriteria,
        StoppingCriteriaList,
        StopStringCriteria,
        SurpriseBudgetCriteria,
        validate_stopping_criteria,
    )
    from transformers.generation.stopping_criteria import STOP_STRING_EMBEDDING_CACHE


@require_torch
class StoppingCriteriaTestCase(unittest.TestCase):
    def _get_tensors(self, length):
        batch_size = 3
        vocab_size = 250

        input_ids = ids_tensor((batch_size, length), vocab_size)
        scores = torch.ones((batch_size, length), device=torch_device, dtype=torch.float) / length
        return input_ids, scores

    def _assert_isolated_token_decode_loses_stop_string(self, tokenizer, text, stop_string):
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        isolated_text = "".join(tokenizer.convert_tokens_to_string([token]) for token in tokens)

        self.assertTrue(tokenizer.decode(input_ids, skip_special_tokens=False).endswith(stop_string))
        self.assertNotIn(stop_string, isolated_text)

    def test_list_criteria(self):
        input_ids, scores = self._get_tensors(5)

        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=10),
                MaxTimeCriteria(max_time=0.1),
            ]
        )

        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(9)
        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(10)
        self.assertTrue(all(criteria(input_ids, scores)))

    def test_max_length_criteria(self):
        criteria = MaxLengthCriteria(max_length=10)

        input_ids, scores = self._get_tensors(5)
        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(9)
        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(10)
        self.assertTrue(all(criteria(input_ids, scores)))

    def test_max_time_criteria(self):
        input_ids, scores = self._get_tensors(5)

        criteria = MaxTimeCriteria(max_time=0.1)
        self.assertFalse(all(criteria(input_ids, scores)))

        criteria = MaxTimeCriteria(max_time=0.1, initial_timestamp=time.time() - 0.2)
        self.assertTrue(all(criteria(input_ids, scores)))

    def test_eos_token_criteria(self):
        criteria = EosTokenCriteria(eos_token_id=0)

        input_ids, scores = self._get_tensors(5)
        input_ids[:, -1] = 0
        self.assertTrue(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(5)
        input_ids[:2, -1] = 0
        input_ids[2, -1] = 1
        self.assertListEqual(criteria(input_ids, scores).tolist(), [True, True, False])

        input_ids, scores = self._get_tensors(5)
        input_ids[:, -1] = 1
        self.assertListEqual(criteria(input_ids, scores).tolist(), [False, False, False])

    def test_confidence_criteria(self):
        criteria = ConfidenceCriteria(assistant_confidence_threshold=0.5)

        vocab_size = 250
        length = 5

        input_ids = ids_tensor((1, length), vocab_size)
        scores = (torch.randn((1, vocab_size)),)

        # Simulate high confidence by setting the probability of the last token to be high
        scores[0][0, input_ids[0, -1]] = 10.0  # Logits before softmax
        self.assertFalse(criteria(input_ids, scores))

        # Simulate low confidence by setting the probability of the last token to be low
        scores[0][0, input_ids[0, -1]] = -10.0  # Logits before softmax
        self.assertTrue(criteria(input_ids, scores))

    def test_validate_stopping_criteria(self):
        validate_stopping_criteria(StoppingCriteriaList([MaxLengthCriteria(10)]), 10)

        with self.assertWarns(UserWarning):
            validate_stopping_criteria(StoppingCriteriaList([MaxLengthCriteria(10)]), 11)

        stopping_criteria = validate_stopping_criteria(StoppingCriteriaList(), 11)

        self.assertEqual(len(stopping_criteria), 1)

    def test_stop_string_criteria(self):
        true_strings = [
            "<|im_start|><|im_end|>",
            "<|im_start|><|im_end|<|im_end|>",
            ">><|im_start|>>stop",
            "stop",
            "e nd",
        ]
        false_strings = [
            "<|im_start|><|im_end|",
            "<|im_start|><|im_end|<|im_end|",
            "<|im_end|><|im_start|>",
            "<|im_end|<>stop<|im_end|",
            "end",
            "en d",
            "eNd",
            "<|im_end|",
            "|im_end|>",
            "s",
        ]
        stop_strings = ["<|im_end|>", "stop", "e nd"]

        # Use a tokenizer that won't actually have special tokens for these
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        true_input_ids = tokenizer(true_strings, return_tensors="pt", padding="longest", add_special_tokens=False)
        false_input_ids = tokenizer(false_strings, return_tensors="pt", padding="longest", add_special_tokens=False)

        scores = None
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)
        for i in range(len(true_strings)):
            self.assertTrue(criteria(true_input_ids["input_ids"][i : i + 1], scores))
        for i in range(len(false_strings)):
            self.assertFalse(criteria(false_input_ids["input_ids"][i : i + 1], scores))

        # Now try it with a tokenizer where those are actually special tokens
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        tokenizer.padding_side = "left"
        true_input_ids = tokenizer(true_strings, return_tensors="pt", padding="longest", add_special_tokens=False)
        false_input_ids = tokenizer(false_strings, return_tensors="pt", padding="longest", add_special_tokens=False)

        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)
        for i in range(len(true_strings)):
            self.assertTrue(criteria(true_input_ids["input_ids"][i : i + 1], scores))
        for i in range(len(false_strings)):
            self.assertFalse(criteria(false_input_ids["input_ids"][i : i + 1], scores))

    def test_stop_string_criteria_byte_fragments(self):
        STOP_STRING_EMBEDDING_CACHE.clear()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        self.assertEqual(StopStringCriteria._get_stop_string_matching_mode(tokenizer), "byte_level")
        self._assert_isolated_token_decode_loses_stop_string(tokenizer, "대화 끝", "끝")
        self._assert_isolated_token_decode_loses_stop_string(tokenizer, "작업 완료", "완료")

        cases = [
            ("대화 끝", "끝", True),
            ("작업 완료", "완료", True),
            ("대화 끝 다음", "끝", False),
        ]

        for text, stop_string, expected in cases:
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=[stop_string])
            self.assertEqual(bool(criteria(input_ids, scores=None)[0]), expected)

    def test_stop_string_criteria_byte_fallback_fragments(self):
        STOP_STRING_EMBEDDING_CACHE.clear()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", use_fast=True)
        self.assertEqual(StopStringCriteria._get_stop_string_matching_mode(tokenizer), "byte_fallback")
        self._assert_isolated_token_decode_loses_stop_string(tokenizer, "대화 끝", "끝")
        self._assert_isolated_token_decode_loses_stop_string(tokenizer, "abc 끝!", "끝!")

        cases = [
            ("대화 끝", "끝", True),
            ("abc 끝!", "끝!", True),
            ("대화 끝 다음", "끝", False),
            ("완료 후속", "완료", False),
        ]

        for text, stop_string, expected in cases:
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=[stop_string])
            self.assertEqual(bool(criteria(input_ids, scores=None)[0]), expected)

    def test_stop_string_criteria_byte_fragment_compile(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is not available")

        STOP_STRING_EMBEDDING_CACHE.clear()
        cases = [
            ("Qwen/Qwen2-0.5B-Instruct", "대화 끝", "끝"),
            ("hf-internal-testing/llama-tokenizer", "abc 끝!", "끝!"),
        ]
        for tokenizer_name, text, stop_string in cases:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=[stop_string])
            compiled_criteria = torch.compile(criteria, backend="eager", fullgraph=True)
            self.assertTrue(bool(compiled_criteria(input_ids, scores=None)[0]))

    def test_stop_string_criteria_byte_level_ascii(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.assertEqual(StopStringCriteria._get_stop_string_matching_mode(tokenizer), "byte_level")

        true_input_ids = tokenizer("the end", return_tensors="pt", add_special_tokens=False)["input_ids"]
        false_input_ids = tokenizer("end of", return_tensors="pt", add_special_tokens=False)["input_ids"]
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=["end"])
        self.assertTrue(bool(criteria(true_input_ids, scores=None)[0]))
        self.assertFalse(bool(criteria(false_input_ids, scores=None)[0]))

    def test_stop_string_criteria_non_byte_level_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.assertIsNone(StopStringCriteria._get_stop_string_matching_mode(tokenizer))

        true_input_ids = tokenizer("the end", return_tensors="pt", add_special_tokens=False)["input_ids"]
        false_input_ids = tokenizer("end of", return_tensors="pt", add_special_tokens=False)["input_ids"]
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=["end"])
        self.assertTrue(bool(criteria(true_input_ids, scores=None)[0]))
        self.assertFalse(bool(criteria(false_input_ids, scores=None)[0]))

    def test_stop_string_matching_mode_helpers(self):
        class Decoder:
            def __init__(self, state):
                self.state = state

            def __getstate__(self):
                return self.state

        class BackendTokenizer:
            def __init__(self, decoder):
                self.decoder = decoder

        class Tokenizer:
            def __init__(self, decoder):
                self.backend_tokenizer = BackendTokenizer(decoder)

        self.assertEqual(
            StopStringCriteria._get_stop_string_matching_mode(
                Tokenizer(Decoder(b'{"type":"Sequence","decoders":[{"type":"ByteLevel"}]}'))
            ),
            "byte_level",
        )
        self.assertEqual(
            StopStringCriteria._get_stop_string_matching_mode(
                Tokenizer(Decoder(b'{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"ByteLevel"}]}'))
            ),
            "byte_fallback",
        )
        self.assertIsNone(
            StopStringCriteria._get_stop_string_matching_mode(
                Tokenizer(Decoder(b'{"type":"Replace","content":"ByteFallback"}'))
            )
        )

        self.assertEqual(StopStringCriteria._token_to_bytes("<0xEB>", "byte_fallback", None), b"\xeb")
        self.assertEqual(StopStringCriteria._token_to_bytes("<0xeb>", "byte_fallback", None), b"\xeb")
        for token in ["<0x+1>", "<0xG1>", "<0x 1>", "<0x1>", "<0x100>", "<0xeb", "hello"]:
            self.assertIsNone(StopStringCriteria._token_to_bytes(token, "byte_fallback", None))

    def test_stop_string_criteria_vocab_size_mismatch(self):
        """Test that StopStringCriteria handles tokens above len(tokenizer) correctly."""
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        # Create input_ids with tokens above len(tokenizer)
        input_ids = torch.tensor([[len(tokenizer) + 1024, 1, 2]], device=torch_device)
        scores = None
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=["test"])

        # This should not raise an error and should return False since no stop string is matched
        self.assertFalse(criteria(input_ids, scores))

    def test_stop_string_matching_positions(self):
        stop_string = "stop"
        token_list = ["last", "top", "topper", "s", "p"]
        token_indices = list(range(len(token_list)))
        all_token_valid_positions, all_token_end_overlaps = StopStringCriteria._stop_string_get_matching_positions(
            token_list=token_list, token_indices=token_indices, stop_strings=[stop_string]
        )
        valid_positions = {
            token_list[idx]: positions for idx, positions in all_token_valid_positions[stop_string].items()
        }
        end_overlaps = {token_list[idx]: overlaps for idx, overlaps in all_token_end_overlaps[stop_string].items()}
        self.assertEqual(valid_positions, {"s": [3], "last": [2]})
        self.assertEqual(end_overlaps, {"top": [3], "topper": [3], "p": [1]})

    def test_stop_string_embedding_vecs(self):
        stop_string = "stop"
        token_list = ["last", "top", "topper", "s", "p"]
        token_indices = list(range(len(token_list)))
        embedding_vec, max_valid_positions, max_valid_end_lens = StopStringCriteria._stop_string_create_embedding_vec(
            token_list=token_list, token_indices=token_indices, stop_strings=[stop_string]
        )

        # Positions inside the stop string where the token matches (excluding end overlaps)
        valid_positions = embedding_vec[:, 0].tolist()
        self.assertEqual(valid_positions, [2, -1, -1, 3, -1, -1])

        # Overlap lengths between end of stop string and start of token
        end_overlaps = embedding_vec[:, 1].tolist()
        self.assertEqual(end_overlaps, [-1, 3, 3, -1, 1, -1])

        # Length of each token
        token_lengths = embedding_vec[:-1, 2].tolist()
        self.assertEqual(token_lengths, [len(token) for token in token_list])

    def test_single_letter_stop_string(self):
        true_strings = ["a", "baa", "abc"]  # "abc" is a single token
        false_strings = ["abbbbbbb", "b"]  # "abbbbbbb" is split into multiple tokens
        stop_strings = ["a"]
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=False)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        true_input_ids = tokenizer(true_strings, return_tensors="pt", padding="longest", add_special_tokens=False)
        false_input_ids = tokenizer(false_strings, return_tensors="pt", padding="longest", add_special_tokens=False)

        scores = None
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)
        for input_ids in true_input_ids["input_ids"]:
            self.assertTrue(criteria(input_ids.unsqueeze(0), scores))
        for input_ids in false_input_ids["input_ids"]:
            self.assertFalse(criteria(input_ids.unsqueeze(0), scores))

    def test_criteria_per_row(self):
        text = "They completed the challenging puzzle, revealing the hidden image at the end"
        stop_strings = ["end"]

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

        scores = None
        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=20),
                StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings),
            ]
        )

        # trigger stopping when at least one criteria is satisfied, one value per batch
        self.assertTrue(criteria(inputs["input_ids"], scores))

        # return False when neither is satisfied
        self.assertFalse(criteria(inputs["input_ids"][:, :-1], scores))

    def test_criteria_per_row_batched(self):
        text = [
            "They completed the challenging puzzle, revealing the hidden image at the end",
            "Today a dragon flew over France",
            "The aroma of freshly baked pizza filled the kitchen",
        ]
        stop_strings = ["end"]

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        inputs = tokenizer(text, return_tensors="pt", padding="longest", add_special_tokens=False)

        scores = None
        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=20),
                StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings),
            ]
        )

        # trigger stopping when at least one criteria is satisfied
        self.assertListEqual(criteria(inputs["input_ids"], scores).tolist(), [True, False, False])

        # False when neither is satisfied
        self.assertListEqual(criteria(inputs["input_ids"][:, :-1], scores).tolist(), [False, False, False])

    def _run_surprise_step(self, criteria, input_ids, scores, new_tokens, step_logits, unfinished=None):
        """Mimics one greedy/sampling generation step: append a token, record the step scores, call the criteria."""
        input_ids = torch.cat([input_ids, new_tokens[:, None].to(input_ids.device)], dim=-1)
        scores = scores + (step_logits,)
        kwargs = {}
        if unfinished is not None:
            kwargs["unfinished_sequences"] = unfinished
        is_done = criteria(input_ids, scores, **kwargs)
        return input_ids, scores, is_done

    def _probs_to_logits(self, probs):
        # `log(p)` are valid logits: softmax(log(p)) == p when the rows of `probs` sum to 1.
        return torch.log(probs.to(torch_device))

    def test_surprise_budget_criteria_invalid_args(self):
        for invalid_budget in (-1.0, float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(ValueError):
                SurpriseBudgetCriteria(surprise_budget=invalid_budget)
        for invalid_minimum in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                SurpriseBudgetCriteria(surprise_budget=1.0, minimum_new_tokens=invalid_minimum)

    def test_surprise_budget_criteria_cumulative_values(self):
        # Single row, 3-token prompt; fixed selection distributions for 4 generated tokens.
        vocab_size = 4
        prompt_length = 3
        input_ids = torch.randint(0, vocab_size, (1, prompt_length), device=torch_device)
        step_probs = [
            torch.tensor([[0.5, 0.5, 0.0, 0.0]]),
            torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
            torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        ]
        selected_tokens = [0, 1, 2, 3]
        expected_surprises = [-math.log([0.5, 0.25, 0.3, 0.4][i]) for i in range(4)]

        criteria = SurpriseBudgetCriteria(surprise_budget=100.0)  # never triggers here
        scores = ()
        for step, (probs, token) in enumerate(zip(step_probs, selected_tokens)):
            new_tokens = torch.tensor([token], device=torch_device)
            input_ids, scores, is_done = self._run_surprise_step(
                criteria, input_ids, scores, new_tokens, self._probs_to_logits(probs)
            )
            self.assertFalse(is_done.any())
            torch.testing.assert_close(
                criteria.cumulative_surprise.cpu(),
                torch.tensor([sum(expected_surprises[: step + 1])], dtype=torch.float32),
                rtol=1e-5,
                atol=1e-6,
            )

        # The prompt itself must never be counted: 4 score steps <=> 4 generated tokens.
        self.assertEqual(input_ids.shape[1] - prompt_length, len(scores))

    def test_surprise_budget_criteria_threshold_boundary(self):
        vocab_size = 4
        probs = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        per_token_surprise = float(-torch.log(torch.tensor(0.5)))

        # Budget exactly matching the first token's surprise: stop as soon as the budget is *reached*.
        criteria = SurpriseBudgetCriteria(surprise_budget=per_token_surprise)
        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        _, _, is_done = self._run_surprise_step(
            criteria, input_ids, (), torch.tensor([0]), self._probs_to_logits(probs)
        )
        self.assertTrue(bool(is_done[0]))

        # An epsilon of room: do not stop after one token, stop once the second token reaches the budget.
        criteria = SurpriseBudgetCriteria(surprise_budget=per_token_surprise + 1e-4)
        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        scores = ()
        input_ids, scores, is_done = self._run_surprise_step(
            criteria, input_ids, scores, torch.tensor([0]), self._probs_to_logits(probs)
        )
        self.assertFalse(bool(is_done[0]))
        _, _, is_done = self._run_surprise_step(
            criteria, input_ids, scores, torch.tensor([1]), self._probs_to_logits(probs)
        )
        self.assertTrue(bool(is_done[0]))

        # Zero budget with zero minimum tokens stops at the first generated token (surprise is non-negative).
        criteria = SurpriseBudgetCriteria(surprise_budget=0.0)
        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        _, _, is_done = self._run_surprise_step(
            criteria, input_ids, (), torch.tensor([0]), self._probs_to_logits(probs)
        )
        self.assertTrue(bool(is_done[0]))

    def test_surprise_budget_criteria_minimum_new_tokens(self):
        vocab_size = 4
        probs = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        per_token_surprise = float(-torch.log(torch.tensor(0.5)))
        criteria = SurpriseBudgetCriteria(surprise_budget=per_token_surprise - 1e-4, minimum_new_tokens=3)

        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        scores = ()
        done_per_step = []
        for _ in range(4):
            input_ids, scores, is_done = self._run_surprise_step(
                criteria, input_ids, scores, torch.tensor([0], device=torch_device), self._probs_to_logits(probs)
            )
            done_per_step.append(bool(is_done[0]))

        # Budget is reached on the first token, but stopping is deferred until `minimum_new_tokens` exist.
        self.assertListEqual(done_per_step, [False, False, True, True])
        # The row latches at step 3; the tokens appended afterwards are not accumulated.
        torch.testing.assert_close(
            criteria.cumulative_surprise.cpu(),
            torch.tensor([3 * per_token_surprise], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_surprise_budget_criteria_batched_independent(self):
        # 3 rows, each selecting tokens of fixed probability.
        row_probs = torch.tensor([0.5, 0.25, 0.1], device=torch_device)
        vocab_size = 6
        criteria = SurpriseBudgetCriteria(surprise_budget=2.0)

        input_ids = torch.randint(0, vocab_size, (3, 2), device=torch_device)
        scores = ()
        stop_steps = [None, None, None]
        for step in range(1, 6):
            step_logits = torch.zeros(3, vocab_size, device=torch_device)
            # Token 0 gets the row-specific probability, the remaining mass is spread uniformly.
            step_logits[:, 0] = torch.log(row_probs)
            step_logits[:, 1:] = torch.log((1 - row_probs) / (vocab_size - 1))[:, None]
            input_ids, scores, is_done = self._run_surprise_step(
                criteria, input_ids, scores, torch.zeros(3, dtype=torch.long, device=torch_device), step_logits
            )
            for row in range(3):
                if stop_steps[row] is None and bool(is_done[row]):
                    stop_steps[row] = step

        # Rows with lower-probability tokens (higher surprise) stop earlier, independently.
        self.assertListEqual(stop_steps, [3, 2, 1])

        # A stopped row must not accumulate the padding tokens appended afterwards: the cumulative value is the
        # value it had when the row stopped.
        frozen_cumulative = -torch.log(row_probs).cpu() * torch.tensor(stop_steps, dtype=torch.float32)
        torch.testing.assert_close(criteria.cumulative_surprise.cpu(), frozen_cumulative, rtol=1e-5, atol=1e-6)

    def test_surprise_budget_criteria_skips_externally_finished_rows(self):
        # Simulate row 0 hitting EOS before step 2: the liveness mask marks it inactive, and the padding token
        # appended at step 2 must not be accumulated nor flag the row through this criteria.
        vocab_size = 4
        probs = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
        criteria = SurpriseBudgetCriteria(surprise_budget=100.0)

        input_ids = torch.randint(0, vocab_size, (2, 1), device=torch_device)
        scores = ()
        input_ids, scores, _ = self._run_surprise_step(
            criteria,
            input_ids,
            scores,
            torch.tensor([0, 0]),
            self._probs_to_logits(probs),
            unfinished=torch.ones(2, dtype=torch.long),
        )
        cumulative_after_step_1 = criteria.cumulative_surprise.clone()

        # Padding tokens for the finished row 0 would look extremely "surprising" -- they still must be ignored.
        pad_logits = torch.full((2, vocab_size), -100.0, device=torch_device)
        input_ids, scores, is_done = self._run_surprise_step(
            criteria,
            input_ids,
            scores,
            torch.tensor([0, 1]),  # token appended to both rows; row 0 is padding in a real generation
            pad_logits,
            unfinished=torch.tensor([0, 1]),
        )
        self.assertFalse(bool(is_done[0]))
        torch.testing.assert_close(
            criteria.cumulative_surprise[0].cpu(), cumulative_after_step_1[0].cpu(), rtol=1e-5, atol=1e-6
        )
        self.assertGreater(float(criteria.cumulative_surprise[1]), float(cumulative_after_step_1[1]))

    def test_surprise_budget_criteria_stable_for_tiny_probabilities(self):
        vocab_size = 4
        # A probability so small that plain softmax underflows to zero, but log_softmax stays finite.
        step_logits = torch.tensor([[0.0, -1000.0, -1000.0, -1000.0]], device=torch_device)
        self.assertEqual(float(torch.softmax(step_logits, dim=-1)[0, 1]), 0.0)

        criteria = SurpriseBudgetCriteria(surprise_budget=10_000.0)
        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        _, _, is_done = self._run_surprise_step(criteria, input_ids, (), torch.tensor([1]), step_logits)
        self.assertFalse(bool(is_done[0]))
        self.assertTrue(torch.isfinite(criteria.cumulative_surprise).all())
        torch.testing.assert_close(criteria.cumulative_surprise.cpu(), torch.tensor([1000.0]), rtol=1e-6, atol=1e-4)

        # A token with exactly zero probability yields infinite surprise and stops immediately against a finite
        # budget, without producing NaNs.
        step_logits = torch.tensor([[0.0, float("-inf"), float("-inf"), float("-inf")]], device=torch_device)
        criteria = SurpriseBudgetCriteria(surprise_budget=1.0)
        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        _, _, is_done = self._run_surprise_step(criteria, input_ids, (), torch.tensor([1]), step_logits)
        self.assertTrue(bool(is_done[0]))
        self.assertTrue(torch.isinf(criteria.cumulative_surprise).all())
        self.assertFalse(torch.isnan(criteria.cumulative_surprise).any())

    def test_surprise_budget_criteria_degenerate_distribution_raises(self):
        vocab_size = 4
        step_logits = torch.full((1, vocab_size), float("-inf"), device=torch_device)
        criteria = SurpriseBudgetCriteria(surprise_budget=1.0)
        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        with self.assertRaises(ValueError):
            self._run_surprise_step(criteria, input_ids, (), torch.tensor([0]), step_logits)

    def test_surprise_budget_criteria_scores_required(self):
        criteria = SurpriseBudgetCriteria(surprise_budget=1.0)
        input_ids = torch.randint(0, 4, (1, 2), device=torch_device)
        with self.assertRaises(ValueError):
            criteria(input_ids, None)

    def test_surprise_budget_criteria_multi_token_step_raises(self):
        # Assisted generation can accept several candidate tokens in one iteration.
        vocab_size = 4
        criteria = SurpriseBudgetCriteria(surprise_budget=100.0)
        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        first_logits = torch.zeros(1, vocab_size, device=torch_device)
        input_ids, scores, _ = self._run_surprise_step(criteria, input_ids, (), torch.tensor([0]), first_logits)
        input_ids = torch.cat([input_ids, torch.tensor([[1, 2]], device=torch_device)], dim=-1)
        scores = scores + (first_logits, first_logits)
        with self.assertRaises(ValueError):
            criteria(input_ids, scores)

    def test_surprise_budget_criteria_reordered_history_raises(self):
        # Beam search can reorder/rewrite the rows between steps.
        vocab_size = 4
        criteria = SurpriseBudgetCriteria(surprise_budget=100.0)
        input_ids = torch.tensor([[3, 0]], device=torch_device)
        first_logits = torch.zeros(1, vocab_size, device=torch_device)
        input_ids, scores, _ = self._run_surprise_step(criteria, input_ids, (), torch.tensor([1]), first_logits)
        reordered = torch.tensor([[3, 2, 1]], device=torch_device)  # prefix token 0 was rewritten into 2
        with self.assertRaises(ValueError):
            criteria(reordered, scores + (first_logits,))

    def test_surprise_budget_criteria_misaligned_scores_raise(self):
        vocab_size = 4
        criteria = SurpriseBudgetCriteria(surprise_budget=100.0)
        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        first_logits = torch.zeros(1, vocab_size, device=torch_device)
        input_ids, scores, _ = self._run_surprise_step(
            criteria, input_ids, (), torch.tensor([0], device=torch_device), first_logits
        )
        # One more token is appended (so the history grows by exactly one), but the scores tuple has an extra
        # step that the generated history cannot be aligned with.
        input_ids = torch.cat([input_ids, torch.tensor([[1]], device=torch_device)], dim=-1)
        with self.assertRaises(ValueError):
            criteria(input_ids, scores + (first_logits, first_logits))

    def test_surprise_budget_criteria_batch_change_raises(self):
        vocab_size = 4
        criteria = SurpriseBudgetCriteria(surprise_budget=100.0)
        input_ids = torch.randint(0, vocab_size, (2, 1), device=torch_device)
        first_logits = torch.zeros(2, vocab_size, device=torch_device)
        self._run_surprise_step(criteria, input_ids, (), torch.tensor([0, 1]), first_logits)

        new_input_ids = torch.randint(0, vocab_size, (3, 1), device=torch_device)
        new_logits = torch.zeros(3, vocab_size, device=torch_device)
        with self.assertRaises(ValueError):
            self._run_surprise_step(criteria, new_input_ids, (), torch.tensor([0, 1, 2]), new_logits)

    def test_surprise_budget_criteria_reset(self):
        vocab_size = 4
        criteria = SurpriseBudgetCriteria(surprise_budget=100.0)
        self.assertIsNone(criteria.cumulative_surprise)

        input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        first_logits = torch.zeros(1, vocab_size, device=torch_device)
        self._run_surprise_step(criteria, input_ids, (), torch.tensor([0]), first_logits)
        self.assertIsNotNone(criteria.cumulative_surprise)

        # Reusing the instance for a new short session without resetting fails on the history check.
        fresh_input_ids = torch.randint(0, vocab_size, (1, 1), device=torch_device)
        with self.assertRaises(ValueError):
            self._run_surprise_step(criteria, fresh_input_ids, (), torch.tensor([0]), first_logits)

        # After `reset`, a new session starts from a clean state.
        criteria.reset()
        self.assertIsNone(criteria.cumulative_surprise)
        self._run_surprise_step(criteria, fresh_input_ids, (), torch.tensor([0]), first_logits)
        self.assertIsNotNone(criteria.cumulative_surprise)

    def test_surprise_budget_criteria_in_generate_greedy(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = LlamaForCausalLM(config).to(torch_device).eval()
        model.generation_config.eos_token_id = None
        model.generation_config.pad_token_id = None
        input_ids = torch.randint(1, config.vocab_size, (1, 3), device=torch_device)

        # First pass: let greedy decoding run to the maximum length and record the cumulative surprise series.
        full_output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=[SurpriseBudgetCriteria(surprise_budget=1e9)],
        )
        generated = full_output.sequences[0, input_ids.shape[1] :]
        cumulative = []
        running = 0.0
        for step, token in enumerate(generated):
            log_prob = torch.log_softmax(full_output.scores[step][0].to(torch.float32), dim=-1)[token]
            running += float(-log_prob)
            cumulative.append(running)
        self.assertEqual(len(cumulative), 20)

        # Second pass: set the budget exactly to the cumulative value after an interior token. Greedy decoding is
        # deterministic, so generation must stop at exactly that step (budget is reached with `>=`).
        stop_step = 4
        budget = cumulative[stop_step - 1]
        self.assertLess(cumulative[stop_step - 2], budget)  # the previous step must still be below the budget

        criteria = SurpriseBudgetCriteria(surprise_budget=budget)
        output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=[criteria],
        )
        self.assertEqual(output.sequences.shape[1], input_ids.shape[1] + stop_step)
        self.assertEqual(len(output.scores), stop_step)
        torch.testing.assert_close(
            criteria.cumulative_surprise.cpu(),
            torch.tensor([budget], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_surprise_budget_criteria_in_generate_unsupported_modes(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = LlamaForCausalLM(config).to(torch_device).eval()
        model.generation_config.pad_token_id = None
        input_ids = torch.randint(1, config.vocab_size, (1, 3), device=torch_device)

        # Beam search is not supported.
        with self.assertRaises(ValueError):
            model.generate(
                input_ids,
                num_beams=2,
                max_new_tokens=5,
                output_scores=True,
                return_dict_in_generate=True,
                stopping_criteria=[SurpriseBudgetCriteria(surprise_budget=1.0)],
            )

        # Assisted generation is not supported.
        with self.assertRaises(ValueError):
            model.generate(
                input_ids,
                max_new_tokens=5,
                output_scores=True,
                return_dict_in_generate=True,
                assistant_model=model,
                stopping_criteria=[SurpriseBudgetCriteria(surprise_budget=1.0)],
            )

        # Per-step scores must be enabled (`output_scores` is ignored without `return_dict_in_generate`).
        with self.assertRaises(ValueError):
            model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=5,
                stopping_criteria=[SurpriseBudgetCriteria(surprise_budget=1.0)],
            )
        with self.assertRaises(ValueError):
            model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=5,
                output_scores=True,
                stopping_criteria=[SurpriseBudgetCriteria(surprise_budget=1.0)],
            )

    def test_surprise_budget_criteria_in_generate_sampling_and_reset(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = LlamaForCausalLM(config).to(torch_device).eval()
        model.generation_config.eos_token_id = None
        model.generation_config.pad_token_id = None
        input_ids = torch.randint(1, config.vocab_size, (1, 3), device=torch_device)

        # First sampled pass (with a fixed RNG seed) records the cumulative surprise series.
        torch.manual_seed(123)
        full_output = model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=[SurpriseBudgetCriteria(surprise_budget=1e9)],
        )
        generated = full_output.sequences[0, input_ids.shape[1] :]
        cumulative = []
        running = 0.0
        for step, token in enumerate(generated):
            log_prob = torch.log_softmax(full_output.scores[step][0].to(torch.float32), dim=-1)[token]
            running += float(-log_prob)
            cumulative.append(running)

        # Second pass with the same RNG seed: sampling reproduces the same tokens, and the exact budget boundary
        # stops it at the expected step, while `minimum_new_tokens` is respected.
        stop_step = 3
        budget = cumulative[stop_step - 1]
        criteria = SurpriseBudgetCriteria(surprise_budget=budget, minimum_new_tokens=2)
        torch.manual_seed(123)
        output = model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=[criteria],
        )
        self.assertEqual(output.sequences.shape[1], input_ids.shape[1] + stop_step)
        self.assertTrue(torch.all(criteria.cumulative_surprise >= budget))

        # The same instance can serve a new session after `reset`.
        criteria.reset()
        output_again = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=[criteria],
        )
        self.assertLessEqual(output_again.sequences.shape[1] - input_ids.shape[1], 20)
