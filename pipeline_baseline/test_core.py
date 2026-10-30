import unittest
from core import make_split, parse_action, tool_f1
from run import agent, input_only
from pathlib import Path
import time


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def generate(self, prompt, deadline):
        self.prompts.append(prompt)
        return next(self.replies)


class FakeEnv:
    def __init__(self):
        self.calls = []

    def validate(self, name, arguments):
        return {"ok": "id" in arguments, "error": "missing id"}

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": True}


class Tests(unittest.TestCase):
    def test_duplicate_groups_do_not_leak(self):
        rows = [{"id": str(i), "audio_text": str(i // 2), "history": [],
                 "label": {"tool_calls": []}} for i in range(40)]
        split = make_split(rows)
        self.assertEqual(split, make_split(rows))
        val = set(split["validation"])
        self.assertTrue(val)
        for i in range(0, 40, 2):
            self.assertEqual(str(i) in val, str(i + 1) in val)

    def test_maximum_matching_not_greedy(self):
        tools = {"x": {"parameters": {"properties": {"text": {"x-match": "fuzzy"}}}}}
        def c(text): return {"name": "x", "arguments": {"text": text}}
        # First prediction matches both gold calls; second only the first.
        self.assertEqual(tool_f1([c("a b"), c("a")], [c("a"), c("b")], tools), 1)
        self.assertAlmostEqual(tool_f1([c("a"), c("a")], [c("a")], tools), 2 / 3)

    def test_label_and_oracle_transcripts_are_not_leaked(self):
        row = {"context": {}, "history": [{"role": "user", "audio": "past.wav", "audio_text": "SECRET"}],
               "audio": "now.wav", "audio_text": "SECRET", "label": {"secret": 1}}
        class ASR:
            def transcribe(self, path, deadline): return "heard"
        item = input_only(row, Path("."), "train", ASR(), False, float("inf"))
        self.assertNotIn("SECRET", str(item))
        self.assertNotIn("label", item)

    def test_invalid_call_is_not_executed(self):
        model = FakeModel(['{"action":"inspect","names":["x"]}',
                           '{"action":"call","name":"x","arguments":{}}',
                           '{"action":"call","name":"x","arguments":{"id":"1"}}',
                           '{"action":"finish"}'])
        env = FakeEnv()
        events, reason = agent({}, {"x": {"name": "x", "kind": "write"}}, env, model, time.perf_counter()+10)
        self.assertEqual(reason, "finish")
        self.assertEqual(env.calls, [("x", {"id": "1"})])
        self.assertIn("missing id", str(events))

    def test_unknown_tool_never_executes(self):
        model = FakeModel(['{"action":"call","name":"bad","arguments":{}}'] * 2)
        env = FakeEnv()
        _, reason = agent({}, {}, env, model, time.perf_counter()+10)
        self.assertEqual(reason, "invalid_output")
        self.assertEqual(env.calls, [])

    def test_strict_parser(self):
        self.assertEqual(parse_action('```json\n{"action":"finish"}\n```')["action"], "finish")
        with self.assertRaises(ValueError):
            parse_action('Explanation {"action":"finish"}')


if __name__ == "__main__":
    unittest.main()
