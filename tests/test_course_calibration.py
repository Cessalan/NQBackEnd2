"""Exercise diagnostic request/response behavior without importing LLM clients."""
import ast
import asyncio
import contextlib
import io
import json
import pathlib
import types
import unittest

source = pathlib.Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
tree = ast.parse(source)
endpoint = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_diagnostic_quiz")
endpoint.decorator_list = []


class CalibrationEndpointTests(unittest.TestCase):
    def run_endpoint(self, questions, language="fr"):
        prompts = []

        class LLM:
            def __init__(self, **kwargs):
                pass

            async def ainvoke(self, prompt):
                prompts.append(prompt)
                return types.SimpleNamespace(content=json.dumps(questions))

        class HTTPError(Exception):
            def __init__(self, status_code, detail):
                self.status_code, self.detail = status_code, detail

        namespace = {
            "DiagnosticQuizRequest": types.SimpleNamespace,
            "ACTIVE_SESSIONS": {"test": types.SimpleNamespace(session=types.SimpleNamespace(
                file_insights={"slides.pdf": {"topics": ["Cardiac", "Renal", "Endocrine", "Assessment"], "concepts": ["fluid balance"]}},
                vectorstore=None,
            ))},
            "ChatOpenAI": LLM, "HTTPException": HTTPError, "json": json,
            "DIAGNOSTIC_QUESTION_COUNT": 6, "DIAGNOSTIC_DEEP_TOPICS": 3,
            "_language_for_prompt": lambda value: "French" if value == "fr" else "English",
        }
        exec(compile(ast.Module(body=[endpoint], type_ignores=[]), "calibration_endpoint", "exec"), namespace)
        request = types.SimpleNamespace(chat_id="test", language=language,
            priorityTopics=["Renal", "Endocrine", "Cardiac"], hardestTopics=["Assessment"])
        with contextlib.redirect_stdout(io.StringIO()):
            result = asyncio.run(namespace["generate_diagnostic_quiz"](request))
        return result, prompts[0]

    def question(self, **changes):
        return {"question": "Sample question", "options": ["A", "B", "C", "D"],
                "correctIndex": 0, "topic": "Renal", **changes}

    def test_course_order_and_language_reach_question_generation(self):
        result, prompt = self.run_endpoint([self.question()])
        self.assertEqual(result["focusTopics"], ["Renal", "Endocrine", "Cardiac"])
        self.assertIn("in French", prompt)
        self.assertIn("EXACT priority-topic labels", prompt)
        self.assertLess(prompt.index("- Renal"), prompt.index("- Cardiac"))

    def test_invalid_answers_never_reach_the_student(self):
        result, _ = self.run_endpoint([self.question(), self.question(correctIndex=True),
            self.question(correctIndex=-1), self.question(options=["A", None, "C", "D"]),
            self.question(topic="Unrelated"), None])
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["correctIndex"], 0)

    def test_non_array_response_returns_no_questions(self):
        result, _ = self.run_endpoint({"questions": [self.question()]})
        self.assertEqual(result["questions"], [])


if __name__ == "__main__":
    unittest.main()
