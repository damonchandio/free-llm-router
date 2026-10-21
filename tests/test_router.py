import json
import unittest
from unittest import mock

from free_llm_router import AIError, Router
import free_llm_router as flr


class Resp:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = text if text is not None else json.dumps(body or {})

    def json(self):
        return self._body


def answer(text):
    return Resp(body={"choices": [{"message": {"content": text}}]})


def router(**kw):
    return Router(**{"groq_key": "g", "gemini_key": "k", **kw})


class PlanTest(unittest.TestCase):
    def names(self, plan):
        return [b for b, _ in plan]

    def test_order(self):
        r = router()
        self.assertEqual(self.names(r.plan()), ["groq", "gemini", "local"])
        self.assertEqual(self.names(r.plan(quality=True)), ["gemini", "groq", "gemini", "local"])

    def test_private_never_plans_gemini(self):
        self.assertEqual(self.names(router().plan(quality=True, private=True)), ["groq", "local"])

    def test_web_only_uses_a_searching_model(self):
        r = router()
        self.assertEqual(r.plan(web=True), [("groq", r.groq_web_model)])
        self.assertEqual(Router(groq_key="", gemini_key="k").plan(web=True), [])

    def test_private_pictures_need_opt_in(self):
        self.assertEqual(router().plan(images=True, private=True), [])
        self.assertEqual(self.names(router(vision_on_gemini=True).plan(images=True, private=True)), ["gemini"])

    def test_no_keys_means_local_only(self):
        self.assertEqual(self.names(Router(groq_key="", gemini_key="").plan()), ["local"])


class AskTest(unittest.TestCase):
    def test_busy_backend_is_rested_and_skipped(self):
        r, urls = router(), []

        def post(url, json=None, **kw):
            urls.append(url)
            if "groq" in url:
                return Resp(429, headers={"retry-after": "30"}, text="rate limit: tokens per minute")
            return answer("from gemini")

        with mock.patch.object(flr.requests, "post", post):
            self.assertEqual(r.ask("s", "p"), "from gemini")
            self.assertFalse(r.usable(f"groq:{r.groq_model}"))
            urls.clear()
            r.ask("s", "p")
        self.assertTrue(all("groq" not in u for u in urls))

    def test_daily_limit_rests_an_hour_and_bad_key_a_day(self):
        r = router()
        r.rest_for("a", Resp(429, text="Rate limit reached: requests per day (RPD)"))
        r.rest_for("b", Resp(401, text="invalid key"))
        now = flr.time.monotonic()
        self.assertGreater(r._rest["a"] - now, 3500)
        self.assertGreater(r._rest["b"] - now, 23 * 3600)

    def test_private_never_reaches_gemini(self):
        r, urls = router(), []

        def post(url, json=None, **kw):
            urls.append(url)
            return Resp(500, text="down")

        with mock.patch.object(flr.requests, "post", post):
            with self.assertRaises(AIError):
                r.ask("s", "my emails", private=True)
        self.assertTrue(urls and all("googleapis" not in u for u in urls))

    def test_local_false_skips_ollama(self):
        r = router()
        with mock.patch.object(flr.requests, "post", return_value=Resp(500, text="down")) as post:
            with self.assertRaises(AIError):
                r.ask("s", "p", local=False)
        self.assertTrue(all("/api/chat" not in c.args[0] for c in post.call_args_list))

    def test_falls_back_to_local(self):
        r = Router(groq_key="", gemini_key="")
        with mock.patch.object(flr.requests, "post", return_value=Resp(body={"message": {"content": "<think>x</think>hi"}})):
            self.assertEqual(r.ask("s", "p"), "hi")

    def test_web_sends_the_search_tool(self):
        r = router()
        with mock.patch.object(flr.requests, "post", return_value=answer("found it")) as post:
            r.ask("s", "news?", web=True)
        self.assertEqual(post.call_args.kwargs["json"]["tools"], [{"type": "browser_search"}])

    def test_ask_json(self):
        with mock.patch.object(flr.requests, "post", return_value=answer('```json\n{"a": 1}\n```')):
            self.assertEqual(router().ask_json("s", "p"), {"a": 1})

    def test_status(self):
        self.assertIn("Groq: no GROQ_API_KEY", Router(groq_key="", gemini_key="").status())


if __name__ == "__main__":
    unittest.main()
