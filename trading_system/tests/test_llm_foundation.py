import logging
import unittest
from unittest.mock import patch

from trading_system.crew import agents, llm


def valid_config():
    return {
        "provider": "groq",
        "api_key_env": "GROQ_API_KEY",
        "groq_integration_verified": True,
        "model_assignments": {role: "test-only-model" for role in llm.ADVISORY_ROLES},
        "temperature": 0.0,
        "max_output_tokens": 128,
        "retry": {"max_attempts": 2, "initial_backoff_seconds": 0,
                  "max_backoff_seconds": 0},
        "debate": {"max_bullets": 3, "max_headlines": 3,
                   "max_recent_lessons": 3, "judge_validation_retries": 1},
        "llm_influence_cap": 0.15,
    }


class FakeFrameworkClient:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = 0

    def call(self, prompt):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class LLMFoundationTests(unittest.TestCase):
    def test_configuration_loads_from_yaml_with_unverified_placeholders(self):
        config = llm.load_llm_config()
        self.assertEqual(config["provider"], "groq")
        self.assertEqual(config["api_key_env"], "GROQ_API_KEY")
        self.assertFalse(config["groq_integration_verified"])
        self.assertTrue(all("[VERIFY" in value
                            for value in config["model_assignments"].values()))
        self.assertIsNone(config["max_output_tokens"])

    def test_missing_api_key_fails_before_client_factory(self):
        with self.assertRaisesRegex(llm.LLMConfigurationError, "GROQ_API_KEY"):
            llm.create_llm_client("news", config=valid_config(), environ={},
                                  llm_factory=lambda **kwargs: self.fail("factory called"))

    def test_api_key_is_read_from_environment_and_factory_is_injectable(self):
        captured = {}
        framework = FakeFrameworkClient()

        def factory(**kwargs):
            captured.update(kwargs)
            return framework

        client = llm.create_llm_client(
            "news", config=valid_config(), environ={"GROQ_API_KEY": "test-secret"},
            llm_factory=factory)
        self.assertIs(client.framework_client, framework)
        self.assertEqual(captured["api_key"], "test-secret")
        self.assertEqual(captured["model"], "groq/test-only-model")
        self.assertEqual(captured["temperature"], 0.0)
        self.assertEqual(captured["max_tokens"], 128)

    def test_unverified_provider_model_and_limit_fail_clearly(self):
        config = valid_config()
        config["groq_integration_verified"] = False
        with self.assertRaisesRegex(llm.LLMConfigurationError, r"\[VERIFY\]"):
            llm.create_llm_client("news", config=config,
                                  environ={"GROQ_API_KEY": "test-secret"},
                                  llm_factory=lambda **kwargs: self.fail("factory called"))

        config = valid_config()
        config["model_assignments"]["news"] = "[VERIFY: approved model]"
        with self.assertRaisesRegex(llm.LLMConfigurationError, "unverified"):
            llm.create_llm_client("news", config=config,
                                  environ={"GROQ_API_KEY": "test-secret"})

        config = valid_config()
        config["max_output_tokens"] = None
        with self.assertRaisesRegex(llm.LLMConfigurationError, "max_output_tokens"):
            llm.create_llm_client("news", config=config,
                                  environ={"GROQ_API_KEY": "test-secret"})

    def test_deterministic_roles_cannot_construct_llm_clients(self):
        with self.assertRaisesRegex(llm.LLMConfigurationError, "deterministic"):
            llm.create_llm_client("scoring", config=valid_config(),
                                  environ={"GROQ_API_KEY": "test-secret"})

    def test_wrapper_construction_and_call_are_injectable_without_network(self):
        fake = FakeFrameworkClient([{"content": "advisory", "usage": {"total_tokens": 9}}])
        client = llm.LLMClient("news", "groq/test-model", valid_config(), fake)
        result = client.call_advisory("compact prompt")
        self.assertEqual(result["text"], "advisory")
        self.assertEqual(result["usage"], {"total_tokens": 9})
        self.assertEqual(fake.calls, 1)

    def test_provider_failure_retries_then_uses_code_fallback(self):
        fake = FakeFrameworkClient([RuntimeError("offline"), RuntimeError("offline")])
        client = llm.LLMClient("judge", "groq/test-model", valid_config(), fake, sleep=lambda _: None)
        result = client.call_advisory("compact", fallback=lambda: {"context": "none"})
        self.assertEqual(fake.calls, 2)
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["text"], {"context": "none"})
        self.assertTrue(result["warnings"])

    def test_secret_never_appears_in_logs_or_results(self):
        secret = "do-not-log-test-secret"
        fake = FakeFrameworkClient([RuntimeError(secret), RuntimeError(secret)])
        client = llm.LLMClient("news", "groq/test-model", valid_config(), fake, sleep=lambda _: None)
        with self.assertNoLogs("trading_system.crew.llm", level=logging.DEBUG):
            result = client.call_advisory("safe prompt")
        self.assertNotIn(secret, str(result))

    def test_invalid_configuration_fails_clearly(self):
        with self.assertRaisesRegex(llm.LLMConfigurationError, "provider"):
            llm.load_llm_config({"provider": "other"})


class AgentFoundationTests(unittest.TestCase):
    def test_all_roles_exist_and_data_scoring_risk_are_code_only(self):
        roles = agents.create_agents()
        self.assertEqual(set(roles), {"data", "scoring", "risk", "news", "bull", "bear",
                                      "judge", "reflection"})
        for key in ("data", "scoring", "risk"):
            self.assertTrue(roles[key].deterministic)
            self.assertIsNone(roles[key].llm_client)
        for key in ("news", "bull", "bear", "judge", "reflection"):
            self.assertFalse(roles[key].deterministic)

    def test_advisory_agent_uses_factory_and_disables_delegation(self):
        client = type("Client", (), {"framework_client": object()})()
        definitions = agents.create_agents(client_factory=lambda role, config: client)
        captured = {}
        made = agents.materialize_advisory_agent(
            definitions["news"], lambda **kwargs: captured.update(kwargs) or kwargs)
        self.assertEqual(made["role"], "News")
        self.assertIs(made["llm"], client.framework_client)
        self.assertFalse(made["allow_delegation"])

    def test_deterministic_role_cannot_be_materialized_as_llm_agent(self):
        with self.assertRaisesRegex(ValueError, "Only advisory"):
            agents.materialize_advisory_agent(agents.create_agents()["scoring"],
                                              lambda **kwargs: kwargs)

    def test_context_is_compact_and_accepts_m4_news_data(self):
        context = agents.build_compact_context(
            {"symbol": "AAPL", "composite": 0.4, "sub_scores": {"trend": 0.2},
             "ohlcv": [1, 2, 3]}, "trending",
            [{"title": "AAPL rises", "source": "M4 fixture"}, "second", "third", "extra"],
            ["lesson 1", "lesson 2", "lesson 3", "extra"], valid_config())
        self.assertNotIn("ohlcv", context["score_row"])
        self.assertEqual(len(context["headlines"]), 3)
        self.assertEqual(context["headlines"][0]["source"], "M4 fixture")
        self.assertEqual(len(context["recent_lessons"]), 3)

    def test_bull_bear_bullets_are_capped(self):
        self.assertEqual(len(agents.validate_debate_bullets("- One\n- Two\n- Three")), 3)
        with self.assertRaises(ValueError):
            agents.validate_debate_bullets("- One\n- Two\n- Three\n- Four")

    def test_judge_output_is_bounded_and_has_no_action_field(self):
        valid = agents.validate_judge_output(
            '{"adjustment":0.1,"reason":"Evidence mildly supports the score."}', 0.15)
        self.assertEqual(valid["adjustment"], 0.1)
        with self.assertRaises(ValueError):
            agents.validate_judge_output(
                '{"adjustment":0,"reason":"No change.","decision":"BUY"}', 0.15)
        with self.assertRaises(ValueError):
            agents.validate_judge_output(
                '{"adjustment":0.2,"reason":"Too high."}', 0.15)

    def test_malformed_judge_retries_once_then_fails_closed(self):
        responses = iter(["invalid", "still invalid"])
        prompts = []
        result = agents.get_judge_adjustment(
            lambda prompt: (prompts.append(prompt), next(responses))[1],
            "compact prompt", config=valid_config())
        self.assertEqual(len(prompts), 2)
        self.assertEqual(result["adjustment"], 0.0)


if __name__ == "__main__":
    unittest.main()
