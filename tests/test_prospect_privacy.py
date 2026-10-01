import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service, gtm_agent


class ProspectPrivacyTest(unittest.TestCase):
    prospect_id = "TEST-PRIVACY"
    sensitive_values = (
        "billing_qualification",
        "tax_id",
        "date_of_birth",
        "card_on_file",
        "credit_check_ref",
        "541-88-2307",
        "1988-04-17",
        "4111111111111111",
        "cc-12345",
    )

    def setUp(self):
        self.record = {
            "prospect_id": self.prospect_id,
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "annual_revenue": 5000000,
            "enrichment_source": "test",
            "engagement_history": [{"event": "viewed"}],
            "account_details": {"industry": "software"},
            "tech_stack": ["Python"],
            "billing_qualification": {
                "tax_id": "541-88-2307",
                "date_of_birth": "1988-04-17",
                "card_on_file": "4111111111111111",
                "credit_check_ref": "cc-12345",
            },
        }
        data_service._PROFILES.clear()

    def tearDown(self):
        data_service._PROFILES.clear()

    def assert_no_sensitive_values(self, value):
        rendered = json.dumps(value)
        for sensitive_value in self.sensitive_values:
            self.assertNotIn(sensitive_value, rendered)

    def test_prospect_tools_exclude_billing_qualification(self):
        with patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record}):
            contact = gtm_agent.get_prospect.invoke({"prospect_id": self.prospect_id})
            fresh = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            fresh_snapshot = json.loads(json.dumps(fresh))

            data_service._PROFILES[self.prospect_id]["billing_qualification"] = self.record[
                "billing_qualification"
            ]
            cached = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

            offering = {
                "offering_id": "OFFER-TEST",
                "description": "Test offering",
                "required_tech_stack": ["Python"],
                "min_annual_revenue": 1000000,
            }
            with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
                scoring_llm.invoke.return_value.model_dump.return_value = {}
                gtm_agent.score_prospect.invoke(
                    {
                        "prospect_profile": {
                            **fresh["prospect_profile"],
                            "billing_qualification": self.record[
                                "billing_qualification"
                            ],
                        },
                        "offering": offering,
                    }
                )

            prompt = scoring_llm.invoke.call_args.args[0][1]["content"]

        self.assert_no_sensitive_values(contact)
        self.assert_no_sensitive_values(fresh_snapshot)
        self.assert_no_sensitive_values(cached)
        self.assert_no_sensitive_values(prompt)


if __name__ == "__main__":
    unittest.main()
