import json
import unittest
from unittest.mock import Mock, patch

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, score_prospect


class ProspectUpdateTest(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-39002"
        self.original_tech_stack = list(
            data_service.PROSPECTS[self.prospect_id]["tech_stack"]
        )
        data_service._PROFILES.clear()

    def tearDown(self):
        data_service.PROSPECTS[self.prospect_id]["tech_stack"] = self.original_tech_stack
        data_service._PROFILES.clear()

    def test_update_persists_through_profile_build_and_scoring(self):
        build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        self.assertEqual(
            data_service.fetch_tech_stack(self.prospect_id).count("Terraform"), 1
        )
        profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Terraform", profile["prospect_profile"]["tech_stack"])

        captured = Mock()
        captured.model_dump.return_value = {"score": 90}
        offering = data_service.get_offering("OFFER-10004")
        scoring_llm = Mock(invoke=Mock(return_value=captured))
        with patch("gtm_agent.gtm_agent._scoring_llm", scoring_llm):
            score_prospect.invoke(
                {"prospect_profile": profile["prospect_profile"], "offering": offering}
            )

        user_payload = json.loads(
            scoring_llm.invoke.call_args.args[0][1]["content"].split(
                "\n\nProspect profile:\n", 1
            )[1]
        )
        missing = [
            technology
            for technology in offering["required_tech_stack"]
            if technology not in user_payload["tech_stack"]
        ]
        self.assertNotIn("Terraform", missing)


if __name__ == "__main__":
    unittest.main()
