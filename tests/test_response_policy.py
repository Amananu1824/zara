import unittest

from app.ai.chat_engine import ChatEngine
from app.ai.response_policy import (
    is_personal_fact_question,
    select_response_policy,
    should_retrieve_guidance
)
from app.terminal.response_rules import (
    detect_output_constraint,
    get_output_limit,
)


class ProfileResponseRuleTests(unittest.TestCase):
    def test_profile_question_uses_concise_output_limit(self):
        query = "aap mere bare me kya kya janti ho zara?"

        constraint = detect_output_constraint(query)

        self.assertEqual(constraint, "profile_summary")
        self.assertEqual(get_output_limit(constraint), 160)

    def test_explicit_detailed_request_keeps_detailed_format(self):
        query = "Aap mere bare me detailed batao."

        self.assertEqual(
            detect_output_constraint(query),
            "detailed",
        )
        self.assertEqual(
            detect_output_constraint("Python mein ek function bana do."),
            "code_generation",
        )
        self.assertEqual(
            get_output_limit("code_generation"),
            1400,
        )
        self.assertEqual(
            get_output_limit("normal"),
            320,
        )

    def test_profile_query_includes_all_saved_fields(self):
        entries = [
            {"memory_key": "name", "content": "Aman Gupta"},
            {"memory_key": "education", "content": "Class 12"},
            {
                "memory_key": "current_learning",
                "content": "Python and AI",
            },
            {
                "memory_key": "long_term_goals",
                "content": "Build a business",
            },
        ]

        context = ChatEngine._select_profile_context(
            entries,
            "aap mere bare me kya kya janti ho zara?",
        )

        combined = "\n".join(context)
        self.assertIn("Aman Gupta", combined)
        self.assertIn("Class 12", combined)
        self.assertIn("Python and AI", combined)
        self.assertIn("Build a business", combined)

    def test_regular_question_uses_relevant_context_and_custom_fields(self):
        entries = [
            {"memory_key": "name", "content": "Aman Gupta"},
            {
                "memory_key": "communication_style",
                "content": "Natural Roman Hinglish",
            },
            {"memory_key": "education", "content": "Class 12"},
            {
                "memory_key": "current_learning",
                "content": "Python and AI",
            },
            {
                "memory_key": "custom_note",
                "content": "Preserve current project context",
            },
        ]

        context = ChatEngine._select_profile_context(
            entries,
            "Python code kaise likhu?",
        )

        combined = "\n".join(context)
        self.assertIn("Natural Roman Hinglish", combined)
        self.assertIn("Python and AI", combined)
        self.assertIn("Preserve current project context", combined)
        self.assertNotIn("Class 12", combined)

class EmotionResponsePolicyTests(unittest.TestCase):
    
    def test_positive_and_negative_emotions_activate_policy(self):
        messages = [
            "Mera code chal gaya, main bahut khush hoon.",
            "Do ghante ka bug do minute mein fix ho gaya 😂",
            "Main baar-baar try karke irritate ho raha hoon.",
            "Mehnat ki lekin result nahi mila; bura lag raha hai.",
        ]

        for message in messages:
            with self.subTest(message=message):
                decision = select_response_policy(
                    message,
                    turn_language="hinglish",
                )

                self.assertIn(
                    "emotion.acknowledgement",
                    decision.categories,
                )

                instructions = " ".join(
                    decision.instructions
                ).casefold()

                self.assertIn(
                    "react to that moment first",
                    instructions,
                )
                self.assertIn(
                    "do not turn it into advice",
                    instructions,
                )
                self.assertIn(
                    "vary wording naturally",
                    instructions,
                )


class ProfileQuestionClassifierTests(unittest.TestCase):
    def test_direct_personal_fact_question_is_detected(self):
        self.assertTrue(
            is_personal_fact_question(
                "me kaun si class me hu zara?"
            )
        )

        self.assertTrue(
            is_personal_fact_question(
                "Mere saved profile se meri padhai, current goal "
                "aur learning interests batao."
            )
        )

        self.assertFalse(
            is_personal_fact_question(
                "Mera Python code finally bina error ke chal gaya!"
            )
        )

        decision = select_response_policy(
            "me kaun si class me hu zara?",
            turn_language="hinglish",
            has_user_profile=True,
        )

        self.assertIn(
            "profile.personal_fact",
            decision.categories,
        )

        combined = " ".join(decision.instructions).casefold()
        self.assertIn("do not add", combined)

    def test_retrieval_policy_separates_social_and_task_intents(self):
        self.assertFalse(
            should_retrieve_guidance(
                "Mera Python code finally chal gaya, main khush hoon!"
            )
        )
        self.assertFalse(
            should_retrieve_guidance(
                "Zara, tumhara friend kaun hai?"
            )
        )
        self.assertFalse(
            should_retrieve_guidance("Hello Zara!")
        )

        self.assertTrue(
            should_retrieve_guidance(
                "Python mein function ka example do."
            )
        )
        self.assertTrue(
            should_retrieve_guidance(
                "Mera Python error debug karne mein help karo."
            )
        )

class ContextIntentPolicyTests(unittest.TestCase):
    def test_explicit_no_solution_request_activates_listening_policy(self):
        decision = select_response_policy(
            "Bas mann halka karna tha, solution nahi chahiye.",
            turn_language="hinglish",
        )

        self.assertIn(
            "conversation.listening_only",
            decision.categories,
        )

        instructions = " ".join(decision.instructions).casefold()

        self.assertIn("do not give advice", instructions)
        self.assertIn("follow-up questions", instructions)

    def test_achievement_without_happy_keyword_gets_policy(self):
        decision = select_response_policy(
            "Aaj mera project finally chal gaya!",
            turn_language="hinglish",
        )

        self.assertIn(
            "social.achievement_sharing",
            decision.categories, 
        )

        instructions = " ".join(decision.instructions).casefold()

        self.assertIn("successful result or completed effort", instructions)
        self.assertIn("recognize the achievement", instructions)
        self.assertIn("do not add advice", instructions)


if __name__ == "__main__":
    unittest.main