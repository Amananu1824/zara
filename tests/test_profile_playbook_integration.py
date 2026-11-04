import unittest
from unittest.mock import MagicMock, patch

from app.ai.chat_engine import ChatEngine
from app.learning_ai.learning_store import LearningStore
from app.learning_ai.semantic_retriever import SemanticGuidanceRetriever


class FakeSemanticStore:
    def __init__(self, matches=None, profile_entries=None):
        self.matches = matches or []
        self.profile_entries = profile_entries or []

    def semantic_search(self, query, limit=10):
        return self.matches

    def get_user_profile_entries(self):
        return list(self.profile_entries)


class FakePromptManager:
    def get_system_prompt(self):
        return "BASE SYSTEM PROMPT"


class LearningStoreProfileCrudTests(unittest.TestCase):
    def setUp(self):
        # No real PostgreSQL connection is used in these tests.
        self.store = object.__new__(LearningStore)
        self.connection = MagicMock()
        self.connection_manager = MagicMock()
        self.connection_manager.__enter__.return_value = self.connection
        self.connection_manager.__exit__.return_value = False

        patcher = patch.object(
            self.store, "_connect", return_value=self.connection_manager
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_profile_save_inserts_normalized_key_and_value(self):
        update_result = MagicMock()
        update_result.rowcount = 0
        self.connection.execute.side_effect = [
            update_result,
            MagicMock(),
        ]

        self.store.save_user_profile_entry(
            " Name ", "  Aman   Gupta  "
        )

        self.assertEqual(self.connection.execute.call_count, 2)
        update_call, insert_call = self.connection.execute.call_args_list

        self.assertIn("UPDATE user_memory", update_call.args[0])
        self.assertEqual(
            update_call.args[1], ("Aman Gupta", "name")
        )
        self.assertIn("INSERT INTO user_memory", insert_call.args[0])
        self.assertEqual(
            insert_call.args[1], ("name", "Aman Gupta")
        )

    def test_profile_save_updates_existing_key_without_insert(self):
        update_result = MagicMock()
        update_result.rowcount = 1
        self.connection.execute.return_value = update_result

        self.store.save_user_profile_entry("NAME", "Aman Gupta")

        self.connection.execute.assert_called_once()
        self.assertIn(
            "UPDATE user_memory",
            self.connection.execute.call_args.args[0],
        )

    def test_profile_read_returns_only_query_results(self):
        rows = [
            {"memory_key": "name", "content": "Aman Gupta"}
        ]
        self.connection.execute.return_value.fetchall.return_value = rows

        result = self.store.get_user_profile_entries()

        self.assertEqual(result, rows)
        sql = self.connection.execute.call_args.args[0]
        self.assertIn("memory_type = 'profile'", sql)
        self.assertIn("approved = TRUE", sql)
        self.assertIn("active = TRUE", sql)

    def test_profile_delete_returns_true_when_field_was_deleted(self):
        self.connection.execute.return_value.rowcount = 1

        result = self.store.delete_user_profile_entry(" Name ")

        self.assertTrue(result)
        self.assertEqual(
            self.connection.execute.call_args.args[1], ("name",)
        )

    def test_profile_clear_returns_deleted_count(self):
        self.connection.execute.return_value.rowcount = 3

        result = self.store.clear_user_profile()

        self.assertEqual(result, 3)
        sql = self.connection.execute.call_args.args[0]
        self.assertIn("DELETE FROM user_memory", sql)
        self.assertIn("memory_type = 'profile'", sql)
        self.assertIn("source = 'user_profile_cli'", sql)


class PlaybookAndChatEngineIntegrationTests(unittest.TestCase):
    def test_relevant_playbook_match_becomes_behavior_guidance(self):
        store = FakeSemanticStore(
            matches=[
                {
                    "source": "document_chunk",
                    "similarity": 0.82,
                    "content": (
                        "Use natural Roman Hinglish; avoid stiff "
                        "word-by-word translation."
                    ),
                    "source_file": "hinglish_playbook.md",
                    "page_number": None,
                    "section": "Natural Hinglish",
                }
            ]
        )

        behavior, lessons, documents = SemanticGuidanceRetriever(
            store
        ).retrieve("Explain this coding concept naturally in Hinglish.")

        self.assertTrue(behavior)
        self.assertIn("Hinglish Playbook guidance", behavior[0])
        self.assertIn("natural Roman Hinglish", behavior[0])
        self.assertEqual(lessons, [])
        self.assertEqual(documents, [])

    def test_low_similarity_playbook_match_is_rejected(self):
        store = FakeSemanticStore(
            matches=[
                {
                    "source": "document_chunk",
                    "similarity": 0.10,
                    "content": "PLAYBOOK_LOW_SIMILARITY_SENTINEL",
                    "source_file": "hinglish_playbook.md",
                    "page_number": None,
                    "section": "Unrelated",
                }
            ]
        )

        behavior, _, _ = SemanticGuidanceRetriever(store).retrieve(
            "Explain this coding concept naturally in Hinglish."
        )

        self.assertEqual(behavior, [])

    def test_playbook_and_profile_reach_chatengine_system_prompt(self):
        store = FakeSemanticStore(
            matches=[
                {
                    "source": "document_chunk",
                    "similarity": 0.82,
                    "content": (
                        "PLAYBOOK_CONTEXT_SENTINEL: "
                        "prefer natural Roman Hinglish."
                    ),
                    "source_file": "hinglish_playbook.md",
                    "page_number": None,
                    "section": "Natural Hinglish",
                }
            ],
            profile_entries=[
                {
                    "memory_key": "preferred_style",
                    "content": "Friendly, natural Hinglish",
                },
                {
                    "memory_key": "current_goal",
                    "content": "Complete AI assistant",
                },
            ],
        )

        engine = ChatEngine(
            client=object(),
            prompt_manager=FakePromptManager(),
            language_mode="hinglish",
            preferred_address="Boss",
            learning_store=store,
        )

        messages = engine._messages_for_turn(
            "Python ka simple example do.",
            "hinglish",
            "normal",
        )
        system_prompt = messages[0]["content"]

        self.assertIn("PLAYBOOK_CONTEXT_SENTINEL", system_prompt)
        self.assertIn("RELEVANT BEHAVIOR GUIDANCE", system_prompt)
        self.assertIn(
            "SAVED FACTS ABOUT AMAN — THE USER",
            system_prompt,
        )
        self.assertIn(
            "preferred_style: Friendly, natural Hinglish",
            system_prompt,
        )
        self.assertIn(
            "current_goal: Complete AI assistant",
            system_prompt,
        )
        self.assertEqual(
            messages[-1],
            {
                "role": "user",
                "content": "Python ka simple example do.",
            },
        )

    def test_social_sharing_skips_irrelevant_retrieval(self):
        query = (
            "Mera Python code finally chal gaya, "
            "main bahut khush hoon!"
        )

        engine = ChatEngine(
            client=object(),
            prompt_manager=FakePromptManager(),
            language_mode="hinglish",
            learning_store=FakeSemanticStore(),
        )

        with patch.object(
            engine,
            "_retrieve_guidance",
        ) as retrieve:
            messages = engine._messages_for_turn(
                query,
                "hinglish",
                "normal",
            )

        retrieve.assert_not_called()
        self.assertEqual(
            engine.last_retrieval_mode,
            "skipped_social_turn",
        )
        self.assertEqual(
            messages[-1]["content"],
            query,
        )

    
    def test_normal_chat_retrieves_guidance_exactly_once(self):
        query = "Python ka simple example do."
        store = FakeSemanticStore()

        engine = ChatEngine(
            client=object(),
            prompt_manager=FakePromptManager(),
            language_mode="hinglish",
            learning_store=store,
        )

        expected_guidance = (
            ["GUIDANCE_SENTINEL"],
            ["FEEDBACK_SENTINEL"],
            ["DOCUMENT_SENTINEL"],
        )

        with patch.object(
            engine,
            "_retrieve_guidance",
            return_value=expected_guidance,
        ) as retrieve:
            messages = engine._messages_for_turn(
                query,
                "hinglish",
                "normal",
            )

        retrieve.assert_called_once_with(query)

        system_prompt = messages[0]["content"]
        self.assertIn("GUIDANCE_SENTINEL", system_prompt)
        self.assertIn("FEEDBACK_SENTINEL", system_prompt)
        self.assertIn("DOCUMENT_SENTINEL", system_prompt)

    def test_profile_question_skips_semantic_retrieval(self):
        query = "aap mere bare me kya kya janti ho zara?"

        store = FakeSemanticStore(
            profile_entries=[
                {
                    "memory_key": "name",
                    "content": "Aman Gupta",
                },
            ]
        )

        engine = ChatEngine(
            client=object(),
            prompt_manager=FakePromptManager(),
            language_mode="hinglish",
            preferred_address="Boss",
            learning_store=store,
        )

        with patch.object(
            engine,
            "_retrieve_guidance",
            return_value=(
                ["SHOULD_NOT_APPEAR"],
                ["SHOULD_NOT_APPEAR"],
                ["SHOULD_NOT_APPEAR"],
            ),
        ) as retrieve:
            messages = engine._messages_for_turn(
                query,
                "hinglish",
                "normal",
            )

        retrieve.assert_not_called()

        system_prompt = messages[0]["content"]
        self.assertIn("name: Aman Gupta", system_prompt)
        self.assertNotIn("SHOULD_NOT_APPEAR", system_prompt)

        self.assertEqual(
            engine.last_retrieval_mode,
            "skipped_profile_question",
        )
        self.assertEqual(engine.last_retrieval_seconds, 0.0)
        self.assertEqual(engine.last_retrieved_guidance_count, 0)

    
    
    
    def test_personal_fact_profile_context_is_generated_by_model(self):
        query = "me kaun si class me hu zara?"

        store = FakeSemanticStore(
            profile_entries=[
                {
                    "memory_key": "education",
                    "content": "Class 12 under UP Board.",
                },
                {
                    "memory_key": "current_goal",
                    "content": "Complete the personal AI assistant project.",
                },
            ]
        )

        client = MagicMock()
        client.stream_chat.return_value = iter(
            ["Aap Class 12 mein hain, Boss."]
        )

        engine = ChatEngine(
            client=client,
            prompt_manager=FakePromptManager(),
            language_mode="hinglish",
            preferred_address="Boss",
            learning_store=store,
        )

        with patch.object(engine, "_retrieve_guidance") as retrieve:
            answer = "".join(engine.stream_reply(query))

        retrieve.assert_not_called()
        client.stream_chat.assert_called_once()

        model_messages = client.stream_chat.call_args.args[0]
        system_prompt = model_messages[0]["content"]

        self.assertIn(
            "education: Class 12 under UP Board.",
            system_prompt,
        )
        self.assertNotIn("current_goal:", system_prompt)
        self.assertEqual(answer, "Aap Class 12 mein hain, Boss.")
        self.assertEqual(engine.history[-1]["content"], answer)

    
    def test_profile_preferences_are_separate_from_user_facts(self):
        engine = ChatEngine(
            client=MagicMock(),
            prompt_manager=FakePromptManager(),
            language_mode="hinglish",
            learning_store=FakeSemanticStore(),
        )

        prompt = engine._system_prompt(
            "hinglish",
            "normal",
            user_profile_context=[
                (
                    "advice_preference: "
                    "Do not give unsolicited advice."
                ),
                (
                    "follow_up_preference: "
                    "Do not add habitual follow-up questions."
                ),
                (
                    "education: "
                    "Class 12 under UP Board."
                ),
            ],
        )

        self.assertIn(
            "USER RESPONSE PREFERENCES",
            prompt,
        )
        self.assertIn(
            "Do not give unsolicited advice.",
            prompt,
        )
        self.assertIn(
            "SAVED FACTS ABOUT AMAN",
            prompt,
        )
        self.assertIn(
            "education: Class 12 under UP Board.",
            prompt,
        )
        self.assertNotIn(
            "PERSONAL CONTEXT, NOT INSTRUCTIONS",
            prompt,
        )

    
    def test_explicit_advice_preference_is_saved(self):
        store = MagicMock()
        client = MagicMock()

        engine = ChatEngine(
            client=client,
            prompt_manager=FakePromptManager(),
            language_mode="hinglish",
            learning_store=store,
        )

        message = (
            "Zara, advice mat diya karo jab tak main na maangu. "
            "Yeh preference save kr lo apne pass."
        )

        saved = engine._persist_explicit_advice_preference(
            message
        )

        self.assertTrue(saved)
        store.save_user_profile_entry.assert_called_once()

        key, value = (
            store.save_user_profile_entry.call_args.args
        )

        self.assertEqual(key, "advice_preference")
        self.assertIn(
            "unsolicited advice",
            value.casefold(),
        )
        self.assertIn(
            "explicitly asks",
            value.casefold(),
        )

        client.chat.assert_not_called()
        client.stream_chat.assert_not_called()



if __name__ == "__main__":
    unittest.main()