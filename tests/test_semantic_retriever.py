import unittest

from app.learning_ai.semantic_retriever import SemanticGuidanceRetriever


class FakeStore:
    def __init__(self, matches):
        self.matches = matches

    def semantic_search(self, query, limit=10):
        return self.matches


class SemanticGuidanceRetrieverTests(unittest.TestCase):
    def test_exact_topic_match_beats_higher_scoring_distractors(self):
        matches = [
            {
                "source": "document_chunk",
                "similarity": 0.4100,
                "content": "TOPIC 63 - Emotion-based tone",
                "source_file": "notes.pdf",
                "page_number": 2,
                "section": "",
            },
            {
                "source": "document_chunk",
                "similarity": 0.3900,
                "content": "TOPIC 61 - AI feelings",
                "source_file": "notes.pdf",
                "page_number": 1,
                "section": "",
            },
            {
                "source": "document_chunk",
                "similarity": 0.3125,
                "content": "TOPIC 62 - EMPATHY-LIKE RESPONSE",
                "source_file": "notes.pdf",
                "page_number": 1,
                "section": "",
            },
        ]

        retriever = SemanticGuidanceRetriever(FakeStore(matches))
        _, _, document_context = retriever.retrieve(
            "TOPIC 62 kahan hai? Sirf page number batao."
        )

        self.assertTrue(document_context)
        self.assertIn("TOPIC 62", document_context[0])
        self.assertIn("page 1", document_context[0].lower())
        self.assertNotIn("TOPIC 63", document_context[0])
        self.assertNotIn("TOPIC 61", document_context[0])

    def test_low_similarity_unmatched_document_is_still_rejected(self):
        matches = [
            {
                "source": "document_chunk",
                "similarity": 0.3125,
                "content": "TOPIC 63 - Emotion-based tone",
                "source_file": "notes.pdf",
                "page_number": 2,
                "section": "",
            }
        ]

        retriever = SemanticGuidanceRetriever(FakeStore(matches))
        _, _, document_context = retriever.retrieve(
            "TOPIC 62 kahan hai? Sirf page number batao."
        )

        self.assertEqual(document_context, [])


if __name__ == "__main__":
    unittest.main()
