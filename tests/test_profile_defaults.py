import unittest

from app.learning_ai.profile_defaults import (
    DEFAULT_PROFILE_FIELDS,
    get_profile_setup_status,
    save_default_profile_fields,
)


class FakeProfileStore:
    def __init__(self, entries=None):
        self.entries = list(entries or [])
        self.save_calls = []

    def get_user_profile_entries(self):
        return [dict(item) for item in self.entries]

    def save_user_profile_entry(self, memory_key, content):
        self.save_calls.append((memory_key, content))
        self.entries = [
            item for item in self.entries
            if item.get("memory_key") != memory_key
        ]
        self.entries.append({"memory_key": memory_key, "content": content})


class ProfileDefaultsTests(unittest.TestCase):
    def test_preview_is_read_only_and_lists_all_missing_fields(self):
        store = FakeProfileStore()

        missing, present = get_profile_setup_status(store)

        self.assertEqual(missing, list(DEFAULT_PROFILE_FIELDS))
        self.assertEqual(present, [])
        self.assertEqual(store.save_calls, [])

    def test_setup_saves_missing_defaults_and_preserves_existing_fields(self):
        store = FakeProfileStore([
            {"memory_key": "name", "content": "Custom Name"},
            {"memory_key": "interests", "content": "Custom interests"},
        ])

        saved, preserved = save_default_profile_fields(store)

        self.assertNotIn("name", saved)
        self.assertNotIn("interests", saved)
        self.assertEqual(preserved, ["name", "interests"])
        self.assertEqual(len(saved), len(DEFAULT_PROFILE_FIELDS) - 2)
        final_entries = {
            item["memory_key"]: item["content"]
            for item in store.get_user_profile_entries()
        }
        self.assertEqual(final_entries["name"], "Custom Name")
        self.assertEqual(final_entries["interests"], "Custom interests")
        self.assertEqual(final_entries["current_goal"], DEFAULT_PROFILE_FIELDS["current_goal"])

    def test_second_setup_run_is_idempotent(self):
        store = FakeProfileStore()

        first_saved, _ = save_default_profile_fields(store)
        second_saved, preserved = save_default_profile_fields(store)

        self.assertEqual(len(first_saved), len(DEFAULT_PROFILE_FIELDS))
        self.assertEqual(second_saved, [])
        self.assertEqual(set(preserved), set(DEFAULT_PROFILE_FIELDS))
        self.assertEqual(len(store.save_calls), len(DEFAULT_PROFILE_FIELDS))

    def test_all_profile_keys_and_values_fit_store_limits(self):
        for key, value in DEFAULT_PROFILE_FIELDS.items():
            with self.subTest(key=key):
                self.assertLessEqual(len(key), 64)
                self.assertTrue(all(c.isalnum() or c in "_-" for c in key))
                self.assertGreater(len(value.strip()), 0)
                self.assertLessEqual(len(value), 600)


if __name__ == "__main__":
    unittest.main()
