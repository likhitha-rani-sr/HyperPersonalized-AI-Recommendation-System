import unittest

from src.query_processor import process_query


class QueryProcessorTests(unittest.TestCase):

    def test_waardroop_is_normalized_to_wardrobe(self):
        result = process_query("waardroop")

        self.assertEqual(result["normalized_query"], "wardrobe")
        self.assertEqual(result["category"], "wardrobe")

    def test_other_that_brand_is_parsed_as_exclusion(self):
        result = process_query("other that Ikea Brand")

        self.assertIsNone(result["brand"])
        self.assertEqual(result["excluded_brands"], ["ikea"])

    def test_not_brand_other_brands_is_parsed_as_exclusion(self):
        result = process_query("not Ikea other brands")

        self.assertIsNone(result["brand"])
        self.assertEqual(result["excluded_brands"], ["ikea"])

    def test_positive_brand_filter_remains_supported(self):
        result = process_query("Ikea wardrobe")

        self.assertEqual(result["brand"], "ikea")
        self.assertEqual(result["excluded_brands"], [])

    def test_dining_table_uses_catalog_table_category(self):
        result = process_query("Recommend a wooden dining table.")

        self.assertEqual(result["category"], "table")
        self.assertEqual(result["material"], "wood")

    def test_general_furniture_request_does_not_infer_a_category(self):
        result = process_query(
            "Recommend modern wooden furniture within my budget."
        )

        self.assertIsNone(result["category"])
        self.assertEqual(result["material"], "wood")
        self.assertEqual(result["style"], "modern")


if __name__ == "__main__":
    unittest.main()
