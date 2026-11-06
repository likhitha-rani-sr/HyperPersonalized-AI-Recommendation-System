import unittest
from unittest.mock import patch

import pandas as pd

from src import api, conversational_rag as rag
from src.query_processor import process_query


class ConversationalRagFilterTests(unittest.TestCase):

    def setUp(self):
        rag.reset_conversation()

    @staticmethod
    def products(*products):
        return pd.DataFrame(products)

    def run_recommend(self, query, products):
        with (
            patch.object(rag, "products_df", products),
            patch.object(
                rag,
                "rank_products",
                side_effect=lambda matches, _query: matches.copy(),
            ),
            patch.object(rag, "display_products"),
        ):
            return rag.recommend(query)

    def test_mandatory_filters_are_applied_together(self):
        products = pd.DataFrame(
            [
                {
                    "product_name": "IKEA Metal Wardrobe",
                    "category": "Wardrobe",
                    "brand": "IKEA",
                    "material": "Metal",
                    "price": 8000,
                },
                {
                    "product_name": "Godrej Metal Wardrobe",
                    "category": "Wardrobe",
                    "brand": "Godrej",
                    "material": "Metal",
                    "price": 9000,
                },
                {
                    "product_name": "Godrej Metal Wardrobe",
                    "category": "Wardrobe",
                    "brand": "Godrej",
                    "material": "Metal",
                    "price": 12000,
                },
                {
                    "product_name": "Godrej Wood Wardrobe",
                    "category": "Wardrobe",
                    "brand": "Godrej",
                    "material": "Wood",
                    "price": 7000,
                },
                {
                    "product_name": "Godrej Metal Sofa",
                    "category": "Sofa",
                    "brand": "Godrej",
                    "material": "Metal",
                    "price": 7000,
                },
            ]
        )
        query_info = process_query(
            "not IKEA metal wardrobe under ₹10,000"
        )

        with patch.object(rag, "products_df", products):
            filtered = rag.filter_products(query_info)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(
            filtered.iloc[0]["product_name"],
            "Godrej Metal Wardrobe",
        )
        self.assertEqual(filtered.iloc[0]["price"], 9000)

    def test_unavailable_mandatory_combination_returns_explanation(self):
        query_info = {
            "category": "wardrobe",
            "color": "blue",
            "material": "metal",
            "excluded_brands": ["ikea"],
            "max_price": 10000,
        }

        response = rag.generate_fast_response(
            pd.DataFrame(),
            query_info,
        )

        self.assertIn("matching all your requirements", response)
        self.assertIn("wardrobe", response)
        self.assertIn("blue color", response)
        self.assertIn("metal material", response)
        self.assertIn("excluding ikea", response)
        self.assertIn("₹10,000", response)

    def test_conversational_api_keeps_alternatives_separate(self):
        exact = [{"product_name": "Exact match"}]
        alternatives = [
            {
                "product_name": "Alternative",
                "alternative_reason": "Does not satisfy the requested material.",
            }
        ]
        result = {
            "answer": "No exact matches were found.",
            "products": exact,
            "alternative_products": alternatives,
        }

        with (
            patch.object(api, "call_function", return_value=result),
            patch.object(
                api,
                "enrich_product_ids",
                side_effect=lambda products: products,
            ),
        ):
            api_products, response, api_alternatives = (
                api.conversational_recommendation(
                    1,
                    "metal wardrobe",
                    5,
                )
            )

        self.assertEqual(api_products, exact)
        self.assertEqual(api_alternatives, alternatives)
        self.assertIn("No exact matches", response)

    def test_metal_wardrobe_alternative_discloses_catalog_material(self):
        catalog = self.products(
            {
                "product_name": "Godrej Wood Wardrobe",
                "category": "wardrobe",
                "brand": "Godrej",
                "color": "Brown",
                "material": "Wood",
                "price": 9000,
            }
        )

        result = self.run_recommend("metal wardrobe", catalog)

        self.assertEqual(result["products"], [])
        self.assertEqual(len(result["alternative_products"]), 1)
        alternative = result["alternative_products"][0]
        self.assertEqual(alternative["material"], "Wood")
        self.assertIn("No exact matches were found", result["answer"])
        self.assertIn("Alternative suggestions", result["answer"])
        self.assertIn("requested material 'metal'", alternative["alternative_reason"])
        self.assertIn("catalog lists 'Wood'", alternative["alternative_reason"])

    def test_blue_sofa_alternative_relaxes_color_but_preserves_category_and_budget(self):
        catalog = self.products(
            {
                "product_name": "Affordable Brown Sofa",
                "category": "sofa",
                "brand": "Durian",
                "color": "Brown",
                "material": "Fabric",
                "price": 22000,
            },
            {
                "product_name": "Over-budget Blue Sofa",
                "category": "sofa",
                "brand": "IKEA",
                "color": "Blue",
                "material": "Fabric",
                "price": 30000,
            },
        )

        result = self.run_recommend(
            "comfortable blue sofa under ₹25,000",
            catalog,
        )

        self.assertEqual(result["products"], [])
        self.assertEqual(
            [product["product_name"] for product in result["alternative_products"]],
            ["Affordable Brown Sofa"],
        )
        self.assertIn("category 'sofa'", result["answer"])
        self.assertIn("color 'blue'", result["answer"])
        self.assertIn("₹25,000", result["answer"])
        self.assertIn(
            "requested color 'blue'",
            result["alternative_products"][0]["alternative_reason"],
        )
        self.assertIn(
            "catalog lists 'Brown'",
            result["alternative_products"][0]["alternative_reason"],
        )

    def test_wooden_dining_table_matches_catalog_table_products(self):
        catalog = self.products(
            {
                "product_name": "Wood Dining Table",
                "category": "Table",
                "brand": "IKEA",
                "color": "Brown",
                "material": "Wood",
                "price": 18000,
            },
            {
                "product_name": "Wood Sofa",
                "category": "Sofa",
                "brand": "IKEA",
                "color": "Brown",
                "material": "Wood",
                "price": 15000,
            },
        )

        result = self.run_recommend(
            "Recommend a wooden dining table.",
            catalog,
        )

        self.assertEqual(
            [product["product_name"] for product in result["products"]],
            ["Wood Dining Table"],
        )
        self.assertEqual(result["alternative_products"], [])
        self.assertNotIn("No exact matches", result["answer"])

    def test_brand_exclusion_is_preserved_for_alternatives(self):
        catalog = self.products(
            {
                "product_name": "Godrej Wood Wardrobe",
                "category": "wardrobe",
                "brand": "Godrej",
                "color": "Brown",
                "material": "Wood",
                "price": 9000,
            },
            {
                "product_name": "IKEA Metal Wardrobe",
                "category": "wardrobe",
                "brand": "IKEA",
                "color": "White",
                "material": "Metal",
                "price": 8000,
            },
        )

        result = self.run_recommend(
            "not IKEA metal wardrobe",
            catalog,
        )

        self.assertEqual(result["products"], [])
        self.assertTrue(result["alternative_products"])
        self.assertTrue(
            all(
                product["brand"] != "IKEA"
                for product in result["alternative_products"]
            )
        )
        self.assertEqual(
            result["preferences"]["excluded_brands"],
            ["ikea"],
        )

    def test_budget_is_preserved_and_can_leave_no_alternatives(self):
        catalog = self.products(
            {
                "product_name": "Godrej Wood Wardrobe",
                "category": "wardrobe",
                "brand": "Godrej",
                "color": "Brown",
                "material": "Wood",
                "price": 9000,
            },
            {
                "product_name": "Durian Metal Wardrobe",
                "category": "wardrobe",
                "brand": "Durian",
                "color": "Black",
                "material": "Metal",
                "price": 12000,
            },
        )

        result = self.run_recommend(
            "metal wardrobe under ₹10,000",
            catalog,
        )

        self.assertEqual(result["products"], [])
        self.assertTrue(result["alternative_products"])
        self.assertTrue(
            all(
                product["price"] <= 10000
                for product in result["alternative_products"]
            )
        )
        self.assertNotIn(
            "Durian Metal Wardrobe",
            [product["product_name"] for product in result["alternative_products"]],
        )

        too_expensive = self.products(
            {
                "product_name": "Durian Metal Wardrobe",
                "category": "wardrobe",
                "brand": "Durian",
                "color": "Black",
                "material": "Metal",
                "price": 12000,
            }
        )
        no_alternatives = self.run_recommend(
            "metal wardrobe under ₹10,000",
            too_expensive,
        )

        self.assertEqual(no_alternatives["products"], [])
        self.assertEqual(no_alternatives["alternative_products"], [])
        self.assertIn("No exact matches were found", no_alternatives["answer"])
        self.assertIn("No suitable alternatives remain", no_alternatives["answer"])

    def test_multi_turn_memory_survives_exclusion_and_typo_correction(self):
        catalog = self.products(
            {
                "product_name": "Godrej Wood Wardrobe",
                "category": "wardrobe",
                "brand": "Godrej",
                "color": "Brown",
                "material": "Wood",
                "price": 9000,
            },
            {
                "product_name": "IKEA Metal Wardrobe",
                "category": "wardrobe",
                "brand": "IKEA",
                "color": "White",
                "material": "Metal",
                "price": 8000,
            },
            {
                "product_name": "Godrej Metal Sofa",
                "category": "sofa",
                "brand": "Godrej",
                "color": "Black",
                "material": "Metal",
                "price": 7000,
            },
        )
        with (
            patch.object(rag, "products_df", catalog),
            patch.object(
                rag,
                "rank_products",
                side_effect=lambda matches, _query: matches.copy(),
            ),
            patch.object(rag, "display_products"),
        ):
            first = rag.recommend("wooden wardrobes")
            self.assertTrue(first["products"])
            self.assertEqual(first["alternative_products"], [])

            second = rag.recommend("not IKEA")
            self.assertEqual(
                second["preferences"]["excluded_brands"],
                ["ikea"],
            )

            third = rag.recommend("show metal wardrobes instead")
            self.assertEqual(third["preferences"]["category"], "wardrobe")
            self.assertEqual(third["preferences"]["material"], "metal")
            self.assertEqual(third["products"], [])
            self.assertTrue(third["alternative_products"])

            corrected = rag.recommend("waardroop")
            self.assertEqual(corrected["preferences"]["category"], "wardrobe")
            self.assertEqual(corrected["preferences"]["material"], "metal")
            self.assertEqual(
                corrected["preferences"]["excluded_brands"],
                ["ikea"],
            )
            self.assertTrue(corrected["alternative_products"])

            budget = rag.recommend("Under ₹10,000")
            self.assertEqual(budget["preferences"]["category"], "wardrobe")
            self.assertEqual(budget["preferences"]["material"], "metal")
            self.assertEqual(
                budget["preferences"]["excluded_brands"],
                ["ikea"],
            )
            self.assertEqual(budget["preferences"]["max_price"], 10000.0)
            self.assertTrue(budget["alternative_products"])
            self.assertTrue(
                all(
                    product["price"] <= 10000
                    and product["brand"] != "IKEA"
                    for product in budget["alternative_products"]
                )
            )

    def test_repeating_category_after_no_matches_retries_without_old_filters(self):
        catalog = self.products(
            {
                "product_name": "Durian Fabric Sofa",
                "category": "sofa",
                "brand": "Durian",
                "color": "Blue",
                "material": "Fabric",
                "price": 30000,
            }
        )

        with (
            patch.object(rag, "products_df", catalog),
            patch.object(
                rag,
                "rank_products",
                side_effect=lambda matches, _query: matches.copy(),
            ),
            patch.object(rag, "display_products"),
        ):
            first = rag.recommend("comfortable sofa under ₹25,000")
            self.assertEqual(first["products"], [])
            self.assertEqual(first["preferences"]["max_price"], 25000.0)

            retry = rag.recommend("sofa")

        self.assertEqual(
            [product["product_name"] for product in retry["products"]],
            ["Durian Fabric Sofa"],
        )
        self.assertIsNone(retry["preferences"]["max_price"])

    def test_category_only_retry_after_restoring_conversation_clears_old_filters(self):
        catalog = self.products(
            {
                "product_name": "Durian Fabric Sofa",
                "category": "sofa",
                "brand": "Durian",
                "color": "Blue",
                "material": "Fabric",
                "price": 30000,
            }
        )
        rag.restore_conversation(
            [
                {"role": "user", "content": "comfortable sofa under ₹25,000"},
                {
                    "role": "assistant",
                    "content": "No exact matches were found for your requested category 'sofa'.",
                },
            ],
            {"category": "sofa", "max_price": 25000.0},
        )

        result = self.run_recommend("sofa", catalog)

        self.assertEqual(
            [product["product_name"] for product in result["products"]],
            ["Durian Fabric Sofa"],
        )
        self.assertIsNone(result["preferences"]["max_price"])

    def test_new_general_furniture_request_clears_stale_category_and_filters(self):
        catalog = self.products(
            {
                "product_name": "Modern Wood Chair",
                "category": "chair",
                "brand": "Durian",
                "color": "Brown",
                "material": "Wood",
                "price": 18000,
            },
            {
                "product_name": "White Godrej Table",
                "category": "table",
                "brand": "Godrej",
                "color": "White",
                "material": "Wood",
                "price": 22000,
            },
        )
        rag.current_preferences.update(
            {
                "category": "dining table",
                "color": "white",
                "material": "wood",
                "style": None,
                "brand": "godrej",
                "max_price": 22360.0,
            }
        )

        result = self.run_recommend(
            "Recommend modern wooden furniture within my budget.",
            catalog,
        )

        self.assertEqual(result["preferences"]["category"], None)
        self.assertEqual(result["preferences"]["color"], None)
        self.assertEqual(result["preferences"]["brand"], None)
        self.assertEqual(result["preferences"]["material"], "wood")
        self.assertEqual(result["preferences"]["style"], "modern")
        self.assertEqual(result["preferences"]["max_price"], 22360.0)
        self.assertEqual(len(result["products"]), 2)
        self.assertTrue(
            all(
                product["price"] <= 22360
                for product in result["products"]
            )
        )

    def test_restored_legacy_dining_table_category_is_normalized(self):
        rag.restore_conversation(
            [],
            {"category": "dining table", "material": "wood"},
        )

        self.assertEqual(rag.current_preferences["category"], "table")


if __name__ == "__main__":
    unittest.main()
