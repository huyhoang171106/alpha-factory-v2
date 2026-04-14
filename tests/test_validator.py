import unittest

from validator import (
    normalize_expression_aliases,
    supports_local_backtest_expression,
    validate_expression,
)


class ValidatorTests(unittest.TestCase):
    def test_adv_aliases_normalize_to_canonical_fields(self):
        expr = "ts_corr(vwap, adv81, 10) + ts_corr(close, adv40, 5) + ts_corr(volume, adv150, 7)"
        normalized = normalize_expression_aliases(expr)

        self.assertIn("adv60", normalized)
        self.assertIn("adv120", normalized)
        self.assertNotIn("adv81", normalized)
        self.assertNotIn("adv40", normalized)
        self.assertNotIn("adv150", normalized)

        ok, reason = validate_expression(normalized)
        self.assertTrue(ok, reason)

    def test_local_bt_support_accepts_canonicalized_adv_alias(self):
        ok, reason = supports_local_backtest_expression(
            "rank(ts_corr(vwap, adv81, 10))"
        )
        self.assertTrue(ok, reason)

    def test_local_bt_support_rejects_comparisons(self):
        ok, reason = supports_local_backtest_expression("rank(close) < rank(open)")
        self.assertFalse(ok)
        self.assertEqual(reason, "comparison_operator")

    def test_local_bt_support_rejects_fundamental_fields(self):
        ok, reason = supports_local_backtest_expression("rank(ebitda)")
        self.assertFalse(ok)
        self.assertEqual(reason, "unsupported_field:ebitda")


if __name__ == "__main__":
    unittest.main()
