import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestIntegrationImportGuards(unittest.TestCase):
    def test_langchain_adapter_missing_dep_raises_helpful_error(self):
        try:
            import langchain_core  # noqa: F401

            self.skipTest("langchain-core is installed; guarded-import path not exercised")
        except ImportError:
            pass
        with self.assertRaises(ImportError) as ctx:
            import rrl.integrations.langchain  # noqa: F401
        self.assertIn("retrieval-reputation-layer[langchain]", str(ctx.exception))

    def test_llamaindex_adapter_missing_dep_raises_helpful_error(self):
        try:
            import llama_index.core  # noqa: F401

            self.skipTest("llama-index-core is installed; guarded-import path not exercised")
        except ImportError:
            pass
        with self.assertRaises(ImportError) as ctx:
            import rrl.integrations.llama_index  # noqa: F401
        self.assertIn("retrieval-reputation-layer[llamaindex]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
