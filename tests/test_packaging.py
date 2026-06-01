import tomllib
import unittest
from pathlib import Path


class PackagingTest(unittest.TestCase):
    def test_codearch_console_script_is_declared(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())

        self.assertEqual(
            pyproject["project"]["scripts"]["codearch"],
            "codearch.cli:app",
        )


if __name__ == "__main__":
    unittest.main()
