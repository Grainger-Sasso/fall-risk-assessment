import os
import sys
import unittest
from pathlib import Path


def collect_test_cases(start_dir: Path, project_root: Path) -> unittest.TestSuite:
    """Collect all test cases from the given directory."""
    suite = unittest.TestSuite()

    for item in start_dir.rglob("test_*.py"):
        # Convert path to module name (starting from test directory)
        rel_path = item.relative_to(project_root)
        module_path = str(rel_path)[:-3].replace(os.sep, ".")
        try:
            # Import the module and add its tests to suite
            __import__(module_path)
            module = sys.modules[module_path]
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        except Exception as e:
            print(f"Error loading tests from {item}: {e}")

    return suite


if __name__ == "__main__":
    # Get absolute paths
    project_root = Path(__file__).parent.parent.resolve()
    test_dir = project_root / "test" / "data_io"

    # Add project root to Python path
    sys.path.insert(0, str(project_root))

    # Change working directory to project root
    os.chdir(project_root)

    # Collect and run tests
    test_suite = collect_test_cases(test_dir, project_root)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Show summary with our formatting
    if result.wasSuccessful():
        print("\n=========================")
        print("🟢 ALL TESTS SUCCESSFUL!")
        print("=========================")
    else:
        print("\n==================")
        print("🔴 TESTS FAILED!")
        print("==================")
