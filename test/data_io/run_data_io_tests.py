import os
import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    # Get project root and add to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Change working directory to project root
    os.chdir(project_root)

    # Find all test files in data_io and subdirectories
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Get all test files recursively
    data_io_dir = Path(__file__).parent
    for test_file in data_io_dir.rglob("test_*.py"):
        # Convert path to module name
        module_path = str(test_file.relative_to(project_root))[:-3].replace(os.sep, ".")
        try:
            # Import module and add its tests
            __import__(module_path)
            module = sys.modules[module_path]
            suite = test_loader.loadTestsFromModule(module)
            test_suite.addTests(suite)
        except Exception as e:
            print(f"Error loading tests from {test_file}: {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Show summary
    if result.wasSuccessful():
        print("\n=========================")
        print("🟢 ALL TESTS SUCCESSFUL!")
        print("=========================")
    else:
        print("\n==================")
        print("🔴 TESTS FAILED!")
        print("==================")
