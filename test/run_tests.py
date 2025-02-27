import argparse
import os
import sys
import unittest
from pathlib import Path
from typing import List, Tuple


def discover_and_load_tests(test_dir: Path) -> Tuple[unittest.TestSuite, List[str]]:
    """
    Discover and load all tests from specified directory, capturing any errors during loading.

    Args:
        test_dir (Path): Directory containing tests to run

    Returns:
        Tuple[unittest.TestSuite, List[str]]: Test suite and list of error messages
    """
    # Get project root and add to path
    project_root = Path(__file__).parent.parent.resolve()  # Get absolute path
    test_dir = test_dir.resolve()  # Get absolute path
    sys.path.insert(0, str(project_root))

    # Change working directory to project root
    os.chdir(project_root)

    # Find all test files in specified directory and subdirectories
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    errors: List[str] = []
    seen_errors: set = set()  # Track unique errors per file

    # Get all test files recursively
    for test_file in test_dir.rglob("test_*.py"):
        try:
            # Convert path to module name relative to project root
            rel_path = test_file.relative_to(project_root)
            module_path = str(rel_path)[:-3].replace(os.sep, ".")

            # Import module and add its tests
            __import__(module_path)
            module = sys.modules[module_path]
            suite = test_loader.loadTestsFromModule(module)
            test_suite.addTests(suite)
        except ValueError as e:
            # Handle path resolution errors
            error_msg = f"Path error for {test_file}: {str(e)}"
            if error_msg not in seen_errors:
                errors.append(error_msg)
                seen_errors.add(error_msg)
        except Exception as e:
            # Handle other errors
            error_msg = f"{test_file}: {str(e)}"
            if error_msg not in seen_errors:
                errors.append(error_msg)
                seen_errors.add(error_msg)

    return test_suite, errors


def print_results(result: unittest.TestResult, load_errors: List[str]) -> None:
    """Print test results and any loading errors in a clear format."""
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)

    total_issues = len(result.failures) + len(result.errors) + len(load_errors)

    if load_errors:
        print("\n🔴 ERRORS LOADING TESTS:")
        print("-" * 50)
        for i, error in enumerate(load_errors, 1):
            print(f"Error {i}:")
            print(f"• {error}")
            print()

    print("\nTEST EXECUTION SUMMARY:")
    print("-" * 50)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Runtime Errors: {len(result.errors)}")
    print(f"Loading Errors: {len(load_errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Total Issues: {total_issues}")

    if result.failures:
        print("\n🔴 TEST FAILURES:")
        for i, failure in enumerate(result.failures, 1):
            print(f"\nFailure {i}:")
            print(f"• {failure[0]}")
            print(f"  {failure[1]}")

    if result.errors:
        print("\n🔴 RUNTIME ERRORS:")
        for i, error in enumerate(result.errors, 1):
            print(f"\nError {i}:")
            print(f"• {error[0]}")
            print(f"  {error[1]}")

    print("\n" + "=" * 50)
    if total_issues == 0:
        print("🟢 ALL TESTS SUCCESSFUL!")
    else:
        print(f"🔴 TESTS COMPLETED WITH {total_issues} ISSUES")
    print("=" * 50 + "\n")


def parse_args() -> Path:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests from specified directory")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing tests to run (default: test/)",
    )
    args = parser.parse_args()

    if not args.dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.dir}")
    if not args.dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.dir}")

    return args.dir


def main():
    # Get test directory from command line args
    try:
        test_dir = parse_args()
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Discover and load tests
    suite, load_errors = discover_and_load_tests(test_dir)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print results
    print_results(result, load_errors)

    # Return appropriate exit code
    if result.failures or result.errors or load_errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
