import unittest


class BaseTest(unittest.TestCase):
    """Base test class with standardized test output."""

    @classmethod
    def run_tests(cls):
        """Run tests with standardized output formatting."""
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("\n=========================")
            print("🟢 ALL TESTS SUCCESSFUL!")
            print("=========================")
        else:
            print("\n==================")
            print("🔴 TESTS FAILED!")
            print("==================") 