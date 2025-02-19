from test.base_test import BaseTest

class TestHDF5FileWriter(BaseTest):
    def setUp(self):
        self.writer = HDF5FileWriter()
        # ... rest of setup ...

    # ... rest of your test methods ...

if __name__ == "__main__":
    TestHDF5FileWriter.run_tests()
