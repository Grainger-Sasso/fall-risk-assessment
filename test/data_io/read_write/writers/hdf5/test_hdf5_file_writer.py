from test.base_test import BaseTest

from src.data_io.read_write.writers.hdf5.hdf5_file_writer import HDF5FileWriter

class TestHDF5FileWriter(BaseTest):
    def setUp(self):
        self.writer = HDF5FileWriter()
        # ... rest of setup ...

    # ... rest of your test methods ...

if __name__ == "__main__":
    TestHDF5FileWriter.run_tests()
