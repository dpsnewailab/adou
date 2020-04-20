import unittest
from adou.utils.logger import timer, ignore_runtime_error

class LoggerTestCase(unittest.TestCase):
    def test_timer(self):
        @timer
        def add(a, b):
            import time
            time.sleep(1)
            return a + b

        add(1, 2)

    def test_ignore_runtime_error(self):
        @ignore_runtime_error
        def divide(a, b):
            return a / b;

        divide(1, 0)

if __name__ == '__main__':
    unittest.main()
