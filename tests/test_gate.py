import unittest

from feeding.gating import OpenDurationGate


class TestOpenDurationGate(unittest.TestCase):
    def test_requires_continuous_open(self):
        gate = OpenDurationGate(required_open_seconds=0.5)

        self.assertFalse(gate.update(0.0, True))
        self.assertFalse(gate.update(0.2, True))
        self.assertTrue(gate.update(0.5, True))

    def test_resets_when_closed(self):
        gate = OpenDurationGate(required_open_seconds=0.5)

        self.assertFalse(gate.update(0.0, True))
        self.assertFalse(gate.update(0.3, False))
        self.assertFalse(gate.update(0.4, True))
        self.assertFalse(gate.update(0.8, True))
        self.assertTrue(gate.update(0.9, True))


if __name__ == "__main__":
    unittest.main()
