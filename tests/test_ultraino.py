import sys
import unittest

import numpy as np

sys.path.insert(0, "src")

from hardware.ultraino import (
    CHANNELS,
    PHASE_OFF,
    START_PHASES,
    SWAP_BUFFERS,
    UltrainoConfig,
    compile_all_off,
    compile_simple_fpga,
)


class UltrainoProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = UltrainoConfig(
            name="test",
            serial_port=None,
            baudrate=230400,
            solver_to_order=np.arange(CHANNELS),
            phase_corrections_pi=np.zeros(CHANNELS),
            calibration_verified=False,
        )

    def test_matches_simple_fpga_two_write_protocol(self):
        phase_write, swap_write = compile_simple_fpga(
            np.zeros(CHANNELS), self.config
        )
        self.assertEqual(len(phase_write), 257)
        self.assertEqual(phase_write[0], START_PHASES)
        self.assertEqual(phase_write[1:], bytes(CHANNELS))
        self.assertEqual(swap_write, bytes((SWAP_BUFFERS,)))

    def test_phase_correction_is_in_pi_units(self):
        config = self.config.__class__(
            **{
                **self.config.__dict__,
                "phase_corrections_pi": np.full(CHANNELS, 1.0),
            }
        )
        phase_write, _ = compile_simple_fpga(np.zeros(CHANNELS), config)
        self.assertEqual(phase_write[1], 16)

    def test_all_off_uses_ultraino_disabled_value(self):
        phase_write, swap_write = compile_all_off()
        self.assertEqual(phase_write[0], START_PHASES)
        self.assertEqual(phase_write[1:], bytes((PHASE_OFF,)) * CHANNELS)
        self.assertEqual(swap_write, bytes((SWAP_BUFFERS,)))


if __name__ == "__main__":
    unittest.main()
