import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hardware.sonicsurface.profile import load_profile
from hardware.sonicsurface.protocol import (
    DISABLED_BIN,
    compile_all_off,
    compile_pattern,
    quantize_phases,
)
from hardware.sonicsurface.service import SonicSurfaceController


class SonicSurfaceProtocolTests(unittest.TestCase):
    def _profile_file(self, protocol="fpga_256", verified=False):
        channels = 256 if protocol == "fpga_256" else 512
        content = f"""schema_version: 1
name: test
board_model: coreep4ce6
protocol: {protocol}
solver_channels: {channels}
solver_to_device: identity
mapping_verified: {str(verified).lower()}
phase_sign: 1
global_phase_offset_rad: 0.0
disabled_device_channels: [3]
serial:
  port: null
  baudrate: 230400
"""
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        handle.write(content)
        handle.close()
        return Path(handle.name)

    def test_quantization_wraps_and_rounds(self):
        phases = np.array([
            0.0,
            np.pi / 32.0,
            2.0 * np.pi - np.pi / 32.0,
            2.0 * np.pi,
        ])
        np.testing.assert_array_equal(
            quantize_phases(phases),
            np.array([0, 1, 0, 0], dtype=np.uint8),
        )

    def test_single_board_frame_and_disabled_channel(self):
        profile = load_profile(self._profile_file())
        pattern = compile_pattern(np.zeros(256), profile)
        self.assertEqual(pattern.frames[0][0], 254)
        self.assertEqual(pattern.frames[0][-1], 253)
        self.assertEqual(len(pattern.frames[0]), 258)
        self.assertEqual(pattern.device_bins[3], DISABLED_BIN)
        self.assertEqual(pattern.device_bins[4], 0)

    def test_esp32_frame_contains_two_board_markers(self):
        profile = load_profile(self._profile_file(protocol="esp32_512"))
        pattern = compile_pattern(np.zeros(512), profile)
        self.assertEqual(len(pattern.frames), 1)
        self.assertEqual(len(pattern.frames[0]), 516)
        self.assertEqual(pattern.frames[0][0], 254)
        self.assertEqual(pattern.frames[0][1], 192)
        self.assertEqual(pattern.frames[0][258], 193)
        self.assertEqual(pattern.frames[0][-1], 253)

    def test_all_off_uses_disabled_bin(self):
        profile = load_profile(self._profile_file())
        pattern = compile_all_off(profile)
        self.assertTrue(np.all(pattern.device_bins == DISABLED_BIN))

    def test_live_mode_requires_verified_mapping(self):
        profile = load_profile(self._profile_file(verified=False))
        with self.assertRaisesRegex(ValueError, "mapping_verified"):
            SonicSurfaceController(profile, live=True)

    def test_off_only_mode_allows_off_but_rejects_patterns(self):
        profile = load_profile(self._profile_file(verified=False))
        controller = SonicSurfaceController(
            profile, live=False, off_only=True
        )
        self.assertFalse(controller.status()["armed"])
        self.assertTrue(controller.off()["accepted"])
        with self.assertRaisesRegex(PermissionError, "off-only"):
            controller.submit(np.zeros(256))

    def test_calibration_mode_rejects_optimization_patterns(self):
        profile = load_profile(self._profile_file(verified=False))
        controller = SonicSurfaceController(
            profile, live=False, calibration_mode=True
        )
        self.assertTrue(
            controller.submit(
                np.zeros(256), purpose="diagnostic", label="checkerboard"
            )["accepted"]
        )
        with self.assertRaisesRegex(PermissionError, "diagnostic"):
            controller.submit(np.zeros(256))

    def test_profile_rejects_non_256_solver_channels(self):
        path = self._profile_file()
        content = path.read_text(encoding="utf-8").replace(
            "solver_channels: 256",
            "solver_channels: 144",
        )
        path.write_text(content, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires 256 solver channels"):
            load_profile(path)

    def test_profile_loads_vendor_16x16_mapping(self):
        path = self._profile_file()
        content = path.read_text(encoding="utf-8").replace(
            "solver_to_device: identity",
            "solver_to_device: vendor_sonicsurface_16x16",
        )
        path.write_text(content, encoding="utf-8")
        profile = load_profile(path)
        self.assertEqual(profile.solver_to_device.shape, (256,))
        self.assertEqual(np.unique(profile.solver_to_device).size, 256)
        self.assertEqual(profile.solver_to_device[1], 7)


if __name__ == "__main__":
    unittest.main()
