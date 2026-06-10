# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import warp._src.build
import warp.config


class TestKernelCache(unittest.TestCase):
    def setUp(self):
        self._original_cache_dir = warp.config.kernel_cache_dir
        self._original_env = os.environ.pop("WARP_CACHE_PATH", None)

    def tearDown(self):
        warp.config.kernel_cache_dir = self._original_cache_dir
        if self._original_env is None:
            os.environ.pop("WARP_CACHE_PATH", None)
        else:
            os.environ["WARP_CACHE_PATH"] = self._original_env

    def test_cache_path_includes_version(self):
        """init_kernel_cache appends the Warp version to user-supplied paths."""
        with tempfile.TemporaryDirectory() as tmp:
            warp._src.build.init_kernel_cache(path=tmp)
            expected = os.path.join(os.path.realpath(tmp), warp.config.version)
            self.assertEqual(warp.config.kernel_cache_dir, expected)
            self.assertTrue(os.path.isdir(expected))

    def test_cache_env_var_includes_version(self):
        """WARP_CACHE_PATH also gets a version subdirectory."""
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["WARP_CACHE_PATH"] = tmp
            warp._src.build.init_kernel_cache()
            expected = os.path.join(os.path.realpath(tmp), warp.config.version)
            self.assertEqual(warp.config.kernel_cache_dir, expected)
            self.assertTrue(os.path.isdir(expected))

    def test_stale_artifacts_warning(self):
        """Warn when the unversioned base directory contains stale wp_ artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "wp_stale_module"))
            with patch("warp._src.logger.log_warning") as mock_warn:
                warp._src.build.init_kernel_cache(path=tmp)
                mock_warn.assert_called_once()
                self.assertIn("previous Warp version", mock_warn.call_args[0][0])

    def test_no_stale_artifacts_warning(self):
        """No warning when the base directory is clean."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch("warp._src.logger.log_warning") as mock_warn:
                warp._src.build.init_kernel_cache(path=tmp)
                mock_warn.assert_not_called()

    def test_default_cache_path_includes_version(self):
        """The default cache path (no env var, no explicit path) includes the version."""
        warp._src.build.init_kernel_cache()
        self.assertIn(warp.config.version, warp.config.kernel_cache_dir)

    def test_lto_cache_does_not_reuse_legacy_prefix_collision(self):
        """LTO cache keys distinguish hashes that share the legacy 7-char prefix."""
        symbol_a = "synthetic_lto_symbol_a"
        symbol_b = "synthetic_lto_symbol_b"
        short_prefix = "abcdef0"
        synthetic_hashes = {
            symbol_a: f"{short_prefix}{'1' * 57}",
            symbol_b: f"{short_prefix}{'2' * 57}",
        }
        self.assertEqual(synthetic_hashes[symbol_a][:7], synthetic_hashes[symbol_b][:7])
        self.assertNotEqual(
            synthetic_hashes[symbol_a][: warp._src.build.LTO_CACHE_KEY_LENGTH],
            synthetic_hashes[symbol_b][: warp._src.build.LTO_CACHE_KEY_LENGTH],
        )

        def compile_lto(symbol):
            def compile_func(temp_paths):
                lto_data = symbol.encode("utf-8")
                with open(temp_paths[".lto"], "wb") as f:
                    f.write(lto_data)
                return True, {".lto": lto_data}

            return compile_func

        with tempfile.TemporaryDirectory() as tmp:
            warp.config.kernel_cache_dir = tmp

            with patch("warp._src.build.hash_symbol", side_effect=lambda symbol: synthetic_hashes[symbol]):
                result_a, lto_data_a = warp._src.build._build_lto_base(symbol_a, compile_lto(symbol_a), builder=None)
                result_b, lto_data_b = warp._src.build._build_lto_base(symbol_b, compile_lto(symbol_b), builder=None)

        self.assertTrue(result_a)
        self.assertTrue(result_b)
        self.assertEqual(lto_data_a, symbol_a.encode("utf-8"))
        self.assertEqual(lto_data_b, symbol_b.encode("utf-8"))

    def test_lto_meta_missing_symbol_is_cache_miss(self):
        """A metadata sidecar without the requested symbol is a cache miss."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "cached.meta")
            with open(meta_path, "w") as f:
                json.dump({"other_symbol": 128}, f)

            self.assertIsNone(warp._src.build.get_cached_lto_meta(meta_path, "requested_symbol"))

    def test_lto_meta_invalid_json_is_cache_miss(self):
        """A corrupt metadata sidecar is a cache miss."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "cached.meta")
            with open(meta_path, "w") as f:
                f.write("{")

            self.assertIsNone(warp._src.build.get_cached_lto_meta(meta_path, "requested_symbol"))

    def test_lto_meta_invalid_encoding_is_cache_miss(self):
        """A metadata sidecar with invalid text encoding is a cache miss."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "cached.meta")
            with open(meta_path, "wb") as f:
                f.write(b"\xff")

            self.assertIsNone(warp._src.build.get_cached_lto_meta(meta_path, "requested_symbol"))

    def test_lto_meta_non_integer_value_is_cache_miss(self):
        """A metadata sidecar with a non-integer value is a cache miss."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "cached.meta")
            with open(meta_path, "w") as f:
                json.dump({"requested_symbol": "oops"}, f)

            self.assertIsNone(warp._src.build.get_cached_lto_meta(meta_path, "requested_symbol"))

    def test_lto_meta_boolean_value_is_cache_miss(self):
        """A metadata sidecar with a boolean value is a cache miss."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "cached.meta")
            with open(meta_path, "w") as f:
                json.dump({"requested_symbol": True}, f)

            self.assertIsNone(warp._src.build.get_cached_lto_meta(meta_path, "requested_symbol"))

    def test_lto_rebuild_does_not_replace_concurrent_cache_output(self):
        """A rebuild keeps a concurrently-created LTO output unless a sidecar is invalid."""
        lto_symbol = "synthetic_concurrent_lto_symbol"
        h = warp._src.build.hash_symbol(lto_symbol)[: warp._src.build.LTO_CACHE_KEY_LENGTH]

        def compile_lto(temp_paths):
            lto_data = b"rebuilt"
            with open(temp_paths[".lto"], "wb") as f:
                f.write(lto_data)

            with open(lto_path, "wb") as f:
                f.write(b"concurrent")

            return True, {".lto": lto_data}

        with tempfile.TemporaryDirectory() as tmp:
            warp.config.kernel_cache_dir = tmp
            lto_dir = warp._src.build.get_lto_cache_dir()
            os.makedirs(lto_dir)
            lto_path = os.path.join(lto_dir, f"{h}.lto")

            result, lto_data = warp._src.build._build_lto_base(lto_symbol, compile_lto, builder=None)

            with open(lto_path, "rb") as f:
                cached_lto_data = f.read()

        self.assertTrue(result)
        self.assertEqual(lto_data, b"rebuilt")
        self.assertEqual(cached_lto_data, b"concurrent")

    def test_lto_rebuild_replaces_invalid_meta_sidecar(self):
        """A rebuilt LTO cache entry heals an invalid metadata sidecar."""
        lto_symbol = "fft_64_4_70_forward_5"
        shared_memory_bytes = 256
        h = warp._src.build.hash_symbol(lto_symbol)[: warp._src.build.LTO_CACHE_KEY_LENGTH]

        def compile_lto(temp_paths):
            lto_data = b"rebuilt"
            with open(temp_paths[".lto"], "wb") as f:
                f.write(lto_data)
            with open(temp_paths[".meta"], "w") as f:
                json.dump({lto_symbol: shared_memory_bytes}, f)
            return True, {".lto": lto_data, ".meta": shared_memory_bytes}

        with tempfile.TemporaryDirectory() as tmp:
            warp.config.kernel_cache_dir = tmp
            lto_dir = warp._src.build.get_lto_cache_dir()
            os.makedirs(lto_dir)

            lto_path = os.path.join(lto_dir, f"{h}.lto")
            meta_path = os.path.join(lto_dir, f"{h}.meta")
            with open(lto_path, "wb") as f:
                f.write(b"cached")
            with open(meta_path, "w") as f:
                json.dump({"other_symbol": 128}, f)

            result, lto_data, cached_shared_memory_bytes = warp._src.build._build_lto_base(
                lto_symbol,
                compile_lto,
                builder=None,
                extra_files={".meta": lambda path: warp._src.build.get_cached_lto_meta(path, lto_symbol)},
            )

            with open(meta_path) as f:
                healed_meta = json.load(f)
            with open(lto_path, "rb") as f:
                healed_lto_data = f.read()

        self.assertTrue(result)
        self.assertEqual(lto_data, b"rebuilt")
        self.assertEqual(cached_shared_memory_bytes, shared_memory_bytes)
        self.assertEqual(healed_meta, {lto_symbol: shared_memory_bytes})
        self.assertEqual(healed_lto_data, b"rebuilt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
