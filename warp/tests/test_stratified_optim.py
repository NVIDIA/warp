# Ensure all constants and operations stay in float32 to match kernel precision
        delta_f32 = np.float32(0.05)
        angles = np.clip(
            (data * delta_f32) / np.float32(4.0), 
            np.float32(-1.5707963), 
            np.float32(1.5707963)
        )
        expected = data * np.cos(angles).astype(np.float32)

        # Use a slightly wider tolerance appropriate for float32 precision
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
