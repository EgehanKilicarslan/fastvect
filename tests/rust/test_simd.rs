// tests/rust/test_simd.rs

use fastvect::core::simd;

/// Verifies that the 4x Loop Unrolled F32 SIMD Dot Product yields mathematically
/// identical results compared to a standard scalar loop, including edge-case tail remnants.
///
/// Evaluates memory lane alignment behaviors over non-standard vector dimensionalities
/// to ensure register tail processing loops prevent calculation pollution.
#[test]
fn test_simd_dot_product_f32_alignment_and_tail() {
    // Dimension chosen to deliberately create a remainder (13 elements, not a multiple of 8 or 32)
    // to rigorously force the instruction loop tail handlers to run.
    let vec_a = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ];
    let vec_b = vec![
        0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0,
    ];

    // Standard scalar baseline calculation running over normal iterator zip lines
    let expected_score: f32 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| x * y).sum();

    // Verify x86_64 intrinsic executions if active hardware runtime support is detected
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let simd_score = unsafe { simd::x86::dot_product_f32(&vec_a, &vec_b) };
            assert!(
                (simd_score - expected_score).abs() < 1e-5,
                "AVX2 F32 Dot Product precision mismatch!"
            );
        }
    }

    // Verify ARM64 Neon intrinsic executions over corresponding hardware targets
    #[cfg(target_arch = "aarch64")]
    {
        let simd_score = unsafe { simd::arm::dot_product_f32(&vec_a, &vec_b) };
        assert!(
            (simd_score - expected_score).abs() < 1e-5,
            "ARM NEON F32 Dot Product precision mismatch!"
        );
    }
}

/// Validates that the bitwise mask/shift packed F8 SIMD dot product computes
/// correct integer sums across internal 32-bit register boundaries.
///
/// Ensures both horizontal byte accumulations and vertical product reductions
/// match standard precision mappings perfectly without 16-bit register saturation overflows.
#[test]
fn test_simd_dot_product_f8_integer_accumulation() {
    // 35 elements ensures the unrolled loop (32 bytes) fires exactly once
    // and the cleanup scalar loop handles the remaining 3 elements.
    let bytes_a = vec![255; 35];
    let bytes_b = vec![2; 35];

    // Build standard integer baseline matrix tracking values via scalar iteration loops
    let mut expected_sums = (0u32, 0u32, 0u32);
    for (&x, &y) in bytes_a.iter().zip(bytes_b.iter()) {
        expected_sums.0 += (x as u32) * (y as u32);
        expected_sums.1 += x as u32;
        expected_sums.2 += y as u32;
    }

    let mut simd_sums = (0u32, 0u32, 0u32);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                simd::x86::dot_product_f8(&bytes_a, &bytes_b, &mut simd_sums);
            }
            assert_eq!(
                simd_sums.0, expected_sums.0,
                "AVX2 F8 cross-product accumulation failure!"
            );
            assert_eq!(
                simd_sums.1, expected_sums.1,
                "AVX2 F8 vector A horizontal sum failure!"
            );
            assert_eq!(
                simd_sums.2, expected_sums.2,
                "AVX2 F8 vector B horizontal sum failure!"
            );
        }
    }
}

/// Confirms that the accelerated Euclidean squared error SIMD routine tracks
/// dimensional variance accurately down to individual floating-point boundaries.
#[test]
fn test_simd_euclidean_squared_f32() {
    let vec_a = vec![5.0, 10.0, 15.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.5];
    let vec_b = vec![3.0, 6.0, 9.0, 12.0, 1.0, 2.0, 3.0, 4.0, 0.5];

    // Compute absolute spatial differential variance via standard mathematical formulas
    let expected_sq_dist: f32 = vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let simd_sq_dist = unsafe { simd::x86::euclidean_sq_f32(&vec_a, &vec_b) };
            assert!(
                (simd_sq_dist - expected_sq_dist).abs() < 1e-5,
                "AVX2 F32 Euclidean squared error mismatch!"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let simd_sq_dist = unsafe { simd::arm::euclidean_sq_f32(&vec_a, &vec_b) };
        assert!(
            (simd_sq_dist - expected_sq_dist).abs() < 1e-5,
            "ARM NEON F32 Euclidean squared error mismatch!"
        );
    }
}
