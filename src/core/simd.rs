// src/core/simd.rs

/// Extreme-performance hardware acceleration abstraction layer mapping high-dimensional
/// mathematical operations directly onto bare-metal CPU register pipelines.

// ============================================================================================
// X86_64 ARCHITECTURE: INTEL & AMD AVX2 / FMA HARDWARE TRACKS
// ============================================================================================
#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use std::arch::x86_64::*;

    /// Computes an accelerated Dot Product over F32 slices using 256-bit AVX2 and FMA intrinsics.
    ///
    /// Drives 4 separate internal CPU execution ports simultaneously by leveraging 4x explicit
    /// loop unrolling to saturate the pipeline and minimize instruction dependency stalls.
    ///
    /// # Safety
    /// This function is unsafe because it uses targeted CPU hardware intrinsics that can cause
    /// illegal instruction crashes if executed on microarchitectures lacking AVX2 or FMA support.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();

        // Edge Case Protection: If vector dimensionality is lower than a single AVX2 register lane (8 floats),
        // instantly short-circuit to a clean scalar compute pass to prevent register tail pollution.
        if len < 8 {
            return a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        }

        // Initialize four independent accumulator lanes to maximize instruction-level parallelism
        let mut sum_vec0 = _mm256_setzero_ps();
        let mut sum_vec1 = _mm256_setzero_ps();
        let mut sum_vec2 = _mm256_setzero_ps();
        let mut sum_vec3 = _mm256_setzero_ps();

        let rem = len % 32;
        let main_len = len - rem;

        // Main unrolled computation loop processing 32 elements (128 bytes) per single iteration step
        for i in (0..main_len).step_by(32) {
            unsafe {
                let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
                sum_vec0 = _mm256_fmadd_ps(va0, vb0, sum_vec0);

                let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                sum_vec1 = _mm256_fmadd_ps(va1, vb1, sum_vec1);

                let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
                let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
                sum_vec2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
                sum_vec2 = _mm256_fmadd_ps(va2, vb2, sum_vec2);

                let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
                let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
                sum_vec3 = _mm256_fmadd_ps(va3, vb3, sum_vec3);
            }
        }

        // Fold and reduce unrolled streams back into a single vector lane register
        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec1);
        sum_vec2 = _mm256_add_ps(sum_vec2, sum_vec3);
        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec2);

        // Process remaining single 8-element AVX2 register block alignment boundaries
        let rem2 = rem % 8;
        let main_len2 = rem - rem2;
        for i in (main_len..main_len + main_len2).step_by(8) {
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                sum_vec0 = _mm256_fmadd_ps(va, vb, sum_vec0);
            }
        }

        // Extrapolate vector register buffers back onto scalar memory tracks
        let mut buffer = [0.0; 8];
        unsafe {
            _mm256_storeu_ps(buffer.as_mut_ptr(), sum_vec0);
        }
        let mut sum = buffer.iter().sum::<f32>();

        // Clean up remaining localized tail elements flawlessly
        for i in (len - rem2)..len {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Computes an accelerated Euclidean squared error sum using 256-bit AVX2 lanes.
    ///
    /// # Safety
    /// Caller must guarantee target processor architectures support standard AVX2 feature extensions.
    #[target_feature(enable = "avx2")]
    pub unsafe fn euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        if len < 8 {
            return a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum();
        }

        let mut sum_vec = _mm256_setzero_ps();
        let rem = len % 8;
        let main_len = len - rem;

        // Process 8 floating point differences concurrently via SIMD subtraction lanes
        for i in (0..main_len).step_by(8) {
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let diff = _mm256_sub_ps(va, vb);
                sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
            }
        }

        let mut buffer = [0.0; 8];
        unsafe {
            _mm256_storeu_ps(buffer.as_mut_ptr(), sum_vec);
        }
        let mut sum = buffer.iter().sum::<f32>();

        // Clean up remaining matrix layout tail elements sequential paths
        for i in main_len..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }

    /// Optimized ultra-fast integer-math dot product over F8 (u8) quantized arrays.
    ///
    /// Employs optimized `_mm256_maddubs_epi16` and vertical `_mm256_sad_epu8` hardware blocks
    /// to perform lightning-fast horizontal additions over byte-quantized memory pools.
    ///
    /// # Safety
    /// Will trigger an illegal instruction fault if called on microprocessors lacking AVX2 instruction lines.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_f8(a: &[u8], b: &[u8], sums: &mut (u32, u32, u32)) {
        let len = a.len();
        let rem = len % 32;
        let main_len = len - rem;

        let mut acc_ab = _mm256_setzero_si256();
        let mut acc_a = _mm256_setzero_si256();
        let mut acc_b = _mm256_setzero_si256();

        let ones_16 = _mm256_set1_epi16(1);
        let zero_vec = _mm256_setzero_si256();

        // Process packed byte chunks (32 dimensions per step) via localized integer math operations
        for i in (0..main_len).step_by(32) {
            unsafe {
                let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

                // Multiply signed/unsigned bytes and accumulate adjacent pairs into 16-bit integers
                let intermediate = _mm256_maddubs_epi16(va, vb);
                acc_ab = _mm256_add_epi32(acc_ab, _mm256_madd_epi16(intermediate, ones_16));

                // Compute sum of absolute differences against zero vector to perform fast horizontal byte summation
                let sad_a = _mm256_sad_epu8(va, zero_vec);
                acc_a = _mm256_add_epi64(acc_a, sad_a);

                let sad_b = _mm256_sad_epu8(vb, zero_vec);
                acc_b = _mm256_add_epi64(acc_b, sad_b);
            }
        }

        let mut buf_ab = [0i32; 8];
        let mut buf_a = [0i64; 4];
        let mut buf_b = [0i64; 4];

        unsafe {
            _mm256_storeu_si256(buf_ab.as_mut_ptr() as *mut __m256i, acc_ab);
            _mm256_storeu_si256(buf_a.as_mut_ptr() as *mut __m256i, acc_a);
            _mm256_storeu_si256(buf_b.as_mut_ptr() as *mut __m256i, acc_b);
        }

        sums.0 += buf_ab.iter().map(|&x| x as u32).sum::<u32>();
        sums.1 += buf_a.iter().map(|&x| x as u32).sum::<u32>();
        sums.2 += buf_b.iter().map(|&x| x as u32).sum::<u32>();

        // Flush remaining alignment tail properties smoothly
        for i in main_len..len {
            sums.0 += (a[i] as u32) * (b[i] as u32);
            sums.1 += a[i] as u32;
            sums.2 += b[i] as u32;
        }
    }
}

// ============================================================================================
// AARCH64 ARCHITECTURE: APPLE SILICON & ARM NEON HARDWARE TRACKS
// ============================================================================================
#[cfg(target_arch = "aarch64")]
pub mod arm {
    use std::arch::aarch64::*;

    /// Computes an accelerated Dot Product over F32 vector slices utilizing 128-bit ARM Neon intrinsics.
    ///
    /// # Safety
    /// This function is unsafe because it executes direct low-level bare-metal ARM processing operations.
    pub unsafe fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        if len < 4 {
            return a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        }

        let mut sum_vec0 = unsafe { vdupq_n_f32(0.0) };
        let mut sum_vec1 = unsafe { vdupq_n_f32(0.0) };
        let rem = len % 8;
        let main_len = len - rem;

        // VLA instruction pipelines processing dual 4-lane float registries sequentially
        for i in (0..main_len).step_by(8) {
            unsafe {
                let va0 = vld1q_f32(a.as_ptr().add(i));
                let vb0 = vld1q_f32(b.as_ptr().add(i));
                sum_vec0 = vmlaq_f32(sum_vec0, va0, vb0);

                let va1 = vld1q_f32(a.as_ptr().add(i + 4));
                let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
                sum_vec1 = vmlaq_f32(sum_vec1, va1, vb1);
            }
        }

        sum_vec0 = unsafe { vaddq_f32(sum_vec0, sum_vec1) };

        let rem2 = rem % 4;
        let main_len2 = rem - rem2;
        for i in (main_len..main_len + main_len2).step_by(4) {
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                sum_vec0 = vmlaq_f32(sum_vec0, va, vb);
            }
        }

        // Horizontal register lane accumulation pass
        let mut sum = unsafe {
            vgetq_lane_f32(sum_vec0, 0)
                + vgetq_lane_f32(sum_vec0, 1)
                + vgetq_lane_f32(sum_vec0, 2)
                + vgetq_lane_f32(sum_vec0, 3)
        };

        for i in (len - rem2)..len {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Computes an accelerated Euclidean squared error sum using 128-bit ARM Neon vector registries.
    pub unsafe fn euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        if len < 4 {
            return a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum();
        }

        let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
        let rem = len % 4;
        let main_len = len - rem;

        for i in (0..main_len).step_by(4) {
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let diff = vsubq_f32(va, vb);
                sum_vec = vmlaq_f32(sum_vec, diff, diff);
            }
        }

        let mut sum = unsafe {
            vgetq_lane_f32(sum_vec, 0)
                + vgetq_lane_f32(sum_vec, 1)
                + vgetq_lane_f32(sum_vec, 2)
                + vgetq_lane_f32(sum_vec, 3)
        };

        for i in main_len..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }
}
