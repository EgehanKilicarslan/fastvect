// src/core/quantization.rs

/// Universal 8-bit Scalar Quantizer compressing high-dimensional float vectors down to u8 arrays.
pub struct ScalarQuantizer {
    pub min: f32,
    pub max: f32,
}

impl ScalarQuantizer {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Compresses an entire f32 slice directly into a u8 vector (RAM savings: 4x).
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .map(|&x| {
                let clamped = x.clamp(self.min, self.max);
                let normalized = (clamped - self.min) / (self.max - self.min);
                (normalized * 255.0).round() as u8
            })
            .collect()
    }

    /// Dequantizes a u8 vector back into f32 coordinates for strict precision evaluation gates.
    pub fn dequantize(&self, q_vector: &[u8]) -> Vec<f32> {
        q_vector
            .iter()
            .map(|&x| {
                let normalized = x as f32 / 255.0;
                self.min + normalized * (self.max - self.min)
            })
            .collect()
    }
}
