use core::{f32, f64};
#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
use rand_core::RngCore; // needed for log

fn u64_to_unit_f64(x: u64) -> f64 {
    (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Methods for float random number generation that are common in GPU/massively parallel
/// applications. Such as uniform or normal f32/64 generation.
pub trait GpuRand: RngCore {
    /// Creates an [`prim@f32`] in the range of `[0.0, 1.0)` and advances the state once.
    fn uniform_f32(&mut self) -> f32;
    /// Creates an [`prim@f64`] in the range of `[0.0, 1.0)` and advances the state once.
    fn uniform_f64(&mut self) -> f64;
    /// Creates an [`prim@f32`] with normal distribution. The value is drawn from a Gaussian of
    /// mean=0 and sigma=1 using the Box-Mueller transform. Advances the state twice.
    fn normal_f32(&mut self) -> f32;
    /// Creates an [`prim@f64`] with normal distribution. The value is drawn from a Gaussian of
    /// mean=0 and sigma=1 using the Box-Mueller transform. Advances the state twice.
    fn normal_f64(&mut self) -> f64;
    /// Same as [`Self::normal_f32`] but doesn't discard the second normal value.
    fn normal_f32_2(&mut self) -> [f32; 2];
    /// Same as [`Self::normal_f64`] but doesn't discard the second normal value.
    fn normal_f64_2(&mut self) -> [f64; 2];
}

impl<T: RngCore> GpuRand for T {
    fn uniform_f32(&mut self) -> f32 {
        u64_to_unit_f64(self.next_u64()) as f32
    }

    fn uniform_f64(&mut self) -> f64 {
        u64_to_unit_f64(self.next_u64())
    }

    fn normal_f32(&mut self) -> f32 {
        let u1 = self.uniform_f32();
        let u2 = self.uniform_f32();

        (-2.0 * u1.ln()).sqrt() * ((f32::consts::PI * 2.0) * u2).cos()
    }

    fn normal_f64(&mut self) -> f64 {
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();

        (-2.0 * u1.ln()).sqrt() * ((f64::consts::PI * 2.0) * u2).cos()
    }

    fn normal_f32_2(&mut self) -> [f32; 2] {
        let u1 = self.uniform_f32();
        let u2 = self.uniform_f32();

        [
            (-2.0 * u1.ln()).sqrt() * ((f32::consts::PI * 2.0) * u2).cos(),
            (-2.0 * u1.ln()).sqrt() * ((f32::consts::PI * 2.0) * u2).sin(),
        ]
    }

    fn normal_f64_2(&mut self) -> [f64; 2] {
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();

        [
            (-2.0 * u1.ln()).sqrt() * ((f64::consts::PI * 2.0) * u2).cos(),
            (-2.0 * u1.ln()).sqrt() * ((f64::consts::PI * 2.0) * u2).sin(),
        ]
    }
}
