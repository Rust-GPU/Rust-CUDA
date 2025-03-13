use crate::xoroshiro::Xoroshiro128StarStar;
use rand_core::{RngCore, SeedableRng};

/// Default random generator which is good for most applications.
///
/// This currently uses [`Xoroshiro128StarStar`], but that may be changed in the future (with a major version bump).
#[cfg_attr(not(target_os = "cuda"), derive(Copy, cust_core::DeviceCopy))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct DefaultRand {
    inner: Xoroshiro128StarStar,
}

impl DefaultRand {
    /// Initializes many states such that each state is offset in the main sequence by at least
    /// `2**64` elements (based on the current default generator). Such that every state is independent
    /// from the others as long as no state requests more than `2**64` random numbers.
    #[cfg_attr(docsrs, doc(cfg(not(target_os = "cuda"))))]
    #[cfg(not(target_os = "cuda"))]
    pub fn initialize_states(seed: u64, num_states: usize) -> Vec<Self> {
        Xoroshiro128StarStar::initialize_states(seed, num_states)
            .into_iter()
            .map(|inner| Self { inner })
            .collect()
    }
}

impl RngCore for DefaultRand {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest)
    }
}

impl SeedableRng for DefaultRand {
    type Seed = <Xoroshiro128StarStar as SeedableRng>::Seed;

    fn seed_from_u64(state: u64) -> Self {
        Self {
            inner: Xoroshiro128StarStar::seed_from_u64(state),
        }
    }

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            inner: Xoroshiro128StarStar::from_seed(seed),
        }
    }
}
