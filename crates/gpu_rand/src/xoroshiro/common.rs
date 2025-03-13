/// Initialize a RNG from a `u64` seed using `SplitMix64`.
macro_rules! from_splitmix {
    ($seed:expr) => {{
        let mut rng = crate::xoroshiro::SplitMix64::seed_from_u64($seed);
        Self::from_rng(&mut rng)
    }};
}

/// Apply the ** scrambler used by some RNGs from the xoshiro family.
macro_rules! starstar_u64 {
    ($x:expr) => {
        $x.wrapping_mul(5).rotate_left(7).wrapping_mul(9)
    };
}

/// Apply the ** scrambler used by some RNGs from the xoshiro family.
macro_rules! starstar_u32 {
    ($x:expr) => {
        $x.wrapping_mul(0x9E3779BB).rotate_left(5).wrapping_mul(5)
    };
}

/// Apply the ++ scrambler used by some RNGs from the xoshiro family.
macro_rules! plusplus_u64 {
    ($x:expr, $y:expr, $rot:expr) => {
        $x.wrapping_add($y).rotate_left($rot).wrapping_add($x)
    };
}

/// Apply the ++ scrambler used by some RNGs from the xoshiro family.
macro_rules! plusplus_u32 {
    ($x:expr, $y:expr) => {
        $x.wrapping_add($y).rotate_left(7).wrapping_add($x)
    };
}

/// Implement a jump function for an RNG from the xoshiro family.
macro_rules! impl_jump {
    (u32, $self:expr, [$j0:expr, $j1:expr]) => {
        const JUMP: [u32; 2] = [$j0, $j1];
        let mut s0 = 0;
        let mut s1 = 0;
        for j in &JUMP {
            for b in 0..32 {
                if (j & 1 << b) != 0 {
                    s0 ^= $self.s0;
                    s1 ^= $self.s1;
                }
                $self.next_u32();
            }
        }
        $self.s0 = s0;
        $self.s1 = s1;
    };
    (u64, $self:expr, [$j0:expr, $j1:expr]) => {
        const JUMP: [u64; 2] = [$j0, $j1];
        let mut s0 = 0;
        let mut s1 = 0;
        for j in &JUMP {
            for b in 0..64 {
                if (j & 1 << b) != 0 {
                    s0 ^= $self.s0;
                    s1 ^= $self.s1;
                }
                $self.next_u64();
            }
        }
        $self.s0 = s0;
        $self.s1 = s1;
    };
    (u32, $self:expr, [$j0:expr, $j1:expr, $j2:expr, $j3:expr]) => {
        const JUMP: [u32; 4] = [$j0, $j1, $j2, $j3];
        let mut s0 = 0;
        let mut s1 = 0;
        let mut s2 = 0;
        let mut s3 = 0;
        for j in &JUMP {
            for b in 0..32 {
                if (j & 1 << b) != 0 {
                    s0 ^= $self.s[0];
                    s1 ^= $self.s[1];
                    s2 ^= $self.s[2];
                    s3 ^= $self.s[3];
                }
                $self.next_u32();
            }
        }
        $self.s[0] = s0;
        $self.s[1] = s1;
        $self.s[2] = s2;
        $self.s[3] = s3;
    };
    (u64, $self:expr, [$j0:expr, $j1:expr, $j2:expr, $j3:expr]) => {
        const JUMP: [u64; 4] = [$j0, $j1, $j2, $j3];
        let mut s0 = 0;
        let mut s1 = 0;
        let mut s2 = 0;
        let mut s3 = 0;
        for j in &JUMP {
            for b in 0..64 {
                if (j & 1 << b) != 0 {
                    s0 ^= $self.s[0];
                    s1 ^= $self.s[1];
                    s2 ^= $self.s[2];
                    s3 ^= $self.s[3];
                }
                $self.next_u64();
            }
        }
        $self.s[0] = s0;
        $self.s[1] = s1;
        $self.s[2] = s2;
        $self.s[3] = s3;
    };
    (u64, $self:expr, [$j0:expr, $j1:expr, $j2:expr, $j3:expr,
                       $j4:expr, $j5:expr, $j6:expr, $j7:expr]) => {
        const JUMP: [u64; 8] = [$j0, $j1, $j2, $j3, $j4, $j5, $j6, $j7];
        let mut s = [0; 8];
        for j in &JUMP {
            for b in 0..64 {
                if (j & 1 << b) != 0 {
                    s[0] ^= $self.s[0];
                    s[1] ^= $self.s[1];
                    s[2] ^= $self.s[2];
                    s[3] ^= $self.s[3];
                    s[4] ^= $self.s[4];
                    s[5] ^= $self.s[5];
                    s[6] ^= $self.s[6];
                    s[7] ^= $self.s[7];
                }
                $self.next_u64();
            }
        }
        $self.s = s;
    };
}

/// Implement the xoroshiro iteration.
macro_rules! impl_xoroshiro_u32 {
    ($self:expr) => {
        $self.s1 ^= $self.s0;
        $self.s0 = $self.s0.rotate_left(26) ^ $self.s1 ^ ($self.s1 << 9);
        $self.s1 = $self.s1.rotate_left(13);
    };
}

/// Implement the xoroshiro iteration.
macro_rules! impl_xoroshiro_u64 {
    ($self:expr) => {
        $self.s1 ^= $self.s0;
        $self.s0 = $self.s0.rotate_left(24) ^ $self.s1 ^ ($self.s1 << 16);
        $self.s1 = $self.s1.rotate_left(37);
    };
}

/// Implement the xoroshiro iteration for the ++ scrambler.
macro_rules! impl_xoroshiro_u64_plusplus {
    ($self:expr) => {
        $self.s1 ^= $self.s0;
        $self.s0 = $self.s0.rotate_left(49) ^ $self.s1 ^ ($self.s1 << 21);
        $self.s1 = $self.s1.rotate_left(28);
    };
}

/// Implement the xoshiro iteration for `u32` output.
macro_rules! impl_xoshiro_u32 {
    ($self:expr) => {
        let t = $self.s[1] << 9;

        $self.s[2] ^= $self.s[0];
        $self.s[3] ^= $self.s[1];
        $self.s[1] ^= $self.s[2];
        $self.s[0] ^= $self.s[3];

        $self.s[2] ^= t;

        $self.s[3] = $self.s[3].rotate_left(11);
    };
}

/// Implement the xoshiro iteration for `u64` output.
macro_rules! impl_xoshiro_u64 {
    ($self:expr) => {
        let t = $self.s[1] << 17;

        $self.s[2] ^= $self.s[0];
        $self.s[3] ^= $self.s[1];
        $self.s[1] ^= $self.s[2];
        $self.s[0] ^= $self.s[3];

        $self.s[2] ^= t;

        $self.s[3] = $self.s[3].rotate_left(45);
    };
}

/// Implement the large-state xoshiro iteration.
macro_rules! impl_xoshiro_large {
    ($self:expr) => {
        let t = $self.s[1] << 11;

        $self.s[2] ^= $self.s[0];
        $self.s[5] ^= $self.s[1];
        $self.s[1] ^= $self.s[2];
        $self.s[7] ^= $self.s[3];
        $self.s[3] ^= $self.s[4];
        $self.s[4] ^= $self.s[5];
        $self.s[0] ^= $self.s[6];
        $self.s[6] ^= $self.s[7];

        $self.s[6] ^= t;

        $self.s[7] = $self.s[7].rotate_left(21);
    };
}

/// Map an all-zero seed to a different one.
macro_rules! deal_with_zero_seed {
    ($seed:expr, $Self:ident) => {
        if $seed.iter().all(|&x| x == 0) {
            return $Self::seed_from_u64(0);
        }
    };
}

#[allow(unused_macros)]
macro_rules! impl_initialize_states {
    ($seed:expr, $num_states:expr) => {{
        // there is unfortunately not a well-performant, clean, and safe way to do
        // what we need to do.
        //
        // We need to go through every element after the first element, and clone the
        // previous state, call jump on the state, and make it the current element.
        // This is fundamentally extremely difficult for rust to statically prove, without bounds
        //
        // Because this function will likely be used for hundreds of thousands of states, we need
        // this to not unnecessarily clone or add unecessary bounds checks, so we implement this using
        // unsafe. However, the code is well documented so it's fine.

        use std::cell::UnsafeCell;
        let states = vec![Self::seed_from_u64($seed); $num_states];

        // we need to use UnsafeCell because we need to index into the vector using an immutable
        // reference to it, otherwise we would be aliasing the mutable ref to the vec.

        // see explanation on the bottom of the function on why we cant just transmute.
        let mut me = std::mem::ManuallyDrop::new(states);
        let elems = (me.as_mut_ptr(), me.len(), me.capacity());

        // SAFETY: raw ptr comes from decomposition of the vec so its guaranteed to be fine.
        // The cast is fine because UnsafeCell is repr(transparent).
        let states = unsafe {
            Vec::from_raw_parts(
                elems.0.cast::<UnsafeCell<Self>>(),
                elems.1,
                elems.2
            )
        };

        // skip the first element so we don't try to index into -1 for the first
        // element.
        for i in 1..$num_states {
            // SAFETY: this access cannot be OOB because we know the vec will be long enough
            // and we skipped the 0th element which has no previous element.
            let prev = unsafe {
                &*states.get_unchecked(i - 1).get()
            };
            // SAFETY: similar to the previous access, this cannot be OOB because we know
            // the vec is long enough.
            let cur = unsafe {
                &mut *states.get_unchecked(i).get()
            };

            let mut cloned = prev.clone();
            cloned.jump();
            *cur = cloned;
        }

        // technically we can't transmute Vec<UnsafeCell<T>> to Vec<T> because Vec is allowed
        // to represent the vecs differently. In practice this will never happen but we don't do it either ways.
        // instead we decompose then recompose the vec.

        // Vec::into_raw_parts is not stable, this is literally its function body.
        let mut me = std::mem::ManuallyDrop::new(states);
        let elems = (me.as_mut_ptr(), me.len(), me.capacity());

        // SAFETY: the pointer comes from the decomposition of the vec so its guaranteed
        // to be valid.
        unsafe {
            Vec::from_raw_parts(
                elems.0.cast(),
                elems.1,
                elems.2
            )
        }
    }};
}

/// 512-bit seed for a generator.
///
/// This wrapper is necessary, because some traits required for a seed are not
/// implemented on large arrays.
#[derive(Clone)]
pub struct Seed512(pub [u8; 64]);

impl Seed512 {
    /// Return an iterator over the seed.
    pub fn iter(&self) -> core::slice::Iter<u8> {
        self.0.iter()
    }
}

impl core::fmt::Debug for Seed512 {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        self.0[..].fmt(f)
    }
}

impl Default for Seed512 {
    fn default() -> Seed512 {
        Seed512([0; 64])
    }
}

impl AsMut<[u8]> for Seed512 {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl AsRef<[u8]> for Seed512 {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
