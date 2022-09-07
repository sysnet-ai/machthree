// Copyright 2016 Revolution Solid & Contributors.
// author(s): sysnett
// TODO: This should exist as a separate small crate

//! Random Numbers Util
//!
//! Wrapper around the rand crate that provides a Seeded
//! and Stateful Random Number Generator.
//!
//! Internally uses rand::XorShiftRng for speed purposes.
//!
//! # Examples
//!
//! RandomCtx - Basic Use case
//!
use rand::{Rng, Rand, SeedableRng, XorShiftRng};
use rand::distributions::range::SampleRange;

use std::fmt;

pub type Seed = [u32; 4];
pub struct RandomCtx
{
    seed: Seed,
    rng:  XorShiftRng,
    name: String,
    seeded: bool,
    values_generated: u32
}

#[allow(dead_code)]
impl RandomCtx
{
// Constructors 
    pub fn new_unseeded(name: String) -> RandomCtx
    {
        let std_rng = XorShiftRng::new_unseeded();
        RandomCtx
        {
            seed: [0; 4],
            rng: std_rng,
            name: name,
            seeded: false,
            values_generated: 0
        }
    }

    pub fn from_seed(seed: Seed, name: String) -> RandomCtx
    {
        let std_rng = SeedableRng::from_seed(seed); 
        RandomCtx
        {
            seed: seed,
            rng:  std_rng,
            name: name,
            seeded: true,
            values_generated: 0
        }
    }

// Random Values - Subset of the RNG Trait
    pub fn gen<T: Rand>(&mut self) -> T where Self: Sized
    {
        self.values_generated += 1;
        self.rng.gen()
    }

    pub fn gen_range<T: PartialOrd + SampleRange>(&mut self, low: T, high: T) -> T
    {
        self.values_generated += 1;
        self.rng.gen_range(low, high)
    }

    pub fn next_u32(&mut self) -> u32 { self.gen::<u32>() }
    pub fn next_u64(&mut self) -> u64 { self.gen::<u64>() }
    pub fn next_f32(&mut self) -> f32 { self.gen::<f32>() }
    pub fn next_f64(&mut self) -> f64 { self.gen::<f64>() }

    pub fn shuffle<T>(&mut self, values: &mut [T]) where Self: Sized, T: Copy
    {
        for i in 0..values.len()-2
        {
            let j = self.gen_range(i, values.len());
            let t = values[i];
            values[i] = values[j];
            values[j] = t;
        }
    }

// Random Values - RandomCtx functions
    pub fn test_value<T: PartialOrd + Rand>(&mut self, value: T) -> bool 
    {
        self.gen::<T>() < value
    }


// Reset State
    pub fn reseed(&mut self, seed: Seed)
    {
        self.seed = seed;
        self.seeded = true;
        self.reset();
    }

    pub fn reset(&mut self)
    {
        self.values_generated = 0;
        if self.seeded
        {
            self.rng.reseed(self.seed);
        }
        else
        {
            self.rng = XorShiftRng::new_unseeded(); 
        }
    }
}

impl fmt::Debug for RandomCtx
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let seeded_str = if self.seeded
            {
                "Seeded"
            }
            else
            {
                "Not Seeded"
            };

        write!(f, "RandomCtx {} - {} {{ seed: {:?}, values_generated: {:?} }}",
               self.name,
               seeded_str,
               self.seed,
               self.values_generated)
    }
}

////////////////////////////////////////
// Tests
#[cfg(test)]
mod test
{
    use super::{Seed, RandomCtx};

    #[test]
    fn same_seed()
    {
        let seed : Seed = [1,2,3,4];
        let mut ctx = RandomCtx::from_seed(seed, String::from("TestRandomCtx")); 
        let mut ctx_2 = RandomCtx::from_seed(seed, String::from("TestRandomCtx2")); 
//        debug!("{:?}", ctx);
//        debug!("{:?}", ctx_2);

        for _ in 0..100
        {
            assert_eq!(ctx.gen::<f64>(), ctx_2.gen::<f64>());
        }
//        debug!("{:?}", ctx);
//        debug!("{:?}", ctx_2);

        for _ in 0..100
        {
            assert_eq!(ctx.gen::<u32>(), ctx_2.gen::<u32>());
        }
//        debug!("{:?}", ctx);
//        debug!("{:?}", ctx_2);
    }

    #[test]
    fn diff_seed()
    {
        let seed_1 : Seed = [1,2,3,4];
        let seed_2 : Seed = [4,3,2,1];
        let mut ctx = RandomCtx::from_seed(seed_1, String::from("TestRandomCtx")); 
        let mut ctx_2 = RandomCtx::from_seed(seed_2, String::from("TestRandomCtx2")); 
//        debug!("{:?}", ctx);
//        debug!("{:?}", ctx_2);

        for _ in 0..100
        {
            assert!(ctx.gen::<f32>() != ctx_2.gen::<f32>());
        }
//        debug!("{:?}", ctx);
//        debug!("{:?}", ctx_2);

        for _ in 0..100
        {
            assert!(ctx.gen::<u64>() != ctx_2.gen::<u64>());
        }
//        debug!("{:?}", ctx);
//        debug!("{:?}", ctx_2);
    }

    #[test]
    fn same_seed_different_types()
    {
        let seed_1 = [1; 4];
        let mut ctx = RandomCtx::from_seed(seed_1, String::from("TestRandomCtx")); 
        let mut ctx_2 = RandomCtx::from_seed(seed_1, String::from("TestRandomCtx")); 
//        debug!("{:?}", ctx.gen::<f32>()); 
//        debug!("{:?}", ctx_2.gen::<i8>()); 
        assert_eq!(ctx.gen::<f32>(), ctx_2.gen::<f32>());
    }
}
