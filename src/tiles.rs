use std::ops::{BitAnd, BitOr, Not};

#[derive(Copy, Clone, Debug, Default)]
pub struct TileMask(pub u64);
impl BitAnd for TileMask
{
    type Output = Self;
    fn bitand(self, Self(rhs): Self) -> Self::Output
    {
        return TileMask( self.0 & rhs )
    }
}

impl BitOr for TileMask
{
    type Output = Self;
    fn bitor(self, Self(rhs): Self) -> Self::Output
    {
        return TileMask( self.0 | rhs )
    }
}

impl Not for TileMask
{
    type Output = Self;
    fn not(self) -> Self::Output
    {
        TileMask( !self.0 )
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct TileData
{
    pub mask: TileMask,
    pub health: usize
}

impl TileData
{
    pub fn empty() -> Self
    {
        Self::new(0,0)
    }
    pub fn new(mask: u64, health: usize) -> Self
    {
        Self { mask: TileMask(mask), health }
    }
    pub fn tiles_match(tile_a: &Self, tile_b: &Self) -> bool
    {
        (tile_a.mask & tile_b.mask).0 != 0
    }
    
    pub fn mask_match(&self, mask: &TileMask) -> bool
    {
        self.mask.0 & mask.0 != 0
    }
    pub fn clear(&mut self)
    {
        self.mask = TileMask(0);
        self.health = 0;
    }
    pub fn is_empty(&self) -> bool
    {
        self.mask.0 & 0xFFFFFFFF == 0 //TODO: Better ways of representing 'ALL MASKS'
    }
}

impl std::fmt::Display for TileData
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        formatter.write_str( &self.mask.0.to_string() )
    }
}
