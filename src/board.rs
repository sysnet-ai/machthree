use crate::tiles;

pub type BoardPosition = (usize, usize);

pub struct Goal;

#[derive(Debug)]
pub struct Board
{
    tiles: Vec<tiles::TileData>,
    dimensions: (usize, usize)
}

impl std::fmt::Display for Board
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        for row in (0..self.height())
        {
            for col in (0..self.width())
            {
                let td = self.get_tile(&(col, row)); 
                formatter.write_fmt(format_args!("|{}|", td));
            }
            formatter.write_str("\n");
            for col in (0..self.width())
            {
                formatter.write_str("---");
            }
            formatter.write_str("\n");
        }
        return std::fmt::Result::Ok(())
    }
}

impl Board
{
    pub fn new(dimensions: (usize, usize)) -> Self
    {
        let mut t = Vec::new();
        t.resize(dimensions.0 * dimensions.1, tiles::TileData::empty());
        Self { tiles: t, dimensions }
    }

    pub fn width(&self) -> usize
    {
        self.dimensions.0
    }

    pub fn height(&self) -> usize
    {
        self.dimensions.1
    }

    pub fn get_tile<'a>(&'a self, at: &BoardPosition) -> &'a tiles::TileData
    {
        let (x, y) = (at.0, at.1);
        &self.tiles[y * self.width() + x]
    }

    pub fn get_tile_mut<'a>(&'a mut self, at: &BoardPosition) -> &'a mut tiles::TileData
    {
        let (x, y) = (at.0, at.1);
        let inx = y * self.width() + x;
        &mut self.tiles[inx]
    }

    pub fn set_tile(&mut self, tile: tiles::TileData, at: &BoardPosition)
    {
        let (x, y) = (at.0, at.1);
        let inx = y * self.width() + x;
        self.tiles[inx] = tile;
    }
}
