use crate::board;
use crate::tiles;
use log::{warn, info, debug};

pub struct GravitySystem
{
    direction: u8
}

impl GravitySystem
{
    pub fn new() -> Self
    {
        Self { direction: 0 } //TODO: Direction is probably a top level type or Enum
    }

    pub fn apply_gravity(&self, board: &mut board::Board)
    {

        /* Apply Gravity Algorithm:
         * For each Column:
         *    Find the first empty space bottom to top.
         *    Count the number of empty spaces to the first non-empty space
         *    Move any pieces above the empty spaces down
         *      Use an offset to do this (i.e. the number of empty spaces is 2, shift each piece by 2) 
         *
         *    If pieces fell - Repeat this process, else move to the next column.
         */
        //TODO: This could change drastically when directions become a thing,
        //      For now we assume gravity always goes down
        debug!("GRAVITY: Before Application \n{}", board);
        for col in 0..board.width()
        {
            loop
            {
                let (start, end) = Self::find_gravity_boundaries_for_column(board, col);
                if start > 0 && end > -1
                {
                    debug!("GRAVITY: Dropping from {} to {}", start, end);
                    let offset:usize = (start - end).try_into().unwrap();  
                    let ustart:usize = start.try_into().unwrap(); // Are all of these really necessary :/ ?

                    for r in (0..ustart+1).rev()
                    {
                        if offset > r
                        {
                            debug!("GRAVITY: Ending Application -  Offset({}) > Row({})", offset, r);
                            break;
                        }

                        let p1 = (col, r - offset);
                        let p2 = (col, r);

                        let old = 
                        {
                            let old = board.get_tile_mut(&p1);
                            let copy = *old;
                            old.clear();
                            copy
                        };

                        board.set_tile(old, &p2);
                        debug!("GRAVITY: Dropping tile {:?} to {:?}", p1, p2);
                    }
                }
                else
                {
                    break;
                }
            }
        }
        debug!("GRAVITY: After Application \n{}", board);
    }

    fn find_gravity_boundaries_for_column(board: &board::Board, col: usize) -> (isize, isize)
    {
        let mut start:isize =  0;
        let mut end:isize = -1;
        for row in (0..board.height()).rev()
        {
            let td = board.get_tile(&(col, row)); 
            if td.is_empty()
            {
                start = row.try_into().unwrap();
                break;
            }
        }

        for row in (0usize..start.try_into().unwrap()).rev()
        {
            let td = board.get_tile(&(col, row)); 
            if !td.is_empty()
            {
                end = row.try_into().unwrap();
                break;
            }
        }

        (start, end)
    }
}


#[cfg(test)]
mod test
{
    use super::*;
    #[test]
    fn simple_drop()
    {
        let _ = env_logger::builder().is_test(true).try_init();
        /* Board:
         *   0 1 2 3
         * 0 X X X X
         * 1 X X X X
         * 2 1 1 1 1
         * 3 X X X X
         */

        let g = GravitySystem::new();
        let mut board = board::Board::new((4, 4));
        for c in 0..4
        {
            board.set_tile(tiles::TileData::new(1, 1), &(c, 2));
        }

        g.apply_gravity(&mut board);

        for c in 0..4
        {
            assert!(  board.get_tile(&(c, 2)).is_empty(), "GRAVITY-TEST: Tiles fail to fall\n{}", board);
            assert!(! board.get_tile(&(c, 3)).is_empty(), "GRAVITY-TEST: Tiles fail to fall\n{}", board);
            assert_eq!(board.get_tile(&(c, 3)).health, 1, "GRAVITY-TEST: Tiles fail to fall\n{}", board);
        }
    }


    #[test]
    fn staggered_drop() 
    {
        let _ = env_logger::builder().is_test(true).try_init();

        /* Board
         *   0 1 2 3
         * 0 2 X X X
         * 1 X 2 X X
         * 2 1 X X X
         * 3 X 1 X X
         */
        let g = GravitySystem::new();
        let mut board = board::Board::new((4, 4));
        board.set_tile(tiles::TileData::new(2, 1), &(0, 0));
        board.set_tile(tiles::TileData::new(1, 1), &(0, 2));
        board.set_tile(tiles::TileData::new(2, 1), &(1, 1));
        board.set_tile(tiles::TileData::new(1, 1), &(1, 3));

        g.apply_gravity(&mut board);

        assert!( board.get_tile(&(0, 0)).is_empty(), "\n{}", board);
        assert!( board.get_tile(&(1, 1)).is_empty(), "\n{}", board);

        assert!(board.get_tile(&(0, 3)).mask_match(&tiles::TileMask(1)), "GRAVITY-TEST: Incorrect Tile at spot (0,3) \n{}", board);
        assert!(board.get_tile(&(0, 2)).mask_match(&tiles::TileMask(2)), "GRAVITY-TEST: Incorrect Tile at spot (0,2) \n{}", board);
        assert!(board.get_tile(&(1, 2)).mask_match(&tiles::TileMask(2)), "GRAVITY-TEST: Incorrect Tile at spot (1,2) \n{}", board);
        assert!(board.get_tile(&(1, 3)).mask_match(&tiles::TileMask(1)), "GRAVITY-TEST: Incorrect Tile at spot (1,3) \n{}", board);
    }

    #[test]
    fn block_drop()
    {
        let _ = env_logger::builder().is_test(true).try_init();

        /* Board
         *   0 1 2 3
         * 0 2 X X X
         * 1 1 2 X X
         * 2 X 1 X X
         * 3 X X X X
         */
        let g = GravitySystem::new();
        let mut board = board::Board::new((4, 4));
        board.set_tile(tiles::TileData::new(2, 1), &(0, 0));
        board.set_tile(tiles::TileData::new(1, 1), &(0, 2));
        board.set_tile(tiles::TileData::new(2, 1), &(1, 1));
        board.set_tile(tiles::TileData::new(1, 1), &(1, 3));

        g.apply_gravity(&mut board);

        assert!( board.get_tile(&(0, 0)).is_empty(), "\n{}", board);
        assert!( board.get_tile(&(1, 1)).is_empty(), "\n{}", board);

        assert!(board.get_tile(&(0, 3)).mask_match(&tiles::TileMask(1)), "GRAVITY-TEST: Incorrect Tile at spot (0,3) \n{}", board);
        assert!(board.get_tile(&(0, 2)).mask_match(&tiles::TileMask(2)), "GRAVITY-TEST: Incorrect Tile at spot (0,2) \n{}", board);
        assert!(board.get_tile(&(1, 2)).mask_match(&tiles::TileMask(2)), "GRAVITY-TEST: Incorrect Tile at spot (1,2) \n{}", board);
        assert!(board.get_tile(&(1, 3)).mask_match(&tiles::TileMask(1)), "GRAVITY-TEST: Incorrect Tile at spot (1,3) \n{}", board);
    }
}
