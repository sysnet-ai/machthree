use std::collections::{HashSet};
use log::{warn, info, debug};

#[path = "conf_random.rs"]
mod conf_random;

mod board;
use board::*;

mod tiles;
use tiles::*;

mod gravity;
use gravity::*;

pub struct Game
{
    board: Board,
    matchers: Vec<Box<dyn Matcher>>,
    processors: Vec<Box<dyn MatchProcessor>>,
}

pub type MatchResult = Option<Match>;
trait Matcher
{
    fn find_match(&mut self, board: &Board, from: &BoardPosition) -> MatchResult;
    fn reset(&mut self);
}

struct OneDimensionalMatcher;
impl OneDimensionalMatcher
{
    fn find_matching_tiles_one_dim<I: Iterator<Item=BoardPosition>>(tile: &TileData, board: &Board, positions: I) -> Vec<BoardPosition>
    {
        let mut to_ret = vec![];
        for pos in positions
        {
            if TileData::tiles_match(tile, board.get_tile(&pos))
            {
                to_ret.push(pos);
            }
            else
            {
                break;
            }
        }
        to_ret
    }
}

#[derive(Default)]
struct HorizontalMatcher
{
    required_len_for_match: usize,
    already_matched: HashSet<BoardPosition>
}
impl HorizontalMatcher
{
    pub fn new(required_len_for_match: usize) -> Self
    {
        Self { required_len_for_match, ..Default::default() } 
    }
}
impl Matcher for HorizontalMatcher
{
    fn find_match(&mut self, board: &Board, from: &BoardPosition) -> MatchResult 
    {
        if self.already_matched.contains(from)
        {
            return None
        }
        let tile_to_match = board.get_tile(from);

        let mut matching = OneDimensionalMatcher::find_matching_tiles_one_dim(&tile_to_match, board, (0..from.0).map(|x| (x, from.1)).rev());
        matching.append(&mut OneDimensionalMatcher::find_matching_tiles_one_dim(&tile_to_match, board, (from.0..board.width()).map(|x| (x, from.1))));

        if matching.len() >= self.required_len_for_match
        {
            for m in &matching
            {
                self.already_matched.insert(*m);
            }

            Some( Match { origin: *from, positions: matching } )
        }
        else
        {
            None
        }
    }

    fn reset(&mut self)
    {
        self.already_matched = HashSet::new();
    }
}

#[derive(Default)]
struct VerticalMatcher
{
    required_len_for_match: usize,
    already_matched: HashSet<BoardPosition>
}
impl VerticalMatcher
{
    pub fn new(required_len_for_match: usize) -> Self
    {
        Self { required_len_for_match, ..Default::default() } 
    }
}
impl Matcher for VerticalMatcher
{
    fn find_match(&mut self, board: &Board, from: &BoardPosition) -> MatchResult 
    {
        if self.already_matched.contains(from)
        {
            return None
        }

        let tile_to_match = board.get_tile(from);

        let mut matching = OneDimensionalMatcher::find_matching_tiles_one_dim(&tile_to_match, board, (0..from.1).map(|y| (from.0, y)).rev());
        matching.append(&mut OneDimensionalMatcher::find_matching_tiles_one_dim(&tile_to_match, board, (from.1..board.height()).map(|y| (from.0, y))));

        if matching.len() >= self.required_len_for_match
        {
            for m in &matching
            {
                self.already_matched.insert(*m);
            }
            Some( Match { origin: *from, positions: matching } )
        }
        else
        {
            None
        }
    }

    fn reset(&mut self)
    {
        self.already_matched = HashSet::new();
    }
}

#[derive(Default)]
struct TMatcher
{
    required_len_for_match: usize,
    already_matched: HashSet<BoardPosition>
}
impl TMatcher
{
    pub fn new(required_len_for_match: usize) -> Self
    {
        Self { required_len_for_match, ..Default::default() }
    }
}
impl Matcher for TMatcher
{
    fn find_match(&mut self, board: &Board, from: &BoardPosition) -> MatchResult
    {
        if self.already_matched.contains(from)
        {
            return None
        }

        let vert_match = VerticalMatcher::new(self.required_len_for_match).find_match(board, from);
        let hor_match = HorizontalMatcher::new(self.required_len_for_match).find_match(board, from);

        if vert_match.is_some() && hor_match.is_some()
        {
            let mut all_poss = vert_match.unwrap().positions.clone();
            all_poss.append(&mut hor_match.unwrap().positions);
            for p in &all_poss
            {
                self.already_matched.insert(*p);
            }
            Some(Match { origin: *from, positions: all_poss})
        }
        else
        {
            None
        }
    }

    fn reset(&mut self)
    {
        self.already_matched = HashSet::new();
    }
}

#[derive(Debug)]
struct Match
{
    origin: BoardPosition,
    positions: Vec<BoardPosition>
}
struct MatchDetector;

impl MatchDetector
{
    fn detect_matches(board: &Board, pattern_matchers: &mut Vec<Box<dyn Matcher>>) -> Vec<Match>
    {
        let mut matches = vec![];

        for matcher in pattern_matchers.iter_mut()
        {
            matcher.reset();
        }

        for i in 0..board.width()
        {
            for j in 0..board.height()
            {
                for matcher in pattern_matchers.iter_mut()
                {
                    if let Some(mtch) = matcher.find_match(board, &(i, j))
                    {
                        matches.push(mtch);
                    }
                }
            }
        }

        debug!("Matches in \n{}\n\n{:?}", board, matches);
        matches
    }
}

#[derive(Debug)]
enum GameEvent
{
    Spawned(BoardPosition, TileData),
    Match(BoardPosition, usize),
    NeighborHit(BoardPosition),
    Destroyed(BoardPosition),
    EventContainer { contained: Vec<GameEvent> }
}

trait MatchProcessor
{
    fn process_matches(&self, board: &mut Board, matches: &Vec<Match>) -> GameEvent;
}

#[derive(Default)]
struct OnMatchProcessor {}
impl MatchProcessor for OnMatchProcessor
{
    fn process_matches(&self, board: &mut Board, matches: &Vec<Match>) -> GameEvent
    {
        let mut generated_events = vec![];
        let mut processed = HashSet::new();
        for mtch in matches
        {

            // Anything special about this match? 

            for pos in &mtch.positions
            {
                if processed.contains(pos)
                {
                    continue;
                }

                let _tdata = board.get_tile_mut(pos);
                // Depending on the tdata I guess you'd want to react differently when "matching"
                //
                // Would a hittable mask make sense?
                // if (hittable)
                generated_events.push(GameEvent::Match(*pos, mtch.positions.len()));

                // Generate events for hitting adjacent positions
                //
                // How do we parameterize adjacency?
                let mut neighbors_hit = HashSet::new();
                let v: Vec<(isize,isize)> = vec![(-1, 0), (0, 1), (1, 0), (0, -1)];
                for (oi, oj) in v
                {
                    if pos.0 == 0 && oi < 0 || pos.1 == 0 &&  oj < 0
                    {
                        continue;
                    }

                    if pos.0 == board.width() - 1 && oi > 0 || pos.1 == board.height() - 1  && oj > 0
                    {
                        continue;
                    }

                    // 
                    let hitpos: (usize, usize) = ((oi+pos.0 as isize).try_into().unwrap(),
                                                  (oj+pos.1 as isize).try_into().unwrap());

                    if neighbors_hit.contains(&hitpos)
                    {
                        continue;
                    }

                    generated_events.push(GameEvent::NeighborHit( (hitpos.0, hitpos.1) ));

                    neighbors_hit.insert(hitpos);
                }


                processed.insert(*pos);
            }

        }

        GameEvent::EventContainer { contained: generated_events }
    }
}

#[derive(Default)]
struct EventProcessor;

impl EventProcessor
{
    pub fn process_events(board: &mut Board, events: &mut Vec<GameEvent>) -> Vec<GameEvent>
    {
        let mut to_ret = vec![];
        for evt in events
        {

            debug!("Board Before Processing Event {:?}:\n{}", evt, board);
            match evt
            {
                GameEvent::EventContainer { contained } =>
                {
                    to_ret.append(&mut Self::process_events(board, contained));
                },
                GameEvent::Match(pos, _len) =>
                {
                    let tile = board.get_tile_mut(pos);
                    tile.health -= 1;
                    if tile.health <= 0
                    {
                        to_ret.push(GameEvent::Destroyed(*pos));
                    }

                    // if len > xx then ... ?
                },
                GameEvent::Destroyed(pos) =>
                {
                    board.get_tile_mut(pos).clear();

                    //println!("Destroyed {:?}", pos);

                    // What else?
                },
                GameEvent::NeighborHit(_) =>
                {
                    // Something like if position.is_neighbor_hittable - then, hit the tile or is
                    // it different?
                },
                _ => {}
            }
            debug!("Board After Processing Event {:?}:\n{}", evt, board);
        }
        to_ret
    }
}


type ProbabilityRange = (f32, f32);

#[derive(Debug)]
struct SpawnProbabilityTable
{
    table: Vec<(ProbabilityRange, TileData)>,
    default_tile: TileData,
}

impl SpawnProbabilityTable
{
    pub fn new(default_tile: TileData) -> Self
    {
        Self { table: Vec::new(), default_tile }
    }

    //TODO: Probably a more expresive interface for this 
    // Should we normalize this automatically?
    pub fn from_iter<T>(iter: T, default_tile:Option<TileData>) -> Self
        where T: IntoIterator<Item=(ProbabilityRange, TileData)>
    {
        let mut table =  Vec::new();
        for v in iter
        {
            table.push( (v.0, v.1) );
        }
        Self { table, default_tile: default_tile.unwrap_or(TileData::new(0, 1)) }
    }

    pub fn generate_tile_data(&self, roll: f32) -> TileData
    {
        let mut res = self.default_tile;
        for (range, td) in &self.table
        {
            if roll > range.0 && roll < range.1
            {
                res = *td;
            }
        }

        debug!("Generated New Tile From: {:?} - {}", roll, res);
        return res; 
    }
}

#[derive(Debug)]
struct BoardFiller<'a>
{
    rng_ctx: &'a mut conf_random::RandomCtx,
    spawn_table: &'a mut SpawnProbabilityTable,
}

impl <'a> BoardFiller<'a>
{
    pub fn new(spawn_table: &'a mut SpawnProbabilityTable, rng_ctx: &'a mut conf_random::RandomCtx) -> Self
    {
        Self { rng_ctx, spawn_table }
    }

    // TODO: Is this just a factory with a different name?
    pub fn fill_board(&mut self, board: &mut Board) -> GameEvent 
    {
        debug!("BOARD FILL: Before Filling \n{}", board);
        let mut spawned = vec![];

        // Always fill top to bottom, left to right
        for row in 0..board.height()
        {
            for col in 0..board.width()
            {
                if !board.get_tile(&(col ,row)).is_empty()
                {
                    continue;
                }

                let tdata: TileData = self.spawn_table.generate_tile_data( self.rng_ctx.gen_range(0., 1.) ); 
                board.set_tile(tdata, &(col, row));
                spawned.push(GameEvent::Spawned((col, row), tdata));
            }
        }

        debug!("BOARD FILL: After Filling \n{}", board);
        GameEvent::EventContainer { contained: spawned } 
    }
}



fn main()
{
}

#[cfg(test)]
mod test
{
    use super::*;
    use env_logger;

    #[test]
    fn test()
    {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut board = Board::new((4, 4));
        let q = MatchDetector::detect_matches(&board, &mut vec![Box::new(HorizontalMatcher::new(3))]);

        assert!(q.len() == 0);

        const RED:   u64 = 1 << 0;
        const GREEN: u64 = 1 << 1;
        const BLUE:  u64 = 1 << 2;

        /* Board:
         *   0 1 2 3 
         * 0 G G B B
         * 1 G G B R
         * 2 G R B R
         * 3 G R R R
         */
        // COL 1
        board.set_tile(TileData::new(GREEN, 1), &(0,0));
        board.set_tile(TileData::new(GREEN, 1), &(0,1));
        board.set_tile(TileData::new(GREEN, 1), &(0,2));
        board.set_tile(TileData::new(GREEN, 1), &(0,3));

        // COL 2
        board.set_tile(TileData::new(GREEN, 1), &(1,0));
        board.set_tile(TileData::new(GREEN, 1), &(1,1));
        board.set_tile(TileData::new(RED,   1), &(1,2));
        board.set_tile(TileData::new(RED,   1), &(1,3));

        // COL 3 
        board.set_tile(TileData::new(BLUE, 1), &(2,0));
        board.set_tile(TileData::new(BLUE, 1), &(2,1));
        board.set_tile(TileData::new(BLUE, 1), &(2,2));
        board.set_tile(TileData::new(RED,  1), &(2,3));

        // COL 4
        board.set_tile(TileData::new(BLUE, 1), &(3,0));
        board.set_tile(TileData::new(RED,  1), &(3,1));
        board.set_tile(TileData::new(RED,  1), &(3,2));
        board.set_tile(TileData::new(RED,  1), &(3,3));


        // Test the OneDimensionalMatcher starting on (0, 0) - Vertical
        let r = OneDimensionalMatcher::find_matching_tiles_one_dim(board.get_tile(&(0,0)), &board, (0..board.height()).map(|y| (0, y)));
        assert!(r.len() == 4, "Incorrect match len: {}", r.len()); 

        // Test the OneDimensionalMatcher starting on (0, 3) - Vertical Reversed
        let r = OneDimensionalMatcher::find_matching_tiles_one_dim(board.get_tile(&(0,0)), &board, (0..board.height()).map(|y| (0, y)).rev());
        assert!(r.len() == 4, "Incorrect match len: {}", r.len()); 

        // Test Vertical Matcher from (2,1) - Vertical in both directions
        let r = VerticalMatcher::new(3).find_match(&board, &(2,1));
        assert!(r.is_some(), "Match not found");
        let l = r.unwrap().positions.len() == 3;
        assert!(l, "Incorrect match len: {}", l); 

        // Test Horizontal Matcher from (2,3) - Horizontal in both directions
        let r = HorizontalMatcher::new(3).find_match(&board, &(2,3));
        assert!(r.is_some(), "Match not found");
        let l = r.unwrap().positions.len() == 3;
        assert!(l, "Incorrect match len: {}", l); 


        // A move would be triggered here

        // Then matches would be detected
        let mut all_matches = MatchDetector::detect_matches(&board, &mut vec![Box::new(HorizontalMatcher::new(3)), Box::new(VerticalMatcher::new(3))]);
        assert!(all_matches.len() == 4, "Wrong match length {}", all_matches.len());

        let mut v = MatchDetector::detect_matches(&board, &mut vec![Box::new(TMatcher::new(3))]);
        assert!(v.len() == 1, "Wrong match length {}", v.len());

        // Matches would be processed
        all_matches.append(&mut v);
        let omp = OnMatchProcessor{};
        let mut evts = if let GameEvent::EventContainer { contained: evctr_evts } = omp.process_matches(&mut board, &all_matches)
        {
            evctr_evts
        }
        else
        {
            vec![]
        };

        // The events generated by those matches would be processed
        while !evts.is_empty()
        {
            evts = EventProcessor::process_events(&mut board, &mut evts);
        }

        // Post-Match Phase - Apply Gravity, Spawn new pieces
        let mut prob_table = SpawnProbabilityTable::from_iter(vec![((0.0, 0.25), TileData::new(RED, 1)), ((0.25, 0.5), TileData::new(BLUE, 1))],
                                                              Some(TileData::new(GREEN, 1)));
        let mut rng_ctx = conf_random::RandomCtx::from_seed([101, 203, 1414, 3141], "Name".to_string()); 
        let mut filler = BoardFiller::new(&mut prob_table, &mut rng_ctx);

        let mut gravity = GravitySystem::new();
        gravity.apply_gravity(&mut board);

        filler.fill_board(&mut board);

        println!("Condensed...");
        // Condensed 
        let mut matchers: Vec<Box<dyn Matcher>> = vec![Box::new(TMatcher::new(3)), Box::new(HorizontalMatcher::new(3)), Box::new(VerticalMatcher::new(3))];
        loop
        {
            // Match Phase
            let all_matches = MatchDetector::detect_matches(&board, &mut matchers);
            let omp = OnMatchProcessor{};

            // Process Match Events Phase
            let mut evts = if let GameEvent::EventContainer { contained: evctr_evts } = omp.process_matches(&mut board, &all_matches)
            {
                evctr_evts
            }
            else
            {
                vec![]
            };

            // Post-Match Phase
            while !evts.is_empty()
            {
                evts = EventProcessor::process_events(&mut board, &mut evts);
            }

            // Gravity Phase
            gravity.apply_gravity(&mut board);

            // Spawn Phase 
            let evts = if let GameEvent::EventContainer { contained: evctr_evts } = filler.fill_board(& mut board)
            {
                evctr_evts
            }
            else
            {
                vec![]
            };

            if evts.is_empty() 
            {
                break;
            }
        }

        assert!(evts.len() == 0, "{:?}", evts);
    }
}
