use ahash::HashMap;

const INPUT_FILE: &str = "input/09.txt";

pub fn a() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input).to_string()
}

fn a_with_input(input: &str) -> usize {
    let coords = parse(input);

    let mut max = 0;

    for i in 0..coords.len() {
        let a = coords[i];
        for j in i + 1..coords.len() {
            let b = coords[j];

            let x = b.x.abs_diff(a.x) + 1;
            let y = b.y.abs_diff(a.y) + 1;

            max = max.max(x * y);
        }
    }

    max
}

pub fn b() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input).to_string()
}

fn b_with_input(input: &str) -> usize {
    let coords = parse(input);
    let Interning { interned_coords } = intern(&coords);

    let grid = flood_fill_grid(&interned_coords);

    let mut best_area = 0;

    for i in 0..coords.len() {
        let a = coords[i];
        for j in i + 1..coords.len() {
            let b = coords[j];

            let x = b.x.abs_diff(a.x) + 1;
            let y = b.y.abs_diff(a.y) + 1;

            let area = x * y;

            if area > best_area && is_valid(interned_coords[i], interned_coords[j], &grid) {
                best_area = area;
            }
        }
    }

    best_area
}

fn is_valid(a: Coord, b: Coord, grid: &[Vec<bool>]) -> bool {
    let x_min = a.x.min(b.x);
    let x_max = a.x.max(b.x);

    let y_min = a.y.min(b.y);
    let y_max = a.y.max(b.y);

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            if !grid[y][x] {
                return false;
            }
        }
    }

    true
}

/// Creates a grid from coordinates, marked whether the tile is flippable (true) or not (false)
/// Indexed by grid[y][x]
fn flood_fill_grid(coords: &[Coord]) -> Vec<Vec<bool>> {
    let mut x_max = 0;
    let mut y_max = 0;

    for c in coords {
        x_max = x_max.max(c.x);
        y_max = y_max.max(c.y);
    }

    // now we can make a grid and populate it; start with the corners and edges

    // None means unknown; Some(true) means known to be a flippable tile; Some(false) means known
    // to be a non-flippable tile
    let mut grid: Vec<Vec<Option<bool>>> = vec![vec![None; x_max + 2]; y_max + 2];

    for i in 0..coords.len() {
        let c = coords[i];
        let c_next = coords[(i + 1) % coords.len()];
        grid[c.y][c.x] = Some(true);
        grid[c_next.y][c_next.x] = Some(true);

        if c.x == c_next.x {
            let x = c.x;

            let y_min = c.y.min(c_next.y);
            let y_max = c.y.max(c_next.y);

            for y in y_min..=y_max {
                grid[y][x] = Some(true);
            }
        } else if c.y == c_next.y {
            let y = c.y;

            let x_min = c.x.min(c_next.x);
            let x_max = c.x.max(c_next.x);

            for x in x_min..=x_max {
                grid[y][x] = Some(true);
            }
        } else {
            panic!("Bad input: adjacent tiles aren't on a line")
        }
    }

    let mut flood_fill_queue = Vec::new();
    flood_fill_queue.push(Coord { x: 0, y: 0 });

    while let Some(Coord { x, y }) = flood_fill_queue.pop() {
        if grid[y][x].is_some() {
            continue;
        }

        grid[y][x] = Some(false);
        if y > 0 {
            flood_fill_queue.push(Coord { x, y: y - 1 });
        }
        if y < grid.len() - 1 {
            flood_fill_queue.push(Coord { x, y: y + 1 });
        }
        if x > 0 {
            flood_fill_queue.push(Coord { x: x - 1, y });
        }
        if x < grid[0].len() - 1 {
            flood_fill_queue.push(Coord { x: x + 1, y });
        }
    }

    grid.into_iter()
        .map(|row| row.into_iter().map(|cell| cell.unwrap_or(true)).collect())
        .collect()
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct Coord {
    x: usize,
    y: usize,
}

fn parse(input: &str) -> Vec<Coord> {
    input
        .lines()
        .map(|line| {
            let nums: Vec<usize> = line.split(",").map(|token| token.parse::<usize>().unwrap()).collect();
            assert_eq!(nums.len(), 2);

            Coord { x: nums[0], y: nums[1] }
        })
        .collect()
}

struct Interning {
    interned_coords: Vec<Coord>,
}

fn intern(coords: &[Coord]) -> Interning {
    let x_lookup = intern_values(coords.iter().map(|c| c.x).collect());
    let y_lookup = intern_values(coords.iter().map(|c| c.y).collect());

    let interned_coords = coords
        .iter()
        .map(|c| Coord {
            x: x_lookup.translate(c.x).unwrap(),
            y: y_lookup.translate(c.y).unwrap(),
        })
        .collect();

    Interning { interned_coords }
}

#[derive(Clone, Debug, Default)]
struct Lookup {
    forward: HashMap<usize, usize>,
}

impl Lookup {
    fn translate(&self, val: usize) -> Option<usize> {
        self.forward.get(&val).copied()
    }
}

fn intern_values(mut values: Vec<usize>) -> Lookup {
    values.sort();

    values.dedup();

    let mut out = Lookup::default();

    let mut last_true_value = 0;
    let mut last_interred_value = 0;

    for val in values.into_iter() {
        // we do this funny thing to ensure that we preserve the property of "there is a gap
        // between these numbers" but not to let that gap get too big, so the grid stays
        // manageable
        let translated = if val >= last_true_value + 2 {
            last_interred_value + 2
        } else {
            last_interred_value + 1
        };

        out.forward.insert(val, translated);

        last_true_value = val;
        last_interred_value = translated;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE), 50);
    }

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE), 24);
    }
}
