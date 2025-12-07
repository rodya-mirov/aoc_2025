use ahash::HashMap;
use ahash::HashSet;
use itertools::Itertools;

const INPUT_FILE: &str = "input/07.txt";

pub fn a() -> usize {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input)
}

fn a_with_input(input: &str) -> usize {
    let parsed = parse(input);

    let mut splits = 0;
    let mut beams = vec![parsed.start];

    for row in &parsed.splitters {
        let mut next_beams = Vec::new();

        for beam in beams {
            if row.contains(&beam) {
                next_beams.push(beam - 1);
                next_beams.push(beam + 1);
                splits += 1;
            } else {
                next_beams.push(beam);
            }
        }

        next_beams.sort();
        next_beams = next_beams.into_iter().dedup().collect();

        beams = next_beams;
    }

    splits
}

pub fn b() -> usize {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input)
}

fn b_with_input(input: &str) -> usize {
    let parsed = parse(input);

    let mut beams = HashMap::default();
    beams.insert(parsed.start, 1);

    for row in &parsed.splitters {
        let mut next_beams = HashMap::default();

        for (beam, mult) in beams.into_iter() {
            if row.contains(&beam) {
                *next_beams.entry(beam - 1).or_default() += mult;
                *next_beams.entry(beam + 1).or_default() += mult;
            } else {
                *next_beams.entry(beam).or_default() += mult;
            }
        }

        beams = next_beams;
    }

    beams.values().sum()
}

struct Parsed {
    start: usize,
    // processed in order; splitters[0], then splitters[1], and so on
    splitters: Vec<HashSet<usize>>,
}

fn parse(input: &str) -> Parsed {
    let mut lines = input.lines();

    let first_line = lines.next().expect("Should have a first line");

    let start = first_line
        .chars()
        .enumerate()
        .find(|(_ind, c)| *c == 'S')
        .expect("Should have a start character")
        .0;

    let mut splitters = Vec::new();

    for line in lines {
        let indices = line
            .chars()
            .enumerate()
            .filter(|(_ind, c)| *c == '^')
            .map(|(ind, _c)| ind)
            .collect();
        splitters.push(indices);
    }

    Parsed { start, splitters }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = ".......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE), 21);
    }

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE), 40);
    }
}
