use nom::IResult;
use nom::Parser;
use nom::bytes::complete::tag;
use nom::character::complete::digit1;
use nom::combinator::eof;

const INPUT_FILE: &str = "input/05.txt";

pub fn a() -> usize {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input)
}

fn a_with_input(input: &str) -> usize {
    let (ranges, vals) = parse(input);

    let mut fresh = 0;

    for v in vals {
        for r in &ranges {
            if r.contains(v) {
                fresh += 1;
                break;
            }
        }
    }

    fresh
}

pub fn b() -> u64 {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input)
}

fn b_with_input(input: &str) -> u64 {
    let (mut ranges, _) = parse(input);

    if ranges.is_empty() {
        return 0;
    }

    ranges.sort_by(|a, b| a.min.cmp(&b.min).then(a.max.cmp(&b.max)));

    let mut better_ranges = Vec::new();

    let mut running = None;

    for r in ranges {
        if running.is_none() {
            running = Some(r);
            continue;
        }

        let r_old = running.unwrap();

        // this is needed to prevent double counting, in case the ranges overlaps
        if r_old.overlaps(r) {
            running = Some(Range {
                min: r.min.min(r_old.min),
                max: r.max.max(r_old.max),
            });
        } else {
            better_ranges.push(r_old);
            running = Some(r);
        }
    }

    if let Some(r) = running {
        better_ranges.push(r);
    }

    better_ranges.into_iter().map(|r| r.max - r.min + 1).sum()
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
struct Range {
    min: u64,
    max: u64,
}

impl Range {
    fn contains(&self, val: u64) -> bool {
        self.min <= val && val <= self.max
    }

    fn overlaps(&self, other: Self) -> bool {
        self.min <= other.max && other.min <= self.max
    }
}

fn parse(input: &str) -> (Vec<Range>, Vec<u64>) {
    fn parse_range(input: &str) -> IResult<&str, Range> {
        let (input, min) = digit1.map(|d: &str| d.parse::<u64>().unwrap()).parse(input)?;
        let (input, _) = tag("-").parse(input)?;
        let (input, max) = digit1.map(|d: &str| d.parse::<u64>().unwrap()).parse(input)?;
        let (_, _) = eof(input)?;

        Ok(("", Range { min, max }))
    }

    let mut ranges = Vec::new();

    let mut lines = input.lines();

    for line in lines.by_ref() {
        if line.is_empty() {
            break;
        }

        ranges.push(parse_range(line).unwrap().1);
    }

    let mut nums = Vec::new();

    for line in lines {
        nums.push(line.parse().unwrap());
    }

    (ranges, nums)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_A: &str = "3-5
10-14
16-20
12-18

1
5
8
11
17
32";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE_A), 3);
    }

    const SAMPLE_B: &str = "3-5
10-14
16-20
12-18";

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE_B), 14);
    }

    #[test]
    fn overlaps_test() {
        let a = Range { min: 5, max: 10 };
        let b = Range { min: 11, max: 16 };

        // all of these touch a, either by being inside, overlapping, or just barely touching
        let c = Range { min: 6, max: 8 };
        let d = Range { min: 8, max: 12 };
        let e = Range { min: 10, max: 13 };
        let f = Range { min: 0, max: 5 };

        assert!(!a.overlaps(b));
        assert!(!b.overlaps(a));

        for r in [c, d, e, f] {
            assert!(a.overlaps(r));
            assert!(r.overlaps(a));
        }
    }
}
