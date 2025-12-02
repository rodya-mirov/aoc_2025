use ahash::HashSet;
use nom::Parser;
use nom::bytes::complete::tag;
use nom::character::complete::digit1;
use nom::combinator::eof;
use nom::IResult;
use nom::multi::separated_list1;

const INPUT_FILE: &str = "input/02.txt";

pub fn a() -> u64 {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input)
}

fn a_with_input(input: &str) -> u64 {
    let ranges = parse_ranges(input);

    // Note: this assumes the ranges are disjoint, which, whatever
    let mut all_bad_ids = HashSet::default();

    for r in ranges {
        all_silly_ids_reps(&r, 2, &mut all_bad_ids);
    }

    all_bad_ids.iter().copied().sum()
}

pub fn b() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input).to_string()
}

fn b_with_input(input: &str) -> u64 {
    let ranges = parse_ranges(input);

    // Note: this assumes the ranges are disjoint, which, whatever
    let mut all_bad_ids = HashSet::default();

    for r in ranges {
        all_silly_ids(&r, &mut all_bad_ids);
    }

    all_bad_ids.iter().copied().sum()
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Range {
    min: u64,
    max: u64,
}

struct SillyCounter {
    repetitions: usize,
    tenpow: u64,
    simple_num: u64,
}

impl SillyCounter {
    fn new(repetitions: usize) -> Self {
        Self {
            repetitions,
            tenpow: 10,
            simple_num: 1,
        }
    }

    fn as_num(&self) -> u64 {
        let mut out = self.simple_num;

        for _ in 1 .. self.repetitions {
            out = out * self.tenpow + self.simple_num;
        }

        out
    }

    fn next(&self) -> Self {
        if self.tenpow == self.simple_num + 1 {
            Self {
                repetitions: self.repetitions,
                tenpow: self.tenpow * 10,
                simple_num: self.simple_num + 1
            }
        } else {
            Self {
                repetitions: self.repetitions,
                tenpow: self.tenpow,
                simple_num: self.simple_num + 1
            }
        }
    }
}

fn all_silly_ids_reps(range: &Range, repetitions: usize, found: &mut HashSet<u64>) {
    let mut counter = SillyCounter::new(repetitions);

    while counter.as_num() < range.min {
        counter = counter.next();
    }

    while counter.as_num() <= range.max {
        found.insert(counter.as_num());
        counter = counter.next();
    }
}

fn all_silly_ids(range: &Range, found: &mut HashSet<u64>) {
    let mut repetitions = 2;

    loop {
        if SillyCounter::new(repetitions).as_num() > range.max {
            break;
        }

        all_silly_ids_reps(range, repetitions, found);

        repetitions += 1;
    }
}

fn parse_ranges(input: &str) -> Vec<Range> {
    fn parse_range_fallible(input: &str) -> IResult<&str, Range> {
        let (rem, digit_list) = separated_list1(tag("-"), digit1.map(|digits: &str| digits.parse::<u64>().unwrap())).parse(input)?;

        let r = match digit_list.as_slice() {
            [a] => Range { min: *a, max: *a },
            [a, b] => Range { min: *a, max: *b },
            _ => panic!("Got wrong number of - separated numbers (expected 1 or 2, got {})", digit_list.len())
        };

        Ok((rem, r))
    }

    fn parse_ranges_fallible(input: &str) -> IResult<&str, Vec<Range>> {
        let (rem, out) = separated_list1(tag(","), parse_range_fallible).parse(input)?;
        let (_, _) = eof(rem)?;
        Ok(("", out))
    }

    parse_ranges_fallible(input).expect("Things should parse").1
}


#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_INPUT: &str = "11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124";

    #[test]
    fn parse_test() {
        let input = "11-22,67,95-115,998-1012,1188511880-1188511890";
        let actual = parse_ranges(input);
        let expected = vec![
            Range { min: 11, max: 22 },
            Range { min: 67, max: 67 },
            Range { min: 95, max: 115 },
            Range { min: 998, max: 1012},
            Range { min: 1188511880, max: 1188511890 }
        ];

        assert_eq!(actual, expected);
    }

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE_INPUT), 1227775554);
    }

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE_INPUT), 4174379265);
    }
}
