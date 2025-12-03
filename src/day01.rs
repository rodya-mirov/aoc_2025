use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::digit1;
use nom::combinator::eof;

const INPUT_FILE: &str = "input/01.txt";

pub fn a() -> usize {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input)
}

fn a_with_input(input: &str) -> usize {
    let mut dial: i32 = 50;
    let mut count_zeroes = 0;

    for line in input.lines() {
        let parsed = parse_line(line);

        dial = (dial + parsed).rem_euclid(100);

        if dial == 0 {
            count_zeroes += 1;
        }
    }

    count_zeroes
}

pub fn b() -> usize {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input)
}

fn b_with_input(input: &str) -> usize {
    let mut dial: i32 = 50;
    let mut count_zeroes = 0;

    for line in input.lines() {
        let parsed = parse_line(line);

        let (new_dial, new_zeroes) = b_step(dial, parsed);
        dial = new_dial;
        count_zeroes += new_zeroes;
    }

    count_zeroes
}

/// Given a dial state and a parsed spin, return the new dial state and the number of times
/// the dial crossed zero
fn b_step(mut dial: i32, parsed: i32) -> (i32, usize) {
    if parsed == 0 {
        return (dial, 0);
    }

    if parsed < 0 && dial == 0 {
        dial = 100;
    }
    if parsed > 0 && dial == 100 {
        dial = 0;
    }

    dial += parsed;

    let mut count_zeroes = 0;

    if dial == 0 || dial == 100 {
        return (0, 1);
    }

    while dial < 0 {
        count_zeroes += 1;
        dial += 100;

        if dial == 0 {
            count_zeroes += 1;
        }
    }

    while dial >= 100 {
        count_zeroes += 1;
        dial -= 100;
    }

    (dial, count_zeroes)
}

fn parse_line(line: &str) -> i32 {
    fn parse_fallible(input: &str) -> IResult<&str, i32> {
        let (input, sign) = alt((tag("L").map(|_| -1_i32), tag("R").map(|_| 1))).parse(input)?;

        let (input, digits) = digit1(input)?;
        let parsed: i32 = digits.parse().expect("Digits should parse");

        let (input, _) = eof(input)?;

        Ok((input, sign * parsed))
    }

    let (_, val) = parse_fallible(line).expect("Line should parse");

    val
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_A: &str = "L68
L30
R48
L5
R60
L55
L1
L99
R14
L82";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE_A), 3);
    }

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE_A), 6);
    }

    #[test]
    fn b_step_tests() {
        assert_eq!(b_step(50, 1000), (50, 10));
        assert_eq!(b_step(50, 1001), (51, 10));
        assert_eq!(b_step(50, 999), (49, 10));
        assert_eq!(b_step(50, -1000), (50, 10));
    }

    #[test]
    fn b_step_tests_edge_stops() {
        assert_eq!(b_step(50, 50), (0, 1));
        assert_eq!(b_step(50, 150), (0, 2));
        assert_eq!(b_step(50, 350), (0, 4));

        assert_eq!(b_step(50, -50), (0, 1));
        assert_eq!(b_step(50, -150), (0, 2));
        assert_eq!(b_step(50, -450), (0, 5));
    }
}
