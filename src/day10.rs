use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::digit1;
use nom::combinator::eof;
use nom::multi::many1;
use nom::multi::separated_list1;

const INPUT_FILE: &str = "input/10.txt";

// can handle up to 32 lights, which seems like plenty
type TriggerMask = u32;

pub fn a() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input).to_string()
}

fn a_with_input(input: &str) -> usize {
    parse(input).iter().map(best_solve_machine_a).sum()
}

pub fn b() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input).to_string()
}

fn b_with_input(_input: &str) -> usize {
    unimplemented!()
}

#[derive(Clone, Eq, PartialEq, Debug)]
struct Machine {
    desired_state_mask: TriggerMask,
    buttons: Vec<TriggerMask>,
    joltage_requirements: Vec<u32>,
}

fn parse(input: &str) -> Vec<Machine> {
    input.lines().map(parse_line).collect()
}

fn parse_line(line: &str) -> Machine {
    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();

    assert!(tokens.len() > 2);

    let desired_state_mask = parse_desired_state(tokens[0]);

    let mut buttons = Vec::with_capacity(tokens.len() - 2);

    for i in 1..tokens.len() - 1 {
        buttons.push(parse_button(tokens[i]));
    }

    buttons.sort_by_key(|b| std::cmp::Reverse(b.count_ones()));

    let joltage_requirements = parse_joltage(tokens[tokens.len() - 1]);

    Machine {
        desired_state_mask,
        buttons,
        joltage_requirements,
    }
}

fn parse_desired_state(token: &str) -> TriggerMask {
    fn parse_fallible(input: &str) -> IResult<&str, TriggerMask> {
        let (input, _) = tag("[")(input)?;
        let (input, on) = many1(alt((tag(".").map(|_| false), tag("#").map(|_| true)))).parse(input)?;
        let (input, _) = tag("]")(input)?;
        let (_, _) = eof(input)?;

        let mask = on
            .into_iter()
            .enumerate()
            .filter(|(_ind, is_on)| *is_on)
            .map(|(ind, _)| 1 << ind)
            .sum();

        Ok(("", mask))
    }

    parse_fallible(token).unwrap().1
}

fn parse_button(input: &str) -> TriggerMask {
    fn parse_fallible(input: &str) -> IResult<&str, TriggerMask> {
        let (input, _) = tag("(")(input)?;
        let (input, indices) = separated_list1(tag(","), digit1).parse(input)?;
        let (input, _) = tag(")")(input)?;
        let (_, _) = eof(input)?;

        Ok(("", indices.into_iter().map(|s| s.parse::<u32>().unwrap()).map(|n| 1 << n).sum()))
    }

    parse_fallible(input).unwrap().1
}

fn parse_joltage(input: &str) -> Vec<u32> {
    fn parse_fallible(input: &str) -> IResult<&str, Vec<u32>> {
        let (input, _) = tag("{")(input)?;
        let (input, nums) = separated_list1(tag(","), digit1).parse(input)?;
        let (input, _) = tag("}")(input)?;
        let (_, _) = eof(input)?;

        let nums = nums.iter().map(|s| s.parse::<u32>().unwrap()).collect();

        Ok(("", nums))
    }

    parse_fallible(input).unwrap().1
}

fn best_solve_machine_a(machine: &Machine) -> usize {
    /// returns true if we can reach the desired mask from the current mask
    /// with the given fuel and transitions
    // Note perf:
    //      (a) there's no reason to ever press a button twice, and
    //      (b) the buttons commute, so we don't need to test AB and BA
    // so we can track min_button carefully to take care of both
    fn bounded_dfs(fuel: usize, current_mask: u32, machine: &Machine, min_button: usize) -> bool {
        if fuel == 0 {
            return current_mask == machine.desired_state_mask;
        }

        for ind in min_button..machine.buttons.len() {
            let button_mask = machine.buttons[ind];
            let next_mask = button_mask ^ current_mask;

            if bounded_dfs(fuel - 1, next_mask, machine, ind + 1) {
                return true;
            }
        }

        false
    }

    let mut i = 0;

    loop {
        if bounded_dfs(i, 0, machine, 0) {
            return i;
        }

        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE), 7);
    }

    #[test]
    fn samples_a() {
        fn test_line(line: &str, expected: usize) {
            let parsed = parse_line(line);
            let actual = best_solve_machine_a(&parsed);
            assert_eq!(actual, expected);
        }

        test_line("[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}", 2);
        test_line("[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}", 3);
        test_line("[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}", 2);
    }

    #[test]
    fn test_parse_a1() {
        let input = "[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}";

        let actual = parse_line(input);

        let expected = Machine {
            desired_state_mask: 2 + 4,
            buttons: vec![
                // buttons reordered to put biggest ones first
                (1 << 1) + (1 << 3),
                (1 << 2) + (1 << 3),
                (1 << 0) + (1 << 2),
                (1 << 0) + (1 << 1),
                1 << 3,
                1 << 2,
            ],
            joltage_requirements: vec![3, 5, 4, 7],
        };

        assert_eq!(actual, expected);
    }
}
