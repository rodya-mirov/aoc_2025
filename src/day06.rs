use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::digit1;
use nom::character::complete::multispace1;
use nom::combinator::eof;
use nom::multi::separated_list1;

const INPUT_FILE: &str = "input/06.txt";

pub fn a() -> u64 {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input)
}

fn a_with_input(input: &str) -> u64 {
    let problems = parse_a(input);

    let mut total = 0;

    for p in problems {
        total += work_problem(&p);
    }

    total
}

fn work_problem(p: &Problem) -> u64 {
    match p.op {
        Op::Plus => p.nums.iter().copied().sum(),
        Op::Mult => p.nums.iter().copied().product::<u64>(),
    }
}

pub fn b() -> u64 {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input)
}

fn b_with_input(input: &str) -> u64 {
    let problems = parse_b(input);

    let mut total = 0;

    for p in problems {
        total += work_problem(&p);
    }

    total
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
enum Op {
    Plus,
    Mult,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Problem {
    nums: Vec<u64>,
    op: Op,
}

fn parse_a(input: &str) -> Vec<Problem> {
    fn parse_nums(line: &str) -> IResult<&str, Vec<u64>> {
        let line = line.trim();
        let (input, nums) = separated_list1(multispace1, digit1.map(|digits: &str| digits.parse::<u64>().unwrap())).parse(line)?;
        let (_, _) = eof(input)?;
        Ok(("", nums))
    }

    fn parse_ops(line: &str) -> IResult<&str, Vec<Op>> {
        let line = line.trim();
        let (input, ops) = separated_list1(
            multispace1,
            alt((tag("+"), tag("*"))).map(|c: &str| match c {
                "+" => Op::Plus,
                "*" => Op::Mult,
                _ => unreachable!(),
            }),
        )
        .parse(line)?;
        let (_, _) = eof(input)?;
        Ok(("", ops))
    }

    let mut lines = input.lines().peekable();

    let (_, first_line_nums) = parse_nums(lines.next().unwrap()).unwrap();

    let mut problem_digits: Vec<Vec<u64>> = first_line_nums.into_iter().map(|num| vec![num]).collect();

    while lines.peek().is_some() {
        let next_line = lines.next().unwrap();

        if lines.peek().is_some() {
            let (_, next_nums) = parse_nums(next_line).unwrap();
            assert_eq!(problem_digits.len(), next_nums.len());
            problem_digits
                .iter_mut()
                .zip(next_nums)
                .for_each(|(problem, num)| problem.push(num));
        } else {
            let (_, ops) = parse_ops(next_line).unwrap();
            assert_eq!(problem_digits.len(), ops.len());

            return problem_digits.into_iter().zip(ops).map(|(nums, op)| Problem { nums, op }).collect();
        }
    }

    unreachable!("Bad input! Last line should be ops")
}

fn parse_b(input: &str) -> Vec<Problem> {
    let mut lines: Vec<Vec<char>> = input.lines().map(|line| line.chars().collect()).collect();

    let max_len = lines.iter().map(|line| line.len()).max().unwrap_or_default();

    for line in lines.iter_mut() {
        while line.len() < max_len {
            line.push(' ');
        }
    }

    let mut problems: Vec<Problem> = Vec::new();

    let mut nums: Vec<u64> = Vec::new();

    for x in (0..lines[0].len()).rev() {
        let mut num: u64 = 0;

        for y in 0..lines.len() - 1 {
            let digit = lines[y][x];
            if let Some(digit) = digit.to_digit(10) {
                num = (num * 10) + (digit as u64);
            }
        }

        // cheap way to detect the empty column; we just split on operations, since they
        // seem to be left-aligned, at which point the empty columns have no further semantic
        // meaning and can be ignored
        if num > 0 {
            nums.push(num);
        }

        if let Some(op) = match lines[lines.len() - 1][x] {
            '+' => Some(Op::Plus),
            '*' => Some(Op::Mult),
            ' ' => None,
            other => panic!("Bad input: {other}"),
        } {
            problems.push(Problem { op, nums });
            nums = Vec::new();
        }
    }

    if !nums.is_empty() {
        panic!("Numbers without final operation");
    }

    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +  ";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE), 4277556);
    }

    #[test]
    fn problem_samples_a() {
        assert_eq!(
            work_problem(&Problem {
                nums: vec![123, 45, 6],
                op: Op::Mult
            }),
            33210
        );
        assert_eq!(
            work_problem(&Problem {
                nums: vec![328, 64, 98],
                op: Op::Plus
            }),
            490
        );
    }

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE), 3263827);
    }

    #[test]
    fn parse_b_test() {
        let actual = parse_b(SAMPLE);

        let expected = vec![
            Problem {
                nums: vec![4, 431, 623],
                op: Op::Plus,
            },
            Problem {
                nums: vec![175, 581, 32],
                op: Op::Mult,
            },
            Problem {
                nums: vec![8, 248, 369],
                op: Op::Plus,
            },
            Problem {
                nums: vec![356, 24, 1],
                op: Op::Mult,
            },
        ];

        assert_eq!(expected, actual);
    }
}
