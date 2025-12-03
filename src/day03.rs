const INPUT_FILE: &str = "input/03.txt";

pub fn a() -> u64 {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input)
}

fn a_with_input(input: &str) -> u64 {
    parse_input(input)
        .into_iter()
        .map(|line| max_joltage_variable(&line, 2))
        .sum()
}

pub fn b() -> u64 {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input)
}

fn b_with_input(input: &str) -> u64 {
    parse_input(input)
        .into_iter()
        .map(|line| max_joltage_variable(&line, 12))
        .sum()
}

/// Super version of max_joltage where you can dynamically describe the number of batteries
fn max_joltage_variable(bank: &[u8], num_batteries: usize) -> u64 {
    // The point here is that the right answer is always to pick the highest value available,
    // and break ties by moving as far left as possible (because 900000 is bigger than 899999;
    // there's no way for the greedy algorithm to fail)

    let mut total_sum: u64 = 0;
    let mut left_ind = 0;

    for battery in 0 .. num_batteries {
        let mut best_val = 0;
        let mut best_ind = left_ind;

        for ind in left_ind .. bank.len() - (num_batteries - battery - 1) {
            let val = bank[ind];
            if val > best_val {
                best_ind = ind;
                best_val = val;
            }
        }

        left_ind = best_ind + 1;
        total_sum = (total_sum * 10) + (best_val as u64);
    }

    total_sum
}

fn parse_line(line: &str) -> Vec<u8> {
    line.chars().map(|c| match c {
        '0' => 0,
        '1' => 1,
        '2' => 2,
        '3' => 3,
        '4' => 4,
        '5' => 5,
        '6' => 6,
        '7' => 7,
        '8' => 8,
        '9' => 9,
        other => unimplemented!("Unrecognized joltage '{other}'")
    })
        .collect()
}

fn parse_input(input: &str) -> Vec<Vec<u8>> {
    input.lines().map(parse_line).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_INPUT: &str = "987654321111111
811111111111119
234234234234278
818181911112111";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE_INPUT), 357);
    }

    #[test]
    fn sample_a_banks() {
        assert_eq!(max_joltage_variable(&parse_line("987654321111111"), 2), 98);
        assert_eq!(max_joltage_variable(&parse_line("811111111111119"), 2), 89);
        assert_eq!(max_joltage_variable(&parse_line("234234234234278"), 2), 78);
        assert_eq!(max_joltage_variable(&parse_line("818181911112111"), 2), 92);
    }

    #[test]
    fn sample_b_banks() {
        assert_eq!(max_joltage_variable(&parse_line("987654321111111"), 12), 987654321111);
        assert_eq!(max_joltage_variable(&parse_line("811111111111119"), 12), 811111111119);
        assert_eq!(max_joltage_variable(&parse_line("234234234234278"), 12), 434234234278);
        assert_eq!(max_joltage_variable(&parse_line("818181911112111"), 12), 888911112111);
    }

    #[test]
    fn test_parser() {
        const INPUT: &str = "123
456";

        assert_eq!(parse_input(INPUT), vec![
            vec![1, 2, 3],
            vec![4, 5, 6]
        ]);
    }
}
