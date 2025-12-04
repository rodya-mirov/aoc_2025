const INPUT_FILE: &str = "input/04.txt";

pub fn a() -> usize {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    a_with_input(&input)
}

fn a_with_input(input: &str) -> usize {
    let input = parse(input);

    let mut reachable = 0;

    for y in 0..input.len() {
        for x in 0..input[y].len() {
            let mut neighbors = 0;

            if !input[y][x] {
                continue;
            }

            // Assuming it's a rectangle
            let has_up = y > 0;
            let has_left = x > 0;
            let has_down = y + 1 < input.len();
            let has_right = x + 1 < input[y].len();

            if has_up && has_left && input[y - 1][x - 1] {
                neighbors += 1;
            }
            if has_up && input[y - 1][x] {
                neighbors += 1;
            }
            if has_up && has_right && input[y - 1][x + 1] {
                neighbors += 1;
            }
            if has_left && input[y][x - 1] {
                neighbors += 1;
            }
            if has_right && input[y][x + 1] {
                neighbors += 1;
            }
            if has_left && has_down && input[y + 1][x - 1] {
                neighbors += 1;
            }
            if has_down && input[y + 1][x] {
                neighbors += 1;
            }
            if has_down && has_right && input[y + 1][x + 1] {
                neighbors += 1;
            }

            if neighbors < 4 {
                reachable += 1;
            }
        }
    }

    reachable
}

pub fn b() -> usize {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input)
}

fn b_with_input(input: &str) -> usize {
    let mut input = parse(input);

    let mut reachable = 0;

    loop {
        let old_reachable = reachable;

        for y in 0..input.len() {
            for x in 0..input[y].len() {
                let mut neighbors = 0;

                if !input[y][x] {
                    continue;
                }

                // Assuming it's a rectangle
                let has_up = y > 0;
                let has_left = x > 0;
                let has_down = y + 1 < input.len();
                let has_right = x + 1 < input[y].len();

                if has_up && has_left && input[y - 1][x - 1] {
                    neighbors += 1;
                }
                if has_up && input[y - 1][x] {
                    neighbors += 1;
                }
                if has_up && has_right && input[y - 1][x + 1] {
                    neighbors += 1;
                }
                if has_left && input[y][x - 1] {
                    neighbors += 1;
                }
                if has_right && input[y][x + 1] {
                    neighbors += 1;
                }
                if has_left && has_down && input[y + 1][x - 1] {
                    neighbors += 1;
                }
                if has_down && input[y + 1][x] {
                    neighbors += 1;
                }
                if has_down && has_right && input[y + 1][x + 1] {
                    neighbors += 1;
                }

                if neighbors < 4 {
                    reachable += 1;
                    // Note -- we don't care about the "rounds" of removal, so it's fine to immediately
                    // remove this roll of paper and then take advantage of the gap for the next roll
                    input[y][x] = false;
                }
            }
        }

        // as soon as we did a whole iteration with no changes, stop
        if old_reachable == reachable {
            break;
        }
    }

    reachable
}

fn parse(input: &str) -> Vec<Vec<bool>> {
    input
        .lines()
        .map(|line| {
            line.chars()
                .map(|c| match c {
                    '@' => true,
                    '.' => false,
                    other => panic!("Bad character {other}"),
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.";

    #[test]
    fn sample_a() {
        assert_eq!(a_with_input(SAMPLE), 13);
    }

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE), 43);
    }
}
