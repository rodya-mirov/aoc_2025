use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::digit1;
use nom::combinator::eof;
use nom::multi::many1;
use nom::multi::separated_list1;

use crate::fraction::Fraction32;

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

pub fn b() -> String {
    let input = std::fs::read_to_string(INPUT_FILE).expect("Input should exist");
    b_with_input(&input).to_string()
}

fn b_with_input(input: &str) -> usize {
    // Step 1 -- transform the input into matrices
    let machines = parse(input);
    let matrices: Vec<Requirements> = machines.into_iter().map(to_matrix).collect();

    matrices
        .into_iter()
        .map(|mut requirements| {
            // Note this is a fraction for type reasons, but we know it's a positive integer
            let highest_light_value = requirements.requirements.iter().copied().max().unwrap();

            // Step 2 -- row reduce so that we have pivots
            row_reduce(&mut requirements);
            trim(&mut requirements);

            // Step 3 -- turn this into inequalities with variables removed
            let substitutions = make_substitutions(&requirements);
            let objective_function = make_objective_function(&substitutions);

            // Step 4 -- solve the inequalities with a region search
            let best_value: MatrixElt = grid_search(&substitutions, &objective_function, highest_light_value);

            assert_eq!(best_value.bot, 1);
            assert!(best_value.top > 0);

            best_value.top as usize
        })
        .sum()
}

fn grid_search(substitutions: &[Substitution], objective: &ObjectiveFunction, highest_light: MatrixElt) -> MatrixElt {
    // So now we want to minimize OBJECTIVE_FUNCTION relative to the constraints:
    // 1. each free variable is a nonnegative integer
    // 2. each substitution is nonnegative (hence: each inequality is satisfied)
    // 3. each substitution is an integer

    // Each free variable is still a button in the original interpretation, and makes at
    // least one light go up one time. So they're bounded below by zero, and above by
    // whatever the highest (original) light value was; and then we can just do a grid
    // search.

    // Idea: for each plausible value in the grid, check if it satisfies all the requirements.
    // If so, grab its objective function value. Then take the minimum of all those.
    let num_free_variables = substitutions[0].coefficients.len();

    if num_free_variables == 0 {
        return objective.evaluate(&[]);
    }

    let mut running_best = Fraction32 { top: i32::MAX, bot: 1 };

    'fv: for free_values in GridIter::new(num_free_variables, highest_light) {
        // check if every substitution is a nonnegative integer
        for sub in substitutions {
            let sub_value = sub.evaluate(&free_values);
            if !sub_value.is_integer() {
                continue 'fv;
            }
            if !sub_value.is_nonnegative() {
                continue 'fv;
            }
        }

        // if we're here, the values are fine
        let objective_value = objective.evaluate(&free_values);
        running_best = running_best.min(objective_value);
    }

    running_best
}

struct GridIter {
    cur: Vec<Fraction32>,
    max_value: Fraction32,
    done: bool,
}

impl GridIter {
    fn new(dimensions: usize, max_value: Fraction32) -> Self {
        assert!(dimensions > 0);
        GridIter {
            cur: vec![Fraction32::ZERO; dimensions],
            max_value,
            done: false,
        }
    }
}

impl Iterator for GridIter {
    type Item = Vec<Fraction32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let out = self.cur.clone();

        let mut idx = self.cur.len() - 1;

        loop {
            self.cur[idx] += Fraction32::ONE;
            if self.cur[idx] > self.max_value {
                self.cur[idx] = Fraction32::ZERO;
                if idx == 0 {
                    self.done = true;
                    break;
                } else {
                    idx -= 1;
                    continue;
                }
            } else {
                break;
            }
        }

        Some(out)
    }
}

/// Represents an equation of the form x + a0z0 + ... +akzk = r
#[derive(Clone, Eq, PartialEq, Debug)]
struct Substitution {
    /// the a0, a1, ..., ak coefficients of the standard form
    coefficients: Vec<MatrixElt>,
    /// The r of the standard form
    result: MatrixElt,
}

impl Substitution {
    fn evaluate(&self, free_variables: &[Fraction32]) -> Fraction32 {
        let mut lhs = Fraction32::ZERO;
        assert_eq!(self.coefficients.len(), free_variables.len());

        for i in 0..self.coefficients.len() {
            lhs += self.coefficients[i] * free_variables[i];
        }

        self.result - lhs
    }
}

/// Represents the objective function, in the form a0z0 + ... + akzk + r
#[derive(Clone, Eq, PartialEq, Debug)]
struct ObjectiveFunction {
    coefficients: Vec<MatrixElt>,
    constant: MatrixElt,
}

impl ObjectiveFunction {
    fn evaluate(&self, free_variables: &[Fraction32]) -> Fraction32 {
        let mut out = Fraction32::ZERO;

        assert_eq!(self.coefficients.len(), free_variables.len());

        for i in 0..self.coefficients.len() {
            out += self.coefficients[i] * free_variables[i];
        }

        out + self.constant
    }
}

fn trim(req: &mut Requirements) {
    if req.button_matrix.is_empty() {
        return;
    }

    let i = req.button_matrix.len() - 1;
    if req.button_matrix[i].iter().all(|&v| v == Fraction32::ZERO) {
        if req.requirements[i] == Fraction32::ZERO {
            req.requirements.pop();
            req.button_matrix.pop();
            trim(req);
        } else {
            panic!("Unsatisfiable constraints -- zero row at bottom, with nonzero requirement");
        }
    }
}

/// Turn a row-reduced, trimmed matrix into a set of variable substitutions
/// These are in order correspondence with the rows in the matrix.
fn make_substitutions(req: &Requirements) -> Vec<Substitution> {
    fn make_substitution(row: &[MatrixElt], result: MatrixElt, num_free_variables: usize) -> Substitution {
        let start_index = row.len() - num_free_variables;
        let coefficients = (start_index..row.len()).map(|i| row[i]).collect();
        Substitution { coefficients, result }
    }

    let mut variables = Vec::with_capacity(req.button_matrix.len());

    let num_free_variables = req.button_matrix[0].len() - req.button_matrix.len();

    for i in 0..req.button_matrix.len() {
        variables.push(make_substitution(&req.button_matrix[i], req.requirements[i], num_free_variables));
    }

    variables
}

/// Given a set of substitutions, and the assumption that the original objective function was
/// "minimize A+B+...", make a new objective function in terms of the remaining free variables
fn make_objective_function(substitutions: &[Substitution]) -> ObjectiveFunction {
    let num_free_variables = substitutions[0].coefficients.len();

    let mut constant = Fraction32::ZERO;
    let mut coefficients = vec![Fraction32::ONE; num_free_variables];

    for variable in substitutions {
        // we know the variable is represented once in the objective function
        // we also know VAR + sum_i(ai zi) = r; so VAR = r - sum_i(ai zi)
        constant += variable.result;
        for i in 0..num_free_variables {
            coefficients[i] -= variable.coefficients[i];
        }
    }

    ObjectiveFunction { coefficients, constant }
}

type MatrixElt = Fraction32;

#[derive(Clone, Eq, PartialEq, Debug)]
struct Requirements {
    // this, times the variable matrix, must equal the requirements
    button_matrix: Vec<Vec<MatrixElt>>,
    requirements: Vec<MatrixElt>,
}

fn to_matrix(machine: Machine) -> Requirements {
    let num_lights = machine.joltage_requirements.len();

    let button_matrix = (0..num_lights)
        .map(|light_idx| {
            machine
                .buttons
                .iter()
                .map(|mask| if mask & (1 << light_idx) > 0 { 1.into() } else { 0.into() })
                .collect()
        })
        .collect();

    let requirements = machine
        .joltage_requirements
        .into_iter()
        .map(|num| {
            let num: i32 = num.try_into().unwrap();
            num.into()
        })
        .collect();

    Requirements {
        button_matrix,
        requirements,
    }
}

impl Requirements {
    fn swap_rows(&mut self, a: usize, b: usize) {
        self.requirements.swap(a, b);
        self.button_matrix.swap(a, b);
    }

    fn swap_cols(&mut self, a: usize, b: usize) {
        for row in self.button_matrix.iter_mut() {
            row.swap(a, b);
        }
    }

    fn mul_row(&mut self, row: usize, multiplicand: MatrixElt) {
        self.button_matrix[row].iter_mut().for_each(|val| *val *= multiplicand);
        self.requirements[row] *= multiplicand;
    }

    fn add_mul(&mut self, target_row: usize, multiple: MatrixElt, source_row: usize) {
        let num_cols = self.button_matrix[0].len();
        for col_idx in 0..num_cols {
            let new_val = multiple * self.button_matrix[source_row][col_idx];
            self.button_matrix[target_row][col_idx] += new_val;
        }
        let new_req_val = multiple * self.requirements[source_row];
        self.requirements[target_row] += new_req_val;
    }
}

fn row_reduce(requirements: &mut Requirements) {
    // We need a pivot on each row. To this end we are allowed three operations:
    //  swap(row a, row b) -- must also swap requirements[a] and [b]
    //  swap(col a, col b) -- has no impact on requirements
    //  row a -= c * row b -- must also have req[a] -= c * req[b]

    let mut next_row_idx = 0;
    let mut next_col_idx = 0;

    let num_cols = requirements.button_matrix[0].len();
    let num_rows = requirements.button_matrix.len();

    while next_row_idx < requirements.requirements.len() {
        // First, skip any columns where pivots are impossible due to being entirely zero
        // past existing pivot rows. If we have to do that, swap that column forward.
        let mut next_pivot_col = next_col_idx;
        while next_pivot_col < num_cols {
            let all_zero = (next_row_idx..num_rows).all(|row_idx| requirements.button_matrix[row_idx][next_pivot_col] == Fraction32::ZERO);

            if all_zero {
                next_pivot_col += 1;
            } else {
                break;
            }
        }

        if next_pivot_col == num_cols {
            // This can happen if the lights are linear dependent (for particular example, because
            // there are too many rows)
            return;
        }

        if next_pivot_col > next_col_idx {
            requirements.swap_cols(next_col_idx, next_pivot_col);
        }

        // Then, ensure the pivot element is on the top row
        let desired_pivot_row = (next_row_idx..num_rows)
            .find(|&row_idx| {
                let val = requirements.button_matrix[row_idx][next_col_idx];
                val != Fraction32::ZERO
            })
            .unwrap();

        let col_val = requirements.button_matrix[desired_pivot_row][next_col_idx];
        if col_val != Fraction32::ZERO {
            requirements.mul_row(desired_pivot_row, col_val.invert());
        }

        assert_eq!(requirements.button_matrix[desired_pivot_row][next_col_idx], Fraction32::ONE);

        if desired_pivot_row > next_row_idx {
            requirements.swap_rows(desired_pivot_row, next_row_idx);
        }

        // Then, clear out the rest of the pivot column
        for row_idx in 0..num_rows {
            if row_idx == next_row_idx {
                continue;
            }

            let multiple = -requirements.button_matrix[row_idx][next_col_idx];
            if multiple != Fraction32::ZERO {
                requirements.add_mul(row_idx, multiple, next_row_idx);
            }
        }

        // then set up for the next row
        next_row_idx += 1;
        next_col_idx += 1;
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
                1 << 3,
                (1 << 1) + (1 << 3),
                1 << 2,
                (1 << 2) + (1 << 3),
                (1 << 0) + (1 << 2),
                (1 << 0) + (1 << 1),
            ],
            joltage_requirements: vec![3, 5, 4, 7],
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn sample_b() {
        assert_eq!(b_with_input(SAMPLE), 33);
    }

    #[test]
    fn test_to_matrix() {
        let machines = parse(SAMPLE);

        let requirements: Vec<Requirements> = machines.into_iter().map(to_matrix).collect();

        let expected_0 = Requirements {
            button_matrix: vec![
                // each button becomes a COLUMN in this matrix
                vec![0.into(), 0.into(), 0.into(), 0.into(), 1.into(), 1.into()],
                vec![0.into(), 1.into(), 0.into(), 0.into(), 0.into(), 1.into()],
                vec![0.into(), 0.into(), 1.into(), 1.into(), 1.into(), 0.into()],
                vec![1.into(), 1.into(), 0.into(), 1.into(), 0.into(), 0.into()],
            ],
            requirements: vec![3.into(), 5.into(), 4.into(), 7.into()],
        };

        assert_eq!(expected_0, requirements[0]);
    }

    #[test]
    fn test_row_reduce() {
        // from sample[0]
        let mut start = Requirements {
            button_matrix: vec![
                // each button becomes a COLUMN in this matrix
                vec![0.into(), 0.into(), 0.into(), 0.into(), 1.into(), 1.into()],
                vec![0.into(), 1.into(), 0.into(), 0.into(), 0.into(), 1.into()],
                vec![0.into(), 0.into(), 1.into(), 1.into(), 1.into(), 0.into()],
                vec![1.into(), 1.into(), 0.into(), 1.into(), 0.into(), 0.into()],
            ],
            requirements: vec![3.into(), 5.into(), 4.into(), 7.into()],
        };

        // swap R0/R3; R0 -= R1; swap C3/C4; R2 -= R3
        row_reduce(&mut start);

        let expected = Requirements {
            button_matrix: vec![
                vec![1.into(), 0.into(), 0.into(), 0.into(), 1.into(), (-1).into()],
                vec![0.into(), 1.into(), 0.into(), 0.into(), 0.into(), 1.into()],
                vec![0.into(), 0.into(), 1.into(), 0.into(), 1.into(), (-1).into()],
                vec![0.into(), 0.into(), 0.into(), 1.into(), 0.into(), 1.into()],
            ],
            requirements: vec![2.into(), 5.into(), 1.into(), 3.into()],
        };

        assert_eq!(start, expected);
    }

    #[test]
    fn test_handle_negated_pivot() {
        let mut input = Requirements {
            button_matrix: vec![vec![1.into(), 1.into()], vec![1.into(), 0.into()]],
            requirements: vec![4.into(), 3.into()],
        };

        row_reduce(&mut input);

        // R1 -= R0; then R1 *= -1; then R0 -= R1
        let expected = Requirements {
            button_matrix: vec![vec![1.into(), 0.into()], vec![0.into(), 1.into()]],
            requirements: vec![3.into(), 1.into()],
        };

        assert_eq!(input, expected);
    }

    #[test]
    fn test_handle_premature_clearing_valid() {
        let mut input = Requirements {
            button_matrix: vec![
                vec![1.into(), 1.into()],
                vec![0.into(), 1.into()],
                // redundant light, it gets cleared out and trivialized
                vec![1.into(), 0.into()],
            ],
            // solution is: button 0 twice, button 1 twice
            requirements: vec![4.into(), 2.into(), 2.into()],
        };

        // Row actions: R2 -= R0; R0 -= R1; R2 += R1
        row_reduce(&mut input);

        let expected = Requirements {
            button_matrix: vec![
                vec![1.into(), 0.into()],
                vec![0.into(), 1.into()],
                // resulting zero row can be ignored (the light's requirements are somehow
                // guaranteed by the other button states, it's literally impossible to screw it up)
                vec![0.into(), 0.into()],
            ],
            // observe the requirement is also zero (if this wasn't true, the original constraints
            // would have been unsatisfiable)
            requirements: vec![2.into(), 2.into(), 0.into()],
        };

        assert_eq!(input, expected);
    }

    #[test]
    fn test_handle_premature_clearing_valid_trim() {
        let mut input = Requirements {
            button_matrix: vec![
                vec![1.into(), 1.into()],
                vec![0.into(), 1.into()],
                // redundant light, it gets cleared out and trivialized
                vec![1.into(), 0.into()],
            ],
            // solution is: button 0 twice, button 1 twice
            requirements: vec![4.into(), 2.into(), 2.into()],
        };

        // Row actions: R2 -= R0; R0 -= R1; R2 += R1
        row_reduce(&mut input);
        trim(&mut input); // drop last row

        let expected = Requirements {
            button_matrix: vec![vec![1.into(), 0.into()], vec![0.into(), 1.into()]],
            requirements: vec![2.into(), 2.into()],
        };

        assert_eq!(input, expected);
    }

    #[test]
    fn test_handle_premature_clearing_invalid() {
        let mut input = Requirements {
            button_matrix: vec![
                vec![1.into(), 1.into()],
                vec![0.into(), 1.into()],
                // redundant light, it gets cleared out and trivialized
                vec![1.into(), 0.into()],
            ],
            requirements: vec![4.into(), 2.into(), 1.into()],
        };

        let expected = Requirements {
            button_matrix: vec![
                vec![1.into(), 0.into()],
                vec![0.into(), 1.into()],
                // resulting zero row can be ignored (the light's requirements are somehow
                // guaranteed by the other button states, it's literally impossible to screw it up)
                vec![0.into(), 0.into()],
            ],
            // observe the final requirement is not zero, which means the original constraints were
            // unsatisfiable (light[2] is a LC of light[1] and light[0], but the requirement
            // doesn't match up)
            requirements: vec![2.into(), 2.into(), (-1).into()],
        };

        // Row actions: R2 -= R0; R0 -= R1; R2 += R1
        row_reduce(&mut input);

        assert_eq!(input, expected);
    }

    #[test]
    #[should_panic]
    fn test_handle_premature_clearing_invalid_trim() {
        let mut input = Requirements {
            button_matrix: vec![
                vec![1.into(), 1.into()],
                vec![0.into(), 1.into()],
                // redundant light, it gets cleared out and trivialized
                vec![1.into(), 0.into()],
            ],
            requirements: vec![4.into(), 2.into(), 1.into()],
        };

        // Row actions: R2 -= R0; R0 -= R1; R2 += R1
        row_reduce(&mut input);

        // should panic here, at the trimming
        trim(&mut input);
    }

    #[test]
    fn test_non_integer_reduction() {
        let mut input = Requirements {
            button_matrix: vec![
                vec![1.into(), 0.into(), 1.into(), 1.into()],
                vec![0.into(), 1.into(), 1.into(), 1.into()],
                vec![1.into(), 1.into(), 0.into(), 1.into()],
            ],
            // "real" solution A+B+2C+D
            requirements: vec![4.into(), 4.into(), 3.into()],
        };

        let expected = Requirements {
            button_matrix: vec![
                vec![1.into(), 0.into(), 0.into(), Fraction32 { top: 1, bot: 2 }],
                vec![0.into(), 1.into(), 0.into(), Fraction32 { top: 1, bot: 2 }],
                vec![0.into(), 0.into(), 1.into(), Fraction32 { top: 1, bot: 2 }],
            ],
            requirements: vec![
                Fraction32 { top: 3, bot: 2 },
                Fraction32 { top: 3, bot: 2 },
                Fraction32 { top: 5, bot: 2 },
            ],
        };

        // R2 -= R0; R2 -= R1; R0 -= R1; R2 -= R1; R2 *= -1/2; R1 -= R2
        row_reduce(&mut input);

        assert_eq!(input, expected);
    }
}
