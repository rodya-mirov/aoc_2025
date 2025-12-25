// I disagree with specific applications of this lint often enough that I'm just turning it off
// globally
#![allow(clippy::needless_range_loop)]
// this lint just sucks
#![allow(clippy::manual_range_contains)]
#![allow(clippy::comparison_chain)]

use std::env;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Write;
use std::time::Instant;

mod fraction;

mod day01;
mod day02;
mod day03;
mod day04;
mod day05;
mod day06;
mod day07;
mod day08;
mod day09;
mod day10;
mod day11;
mod day12;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum Side {
    A,
    B,
}

impl Display for Side {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::A => f.write_char('a'),
            Side::B => f.write_char('b'),
        }
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        Err("Usage: [run] [problemnumber] [subcase] ; eg:\n\tcargo run --release -- 1 a".to_string())
    } else {
        let a: i32 = args[1]
            .parse::<i32>()
            .map_err(|s| format!("Cannot parse argument '{}' as int", s))?;
        let b: Side = match args[2].as_str() {
            "a" => Ok(Side::A),
            "b" => Ok(Side::B),
            _err => Err(format!(
                "Cannot parse argument '{}' as subcase; should be 'a' or 'b'",
                args[0].as_str()
            )),
        }?;

        let start = Instant::now();

        let out: String = match (a, b) {
            (1, Side::A) => Ok(day01::a().to_string()),
            (1, Side::B) => Ok(day01::b().to_string()),
            (2, Side::A) => Ok(day02::a().to_string()),
            (2, Side::B) => Ok(day02::b().to_string()),
            (3, Side::A) => Ok(day03::a().to_string()),
            (3, Side::B) => Ok(day03::b().to_string()),
            (4, Side::A) => Ok(day04::a().to_string()),
            (4, Side::B) => Ok(day04::b().to_string()),
            (5, Side::A) => Ok(day05::a().to_string()),
            (5, Side::B) => Ok(day05::b().to_string()),
            (6, Side::A) => Ok(day06::a().to_string()),
            (6, Side::B) => Ok(day06::b().to_string()),
            (7, Side::A) => Ok(day07::a().to_string()),
            (7, Side::B) => Ok(day07::b().to_string()),
            (8, Side::A) => Ok(day08::a().to_string()),
            (8, Side::B) => Ok(day08::b().to_string()),
            (9, Side::A) => Ok(day09::a().to_string()),
            (9, Side::B) => Ok(day09::b().to_string()),
            (10, Side::A) => Ok(day10::a().to_string()),
            (10, Side::B) => Ok(day10::b().to_string()),
            (11, Side::A) => Ok(day11::a().to_string()),
            (11, Side::B) => Ok(day11::b().to_string()),
            (12, Side::A) => Ok(day12::a().to_string()),
            (12, Side::B) => Ok(day12::b().to_string()),
            (day, side) => Err(format!("Day {}, side {} is not yet supported", day, side)),
        }?;

        let elapsed = start.elapsed();

        println!("Day {} -- {}:\n{}", a, b, out);
        println!("Took {0:3} ms", elapsed.as_secs_f32() * 1000.0);

        Ok(())
    }
}
