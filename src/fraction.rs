use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

use num_integer::Integer;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Fraction32 {
    pub top: i32,
    // Really this needs to be positive, and relatively prime to top, but it's not
    // worth encoding that into the type system for an ADVENT OF CODE problem
    pub bot: i32,
}

impl Fraction32 {
    pub const ZERO: Fraction32 = Fraction32 { top: 0, bot: 1 };
    pub const ONE: Fraction32 = Fraction32 { top: 1, bot: 1 };

    pub fn invert(self) -> Self {
        if self.top == 0 {
            panic!("Cannot divide by zero");
        }
        Self {
            top: self.bot,
            bot: self.top,
        }
    }

    pub fn is_integer(self) -> bool {
        self.bot == 1
    }

    pub fn is_nonnegative(self) -> bool {
        self.top >= 0
    }
}

impl Debug for Fraction32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.top, self.bot)
    }
}

impl From<i32> for Fraction32 {
    fn from(value: i32) -> Self {
        Self { top: value, bot: 1 }
    }
}

impl Add for Fraction32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let a = self.top as i64;
        let b = self.bot as i64;
        let c = rhs.top as i64;
        let d = rhs.bot as i64;

        let den = b.checked_mul(d).unwrap();
        let num = a.checked_mul(d).unwrap().checked_add(b.checked_mul(c).unwrap()).unwrap();

        let gcd = num.gcd(&den);

        let num = num / gcd;
        let den = den / gcd;

        let mut top: i32 = num.try_into().unwrap();
        let mut bot: i32 = den.try_into().unwrap();

        if bot < 0 {
            bot = -bot;
            top = -top;
        }

        Self { top, bot }
    }
}

impl AddAssign for Fraction32 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul<i32> for Fraction32 {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self::Output {
        self * Fraction32 { top: rhs, bot: 1 }
    }
}

impl Mul<Fraction32> for i32 {
    type Output = Fraction32;

    fn mul(self, rhs: Fraction32) -> Self::Output {
        rhs * self
    }
}

impl Mul<Fraction32> for Fraction32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.top as i64;
        let b = self.bot as i64;
        let c = rhs.top as i64;
        let d = rhs.bot as i64;

        let den = b.checked_mul(d).unwrap();
        let num = a.checked_mul(c).unwrap();

        let gcd = num.gcd(&den);

        let num = num / gcd;
        let den = den / gcd;

        let mut top: i32 = num.try_into().unwrap();
        let mut bot: i32 = den.try_into().unwrap();

        if bot < 0 {
            bot = -bot;
            top = -top;
        }

        Self { top, bot }
    }
}

impl MulAssign for Fraction32 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = (*self) * rhs;
    }
}

impl Sub for Fraction32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl SubAssign for Fraction32 {
    fn sub_assign(&mut self, rhs: Self) {
        *self += -rhs;
    }
}

impl Neg for Fraction32 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            top: -self.top,
            bot: self.bot,
        }
    }
}

impl Ord for Fraction32 {
    fn cmp(&self, other: &Self) -> Ordering {
        let diff = (*self - *other).top;
        if diff > 0 {
            Ordering::Greater
        } else if diff == 0 {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for Fraction32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
