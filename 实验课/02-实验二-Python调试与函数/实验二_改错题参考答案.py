"""实验二改错题参考答案，供助教验收使用。"""

from __future__ import annotations

from collections.abc import Iterator
from math import gcd


def shipping_fee(amount: float) -> int:
    if amount >= 99:
        return 0
    return 8


def add_bonus(score_text: str) -> int:
    result = int(score_text) + 5
    if result > 100:
        result = 100
    return result


def first_negative_index(values: list[int | float]) -> int:
    for index in range(len(values)):
        if values[index] < 0:
            return index
    return -1


def get_email(student: dict[str, str]) -> str:
    return student.get("email", "未填写")


def score_level(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 60:
        return "pass"
    return "fail"


def sum_non_negative(values: list[int | float]) -> int | float:
    total = 0
    for value in values:
        if value >= 0:
            total += value
    return total


def count_cups(category_cups: dict[str, int]) -> int:
    total = 0
    for cups in category_cups.values():
        total += cups
    return total


def parse_confidence(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    if value < 0 or value > 1:
        return None
    return value


class Pair:
    def __init__(self, first: int | float | str, second: int | float | str) -> None:
        self.first = float(first)
        self.second = float(second)

    def __repr__(self) -> str:
        return f"Pair(first={self.first}, second={self.second})"

    def as_tuple(self) -> tuple[int | float, int | float]:
        return (self.first, self.second)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()

    def __iter__(self) -> Iterator[int | float]:
        return iter(self.as_tuple())


class Vector2D(Pair):
    def __repr__(self) -> str:
        return f"Vector2D(x={self.first}, y={self.second})"

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(
            self.first + other.first,
            self.second + other.second,
        )

    def __abs__(self) -> float:
        return (self.first**2 + self.second**2) ** 0.5

    def dot(self, other: "Vector2D") -> float:
        return self.first * other.first + self.second * other.second

    def hadamard(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(
            self.first * other.first,
            self.second * other.second,
        )

    def __matmul__(self, other: "Vector2D") -> float:
        return self.dot(other)

    def __mul__(self, other: "Vector2D") -> "Vector2D":
        return self.hadamard(other)

    def __rmul__(self, scalar: int | float) -> "Vector2D":
        return Vector2D(
            scalar * self.first,
            scalar * self.second,
        )


class ComplexNumber(Pair):
    def __repr__(self) -> str:
        return f"ComplexNumber(real={self.real}, imag={self.imag})"

    def __str__(self) -> str:
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real} {sign} {abs(self.imag)}i"

    @property
    def real(self) -> float:
        return self.first

    @property
    def imag(self) -> float:
        return self.second

    def __add__(self, other: "ComplexNumber") -> "ComplexNumber":
        return ComplexNumber(
            self.real + other.real,
            self.imag + other.imag,
        )

    def __sub__(self, other: "ComplexNumber") -> "ComplexNumber":
        return ComplexNumber(
            self.real - other.real,
            self.imag - other.imag,
        )

    def __mul__(self, other: "ComplexNumber") -> "ComplexNumber":
        return ComplexNumber(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __abs__(self) -> float:
        return (self.real**2 + self.imag**2) ** 0.5

    def conjugate(self) -> "ComplexNumber":
        return ComplexNumber(self.real, -self.imag)

    def __neg__(self) -> "ComplexNumber":
        return ComplexNumber(-self.real, -self.imag)


class Rational(Pair):
    def __init__(self, numerator: int, denominator: int = 1) -> None:
        numerator = int(numerator)
        denominator = int(denominator)
        if denominator == 0:
            raise ValueError("denominator cannot be zero")

        if denominator < 0:
            numerator = -numerator
            denominator = -denominator

        divisor = gcd(abs(numerator), denominator)
        self.first = numerator // divisor
        self.second = denominator // divisor

    def __repr__(self) -> str:
        return f"Rational(numerator={self.numerator}, denominator={self.denominator})"

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def numerator(self) -> int:
        return self.first

    @property
    def denominator(self) -> int:
        return self.second

    def __add__(self, other: "Rational") -> "Rational":
        return Rational(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )

    def __sub__(self, other: "Rational") -> "Rational":
        return Rational(
            self.numerator * other.denominator - other.numerator * self.denominator,
            self.denominator * other.denominator,
        )

    def __mul__(self, other: "Rational") -> "Rational":
        return Rational(
            self.numerator * other.numerator,
            self.denominator * other.denominator,
        )

    def __truediv__(self, other: "Rational") -> "Rational":
        if other.numerator == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return Rational(
            self.numerator * other.denominator,
            self.denominator * other.numerator,
        )

    def __float__(self) -> float:
        return self.numerator / self.denominator

    def __neg__(self) -> "Rational":
        return Rational(-self.numerator, self.denominator)

    def __lt__(self, other: "Rational") -> bool:
        return self.numerator * other.denominator < other.numerator * self.denominator


assert shipping_fee(99) == 0
assert shipping_fee(98.9) == 8
assert add_bonus("80") == 85
assert add_bonus("100") == 100
assert first_negative_index([3, 0, -2]) == 2
assert first_negative_index([]) == -1
assert get_email({}) == "未填写"
assert score_level(85) == "excellent"
assert sum_non_negative([3, -2, 5, 0]) == 8
assert sum_non_negative([]) == 0
assert count_cups({"coffee": 5, "tea": 3}) == 8
assert parse_confidence("NA") is None
assert parse_confidence("1") == 1.0
pair = Pair(3, "4")
assert pair.as_tuple() == (3.0, 4.0)
assert repr(pair) == "Pair(first=3.0, second=4.0)"
assert pair == Pair(3, 4)
assert pair != Pair(3, 5)

v1 = Vector2D(3, 4)
v2 = Vector2D(1, -2)
assert isinstance(v1, Pair)
assert (v1 + v2).as_tuple() == (4.0, 2.0)
assert abs(v1) == 5.0
assert v1.dot(v2) == -5.0
assert v1.hadamard(v2).as_tuple() == (3.0, -8.0)
assert v1 @ v2 == -5.0
assert (v1 * v2).as_tuple() == (3.0, -8.0)
assert (3 * v2).as_tuple() == (3.0, -6.0)
assert repr(v2) == "Vector2D(x=1.0, y=-2.0)"
assert v1.as_tuple() == (3.0, 4.0)

z1 = ComplexNumber(3, 4)
z2 = ComplexNumber(1, -2)
assert isinstance(z1, Pair)
assert (z1.real, z1.imag) == (3.0, 4.0)
assert (z1 + z2).as_tuple() == (4.0, 2.0)
assert (z1 - z2).as_tuple() == (2.0, 6.0)
assert (z1 * z2).as_tuple() == (11.0, -2.0)
assert abs(z1) == 5.0
assert z1.conjugate().as_tuple() == (3.0, -4.0)
assert z1.as_tuple() == (3.0, 4.0)
assert repr(z2) == "ComplexNumber(real=1.0, imag=-2.0)"
assert str(z1) == "3.0 + 4.0i"
assert str(z2) == "1.0 - 2.0i"
assert (-z2).as_tuple() == (-1.0, 2.0)
assert (ComplexNumber(0, 1) * ComplexNumber(0, 1)).as_tuple() == (-1.0, 0.0)

r1 = Rational(2, 4)
r2 = Rational(1, -3)
assert isinstance(r1, Pair)
assert (r1.numerator, r1.denominator) == (1, 2)
assert r2.as_tuple() == (-1, 3)
assert (r1 + r2).as_tuple() == (1, 6)
assert (r1 - r2).as_tuple() == (5, 6)
assert (r1 * r2).as_tuple() == (-1, 6)
assert (r1 / Rational(3, 4)).as_tuple() == (2, 3)
assert Rational(0, -7).as_tuple() == (0, 1)
assert float(r1) == 0.5
assert repr(r1) == "Rational(numerator=1, denominator=2)"
assert str(r1) == "1/2"
assert -r1 == Rational(-1, 2)
assert Rational(1, 3) < Rational(1, 2)

x, y = v1
assert (x, y) == (3.0, 4.0)
assert Vector2D(1, 2) != ComplexNumber(1, 2)

try:
    Rational(1, 0)
except ValueError:
    pass
else:
    raise AssertionError("分母为零时应抛出 ValueError")

try:
    r1 / Rational(0, 1)
except ZeroDivisionError:
    pass
else:
    raise AssertionError("除以零有理数时应抛出 ZeroDivisionError")

print("✅ 实验二基础题与类练习参考答案自测通过")
