"""实验二：面向对象编程练习。

补全下列类中标有 TODO 的方法，不要修改文件末尾的测试代码。
本文件不需要读取输入，也不需要打印输出；运行后没有异常即表示测试通过。
"""

from __future__ import annotations

from collections.abc import Iterator


class Pair:
    """保存两个数值分量的通用基类。"""

    def __init__(
        self,
        first: int | float | str,
        second: int | float | str,
    ) -> None:
        # TODO：将两个分量转换为 float 后保存。
        pass

    def __repr__(self) -> str:
        # TODO：返回 Pair(first=..., second=...)。
        pass

    def as_tuple(self) -> tuple[int | float, int | float]:
        # TODO：返回由两个分量组成的普通元组。
        pass

    def __eq__(self, other: object) -> bool:
        # TODO：只比较实际类型相同的对象；类型不同时返回 NotImplemented。
        pass

    def __iter__(self) -> Iterator[int | float]:
        # TODO：使二元对象支持 x, y = pair。
        pass


class Vector2D(Pair):
    """二维向量。"""

    def __repr__(self) -> str:
        # TODO：返回 Vector2D(x=..., y=...)。
        pass

    def __add__(self, other: Vector2D) -> Vector2D:
        # TODO：返回两个向量之和。
        pass

    def __abs__(self) -> float:
        # TODO：返回向量的欧氏模长。
        pass

    def dot(self, other: Vector2D) -> float:
        # TODO：返回点积标量。
        pass

    def hadamard(self, other: Vector2D) -> Vector2D:
        # TODO：返回对应分量相乘得到的新向量。
        pass

    def __matmul__(self, other: Vector2D) -> float:
        # TODO：使 v1 @ v2 等价于 v1.dot(v2)。
        pass

    def __mul__(self, other: Vector2D) -> Vector2D:
        # TODO：使 v1 * v2 等价于 v1.hadamard(v2)。
        pass

    def __rmul__(self, scalar: int | float) -> Vector2D:
        # TODO：支持左侧标量乘法，如 3 * v。
        pass


class ComplexNumber(Pair):
    """复数 a + bi。"""

    def __repr__(self) -> str:
        # TODO：返回 ComplexNumber(real=..., imag=...)。
        pass

    def __str__(self) -> str:
        # TODO：返回 a + bi 或 a - bi。
        pass

    @property
    def real(self) -> float:
        # TODO：返回实部。
        pass

    @property
    def imag(self) -> float:
        # TODO：返回虚部。
        pass

    def __add__(self, other: ComplexNumber) -> ComplexNumber:
        # TODO：返回复数之和。
        pass

    def __sub__(self, other: ComplexNumber) -> ComplexNumber:
        # TODO：返回复数之差。
        pass

    def __mul__(self, other: ComplexNumber) -> ComplexNumber:
        # TODO：实现 (a + bi)(c + di) = (ac - bd) + (ad + bc)i。
        pass

    def __abs__(self) -> float:
        # TODO：返回复数模长。
        pass

    def conjugate(self) -> ComplexNumber:
        # TODO：返回新的共轭复数，不修改原对象。
        pass

    def __neg__(self) -> ComplexNumber:
        # TODO：返回该复数的相反数。
        pass


class Rational(Pair):
    """保持最简形式且分母为正的有理数。"""

    def __init__(self, numerator: int, denominator: int = 1) -> None:
        # TODO：转换为整数、拒绝零分母、统一符号并使用 gcd 约分。
        pass

    def __repr__(self) -> str:
        # TODO：返回 Rational(numerator=..., denominator=...)。
        pass

    def __str__(self) -> str:
        # TODO：返回 numerator/denominator。
        pass

    @property
    def numerator(self) -> int:
        # TODO：返回分子。
        pass

    @property
    def denominator(self) -> int:
        # TODO：返回分母。
        pass

    def __add__(self, other: Rational) -> Rational:
        # TODO：返回两个有理数之和。
        pass

    def __sub__(self, other: Rational) -> Rational:
        # TODO：返回两个有理数之差。
        pass

    def __mul__(self, other: Rational) -> Rational:
        # TODO：返回两个有理数之积。
        pass

    def __truediv__(self, other: Rational) -> Rational:
        # TODO：返回两个有理数之商；除以零时抛出 ZeroDivisionError。
        pass

    def __float__(self) -> float:
        # TODO：返回对应的浮点数。
        pass

    def __neg__(self) -> Rational:
        # TODO：返回该有理数的相反数。
        pass

    def __lt__(self, other: Rational) -> bool:
        # TODO：使用整数交叉相乘比较，不转换为 float。
        pass


# 不要修改以下测试代码。

pair = Pair(3, "4")
assert pair.first == 3.0 and pair.second == 4.0
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
assert r1.as_tuple() == (1, 2)
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
assert not (Rational(3, 4) < Rational(2, 3))

x, y = v1
assert (x, y) == (3.0, 4.0)
assert Vector2D(1, 2) != ComplexNumber(1, 2)

try:
    Rational(1, 0)
except ValueError:
    pass
else:
    assert False, "分母为零时应抛出 ValueError"

try:
    r1 / Rational(0, 1)
except ZeroDivisionError:
    pass
else:
    assert False, "除以零有理数时应抛出 ZeroDivisionError"
