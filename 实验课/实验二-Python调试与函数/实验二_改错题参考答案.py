"""实验二改错题参考答案，供助教验收使用。"""


def shipping_fee(amount):
    if amount >= 99:
        return 0
    return 8


def add_bonus(score_text):
    result = int(score_text) + 5
    if result > 100:
        result = 100
    return result


def first_negative_index(values):
    for index in range(len(values)):
        if values[index] < 0:
            return index
    return -1


def get_email(student):
    return student.get("email", "未填写")


def score_level(score):
    if score >= 85:
        return "excellent"
    if score >= 60:
        return "pass"
    return "fail"


def sum_non_negative(values):
    total = 0
    for value in values:
        if value >= 0:
            total += value
    return total


def count_cups(category_cups):
    total = 0
    for cups in category_cups.values():
        total += cups
    return total


def parse_confidence(text):
    try:
        value = float(text)
    except ValueError:
        return None
    if value < 0 or value > 1:
        return None
    return value


class RealNumber:
    def __init__(self, value):
        self.value = float(value)

    def __repr__(self):
        return f"RealNumber({self.value})"

    def __add__(self, other):
        return RealNumber(self.value + other.value)

    def __mul__(self, other):
        return RealNumber(self.value * other.value)

    def __abs__(self):
        return abs(self.value)

    def to_complex(self):
        return ComplexNumber(self.value, 0)


class ComplexNumber:
    def __init__(self, real, imag=0):
        self.real = float(real)
        self.imag = float(imag)

    def __repr__(self):
        return f"ComplexNumber({self.real}, {self.imag})"

    def __add__(self, other):
        return ComplexNumber(
            self.real + other.real,
            self.imag + other.imag,
        )

    def __mul__(self, other):
        return ComplexNumber(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __abs__(self):
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)


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
assert (RealNumber(3) + RealNumber("-2")).value == 1.0
assert (RealNumber(3) * RealNumber("-2")).value == -6.0
assert abs(RealNumber(-2)) == 2.0

z1 = ComplexNumber(3, 4)
z2 = ComplexNumber(1, -2)
assert ((z1 + z2).real, (z1 + z2).imag) == (4.0, 2.0)
assert ((z1 * z2).real, (z1 * z2).imag) == (11.0, -2.0)
assert abs(z1) == 5.0
assert (z1.conjugate().real, z1.conjugate().imag) == (3.0, -4.0)
assert (z1.real, z1.imag) == (3.0, 4.0)
assert (RealNumber(2).to_complex() + ComplexNumber(3, 4)).real == 5.0

print("✅ 实验二基础题与类练习参考答案自测通过")
