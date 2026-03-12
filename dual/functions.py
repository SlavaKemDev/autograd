import math
from dual.DualNumber import DualNumber


def accepts_numbers(func):  # we need to accept int/float as well
    def wrapper(x):
        if not hasattr(x, 'dual'):
            x = DualNumber(x, 0)
        return func(x)

    return wrapper


# exp / log

@accepts_numbers
def exp(x: DualNumber) -> DualNumber:
    # return math.e ** x
    e_val = math.exp(x.real)
    return DualNumber(e_val, x.dual * e_val)  # this is faster


@accepts_numbers
def log(x: DualNumber) -> DualNumber:
    return DualNumber(math.log(x.real), x.dual / x.real)


# trigonometric functions

@accepts_numbers
def sin(x: DualNumber) -> DualNumber:
    return DualNumber(math.sin(x.real), x.dual * math.cos(x.real))


@accepts_numbers
def cos(x: DualNumber) -> DualNumber:
    return DualNumber(math.cos(x.real), -x.dual * math.sin(x.real))


@accepts_numbers
def tan(x: DualNumber) -> DualNumber:
    return sin(x) / cos(x)


@accepts_numbers
def cot(x: DualNumber) -> DualNumber:
    return cos(x) / sin(x)


# inverse trigonometric functions

@accepts_numbers
def asin(x: DualNumber) -> DualNumber:
    return DualNumber(
        math.asin(x.real),
        x.dual / (1 - x.real ** 2) ** 0.5
    )


@accepts_numbers
def acos(x: DualNumber) -> DualNumber:
    return DualNumber(
        math.acos(x.real),
        -x.dual / (1 - x.real ** 2) ** 0.5
    )


@accepts_numbers
def atan(x: DualNumber) -> DualNumber:
    return DualNumber(
        math.atan(x.real),
        x.dual / (1 + x.real ** 2)
    )


# hyperbolic functions

@accepts_numbers
def sinh(x: DualNumber) -> DualNumber:
    # return (exp(x) - exp(-x)) / 2
    return DualNumber(math.sinh(x.real), x.dual * math.cosh(x.real))  # this is faster


@accepts_numbers
def cosh(x: DualNumber) -> DualNumber:
    # return (exp(x) + exp(-x)) / 2
    return DualNumber(math.cosh(x.real), x.dual * math.sinh(x.real))  # this is faster


@accepts_numbers
def tanh(x: DualNumber) -> DualNumber:
    return sinh(x) / cosh(x)


# other functions

@accepts_numbers
def sqrt(x: DualNumber) -> DualNumber:
    sqrt_val = math.sqrt(x.real)
    return DualNumber(sqrt_val, x.dual / (2 * sqrt_val))


@accepts_numbers
def dabs(x: DualNumber) -> DualNumber:
    sign = 0
    if x.real > 0:
        sign = 1
    elif x.real < 0:
        sign = -1

    return DualNumber(
        abs(x.real),
        x.dual * sign
    )
