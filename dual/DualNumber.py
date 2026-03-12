import math

class DualNumber:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        else:
            return DualNumber(self.real + other, self.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        else:
            return DualNumber(self.real - other, self.dual)

    def __rsub__(self, other):
        return DualNumber(other, 0).__sub__(self)

    def __neg__(self):
        return DualNumber(-self.real, -self.dual)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return DualNumber(self.real * other, self.dual * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        if self.dual == 0:
            return f"{self.real}"
        if self.real == 0:
            return f"{self.dual}e"
        if self.dual > 0:
            return f"{self.real} + {self.dual}e"
        else:
            return f"{self.real} - {-self.dual}e"

    def __pow__(self, power, modulo=None):
        A = self.real
        B = self.dual
        C = power.real if isinstance(power, DualNumber) else power
        D = power.dual if isinstance(power, DualNumber) else 0

        if D == 0:
            return DualNumber(A ** C, B * (A ** (C - 1)) * C)

        if A <= 0:
            raise ValueError("Base must be positive when exponent is a dual number with non-zero dual part.")

        return DualNumber(
            A ** C,
            D * (A ** C) * math.log(A) + (A ** (C - 1)) * B * C
        )

    def __rpow__(self, other):
        return DualNumber(other, 0).__pow__(self)

    def __truediv__(self, other):
        A = self.real
        B = self.dual
        C = other.real if isinstance(other, DualNumber) else other
        D = other.dual if isinstance(other, DualNumber) else 0

        return DualNumber(
            A / C,
            B / C - (A * D) / (C * C)
        )

    def __rtruediv__(self, other):
        return DualNumber(other, 0).__truediv__(self)
