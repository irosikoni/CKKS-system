import numpy as np

class PolyRing():
    q = 17
    f = np.array([1, 0, 0, 0, 1])
    def __init__(self, vec):
        _, r = np.polydiv(np.array(vec), self.f)
        self.vec = (r % self.q).astype(int)

    def __repr__(self):
        return str(self.vec)

    def __add__(self, other):
        if isinstance(other, PolyRing):
            return PolyRing(np.polyadd(self.vec, other.vec))
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'PolyRing' and '{type(other).__name__}'")

    def __sub__(self, other):
        if isinstance(other, PolyRing):
            return PolyRing(np.polysub(self.vec, other.vec))
        else:
            raise TypeError(f"unsupported operand type(s) for -: 'PolyRing' and '{type(other).__name__}'")

    def __mul__(self, other):
        if isinstance(other, PolyRing):
            return PolyRing(np.polymul(self.vec, other.vec))
        elif isinstance(other, int):
            return PolyRing(other * self.vec)
        else:
            raise TypeError(f"unsupported operand type(s) for *: 'PolyRing' and '{type(other).__name__}'")

    def __rmul__(self, other):
        if isinstance(other, int):
            return PolyRing(other * self.vec)
        
    @classmethod
    def from_complex_vector(cls, z_vec, delta):
        n = len(cls.f) - 1
        z_full = np.concatenate([z_vec, z_vec[::-1]])
        z_scaled = z_full * delta
        coeffs = np.fft.ifft(z_scaled)
        rounded = np.round(coeffs.real).astype(np.int64)
        return cls(rounded)

    def to_complex_vector(self, delta):
        coeffs = self.vec.astype(np.float64)
        z_approx = np.fft.fft(coeffs) / delta
        n = len(z_approx) // 2
        return z_approx[:n]
