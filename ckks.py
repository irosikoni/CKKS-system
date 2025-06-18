import numpy as np
from PolyRing import PolyRing


class Ciphertext:
    def __init__(self, c0: PolyRing, c1: PolyRing, delta: float):
        self.c0 = c0
        self.c1 = c1
        self.delta = delta
        self._last_relin_key = None  # dla .__mul__()

    def __add__(self, other):
        assert self.delta == other.delta
        return Ciphertext(self.c0 + other.c0, self.c1 + other.c1, self.delta)

    def __mul__(self, other):
        raise NotImplementedError("Użyj metody `multiply(other, relin_key)` i `relinearize(relin_key)` jawnie.")


    def multiply(self, other, relin_key):
        assert self.delta == other.delta
        d0 = self.c0 * other.c0
        d1 = self.c0 * other.c1 + self.c1 * other.c0
        d2 = self.c1 * other.c1
        a_r, b_r = relin_key
        c0 = d0 + b_r * d2
        c1 = d1 + a_r * d2
        new_ctxt = Ciphertext(self._rescale(c0), self._rescale(c1), self.delta)
        new_ctxt._last_relin_key = relin_key
        return new_ctxt

    def relinearize(self, relin_key):
        self._last_relin_key = relin_key
        return self

    def rescale(self, delta):
        self.c0 = self._rescale(self.c0)
        self.c1 = self._rescale(self.c1)
        self.delta = delta
        return self

    def decrypt_and_decode(self, secret_key: PolyRing):
        m = self.c0 + self.c1 * secret_key
        return m.to_complex_vector(self.delta)

    def _rescale(self, poly: PolyRing) -> PolyRing:
        scaled = np.round(poly.vec.astype(np.float64) / self.delta)
        reduced = np.mod(scaled, PolyRing.q).astype(np.int64)
        return PolyRing(reduced)

    @staticmethod
    def encrypt(m: PolyRing, public_key, n, delta):
        a, b = public_key
        u = Ciphertext._small_random_poly(n)
        e1 = Ciphertext._noise_poly(n)
        e2 = Ciphertext._noise_poly(n)
        c0 = b * u + e1 + m
        c1 = a * u + e2
        return Ciphertext(c0, c1, delta)

    @staticmethod
    def decrypt(ciphertext, keygen):
        return ciphertext.c0 + ciphertext.c1 * keygen.secret_key

    @staticmethod
    def generate_keys(n):
        s = Ciphertext._small_random_poly(n)
        a = PolyRing(np.random.randint(0, PolyRing.q, size=n))
        e = Ciphertext._noise_poly(n)
        b = -a * s + e
        return s, (a, b)

    @staticmethod
    def generate_relin_key(s, n):
        s_squared = s * s
        a_r = PolyRing(np.random.randint(0, PolyRing.q, size=n))
        e_r = Ciphertext._small_random_poly(n, bound=5)
        b_r = -a_r * s_squared + e_r
        return a_r, b_r

    @staticmethod
    def _small_random_poly(n, bound=1):
        return PolyRing(np.random.randint(-bound, bound + 1, size=n))

    @staticmethod
    def _noise_poly(n):
        return Ciphertext._small_random_poly(n, bound=5)


def encode(z: np.ndarray, delta: float) -> PolyRing:
    return PolyRing.from_complex_vector(z, delta)


def decode(poly: PolyRing, delta: float) -> np.ndarray:
    return poly.to_complex_vector(delta)


class KeyGenerator:
    def __init__(self, n):
        self.n = n
        self.delta = 2**40
        self.secret_key, self.public_key = Ciphertext.generate_keys(n)
        self.relin_key = Ciphertext.generate_relin_key(self.secret_key, n)

    def encrypt(self, z: np.ndarray) -> Ciphertext:
        poly = encode(z, self.delta)
        return Ciphertext.encrypt(poly, self.public_key, self.n, self.delta)

    def decrypt(self, ctxt: Ciphertext) -> np.ndarray:
        return decode(Ciphertext.decrypt(ctxt, self), self.delta)
