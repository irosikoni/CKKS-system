from PolyRing import PolyRing
import numpy as np
import unittest

class TestPolyRing(unittest.TestCase):
    def setUp(self):
        self.n = 8
        self.delta = 2**40
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j])

    def test_poly_ring_operations(self):
        # est dodawania
        p1 = PolyRing(np.array([1, 2, 3, 4, 0, 0, 0, 0]))
        p2 = PolyRing(np.array([2, 3, 4, 5, 0, 0, 0, 0]))
        result_add = p1 + p2
        expected_add = PolyRing(np.array([3, 5, 7, 9, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_add.vec, expected_add.vec)

        # Test odejmowania
        result_sub = p2 - p1
        expected_sub = PolyRing(np.array([1, 1, 1, 1, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_sub.vec, expected_sub.vec)

        # Test mnożenia
        p3 = PolyRing(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
        p4 = PolyRing(np.array([2, 0, 0, 0, 0, 0, 0, 0]))
        result_mul = p3 * p4
        expected_mul = PolyRing(np.array([2, 0, 0, 0, 0, 0, 0, -2]) % PolyRing.q)
        np.testing.assert_array_equal(result_mul.vec, expected_mul.vec)

        # Test negacji
        result_neg = -p1
        expected_neg = PolyRing(np.array([-1, -2, -3, -4, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_neg.vec, expected_neg.vec)


def small_random_poly(n, bound=1):
    coeffs = np.random.randint(-bound, bound + 1, size=n)
    return PolyRing(coeffs)

def noise_poly(n):
    return small_random_poly(n, bound=5)

def generate_keys(n):
    s = small_random_poly(n)
    a = PolyRing(np.random.randint(0, PolyRing.q, size=n))
    e = noise_poly(n)

    b = (-a * s) + e
    return s, (a, b)

def encrypt(m: PolyRing, public_key, n):
    a, b = public_key
    u = small_random_poly(n)
    e1 = noise_poly(n)
    e2 = noise_poly(n)

    c0 = (b * u) + e1 + m
    c1 = (a * u) + e2
    return c0, c1

def decrypt(c0: PolyRing, c1: PolyRing, s: PolyRing):
    m_prime = c0 + (c1 * s)
    return m_prime

# TODO:
# Test operations on PolyRing
# Test if it's homomorphic
# Test if it works with complex vectors

def main():
    test_poly_ring_operations()


if __name__ == "__main__":
    main()