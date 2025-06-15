from PolyRing import PolyRing
import numpy as np

def test_poly_ring_operations():
    z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j])
    delta = 2**40
    n = 8

    m = PolyRing.from_complex_vector(z, delta)
    s, public_key = generate_keys(n)

    c0, c1 = encrypt(m, public_key, n)

    decrypted_poly = decrypt(c0, c1, s)
    z_decoded = decrypted_poly.to_complex_vector(delta)

    print("Oryginał:", z)
    print("Po deszyfrowaniu i dekodowaniu:", z_decoded)


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