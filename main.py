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

def test_homomorphic_consistency(trials=10000, delta=2**40, tolerance=1e-3):
    n = 8
    num_slots = n // 2

    all_errors = []
    max_errors_per_trial = []
    passed = 0

    for i in range(trials):
        z = np.random.randn(num_slots) + 1j * np.random.randn(num_slots)
        m = PolyRing.from_complex_vector(z, delta)

        s, public_key = generate_keys(n)
        c0, c1 = encrypt(m, public_key, n)
        decrypted_poly = decrypt(c0, c1, s)
        z_decoded = decrypted_poly.to_complex_vector(delta)

        errors = np.abs(z - z_decoded)
        all_errors.extend(errors)
        max_error = np.max(errors)
        max_errors_per_trial.append(max_error)

        if max_error < tolerance:
            passed += 1
        else:
            print(f"❌ Próba {i+1}: Błąd maksymalny = {max_error:.5e}")

    all_errors_array = np.array(all_errors)
    max_errors_array = np.array(max_errors_per_trial)

    print(f"\nTesty zakończone: {passed}/{trials} prób udanych (tolerancja: {tolerance})")
    print(f"Statystyki błędów:")
    print(f"  Maksymalny błąd bezwzględny     : {np.max(all_errors_array):.5e}")
    print(f"  Minimalny błąd bezwzględny       : {np.min(all_errors_array):.5e}")
    print(f"  Średni błąd bezwzględny          : {np.mean(all_errors_array):.5e}")
    print(f"  Średni maksymalny błąd na próbę  : {np.mean(max_errors_array):.5e}")
# TODO:
# Test operations on PolyRing
# Test if it's homomorphic
# Test if it works with complex vectors

def main():
    test_poly_ring_operations()
    test_homomorphic_consistency(trials=10000, delta=2**40, tolerance=1e-9)

if __name__ == "__main__":
    main()