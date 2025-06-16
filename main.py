# Plik zawiera testy dla klasy PolyRing, ktora implementuje pierscien wielomianow
# uzywany w szyfrowaniu homomorficznym. Testy weryfikują:
# 1. Poprawnosc operacji na PolyRing (dodawanie, odejmowanie, mnozenie, negacja).
# 2. Wlasnosci homomorficzne szyfrowania (dodawanie i mnozenie zaszyfrowanych danych).
# 3. Kodowanie i dekodowanie wektorow zespolonych oraz ich szyfrowanie/deszyfrowanie.
#
# Wymagania:
# - Python 3.x
# - Zainstalowany modul numpy (`pip install numpy`)
# - Plik `PolyRing.py` z definicja klasy `PolyRing` w tym samym katalogu
#
# Jak uruchomic testy:
# 1. Upewnij sie, ze plik `PolyRing.py` znajduje sie w tym samym katalogu.
# 2. W terminalu przejdz do katalogu z plikiem i wpisz:
#    python -m unittest main.py
# 3. Wynik pokaze, czy testy przeszly (`OK`) czy ktorys sie nie powiodl (z opisem bledu).
#
# Uwagi:
# - Testy uzywaja tolerancji numerycznej (atol=1e-5) ze wzgledu na szum w szyfrowaniu.
# - Upewnij sie, ze masz wystarczajaca ilosc pamieci, poniewaz operacje na duzych
#   wielomianach moga byc zasobozerne.

from PolyRing import PolyRing
import numpy as np
import unittest

# wyprintowanie informacji w terminalu przed testami
print("Rozpoczynam testy dla klasy PolyRing.")
print("Testy sprawdzaja:")
print("- Poprawnosc operacji: dodawanie, odejmowanie, mnozenie, negacja.")
print("- Wlasnosci homomorficzne: dodawanie i mnozenie zaszyfrowanych danych.")
print("- Kodowanie i dekodowanie wektorow zespolonych.")
print("Jesli testy sie nie powioda, sprawdz szczegoly bledu w komunikacie ponizej.")
print("---")

class TestPolyRing(unittest.TestCase):
    # wywolanie przed kazdym testem 
    def setUp(self):
        self.n = 8
        self.delta = 2**40
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j])

    def test_poly_ring_operations(self):
        # test dodawania
        p1 = PolyRing(np.array([1, 2, 3, 4, 0, 0, 0, 0]))
        p2 = PolyRing(np.array([2, 3, 4, 5, 0, 0, 0, 0]))
        result_add = p1 + p2
        expected_add = PolyRing(np.array([3, 5, 7, 9, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_add.vec, expected_add.vec)

        # test odejmowania
        result_sub = p2 - p1
        expected_sub = PolyRing(np.array([1, 1, 1, 1, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_sub.vec, expected_sub.vec)

        # TODO: test mnozenia

        # test negacji
        result_neg = -p1
        expected_neg = PolyRing(np.array([-1, -2, -3, -4, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_neg.vec, expected_neg.vec)
    
    def test_homomorphic_properties(self):
        # generowanie kluczy
        s, public_key = generate_keys(self.n)

        # tworzenie dwoch wiadomosci
        z1 = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j])
        z2 = np.array([0.5 + 0.5j, 1 - 0.5j, 0.25 - 0.25j, 1 + 0j])
        m1 = PolyRing.from_complex_vector(z1, self.delta)
        m2 = PolyRing.from_complex_vector(z2, self.delta)

        # szyfrowanie obu wiadomosci
        c0_m1, c1_m1 = encrypt(m1, public_key, self.n)
        c0_m2, c1_m2 = encrypt(m2, public_key, self.n)

        # test homomorficznego dodawania
        c0_add = c0_m1 + c0_m2
        c1_add = c1_m1 + c1_m2
        decrypted_add = decrypt(c0_add, c1_add, s)
        z_decoded_add = decrypted_add.to_complex_vector(self.delta)
        expected_add = z1 + z2
        np.testing.assert_allclose(z_decoded_add, expected_add, atol=1e-5)

        # test homomorficznego mnozenia
        c0_mul = c0_m1 * c0_m2
        c1_mul = c1_m1 * c1_m2
        decrypted_mul = decrypt(c0_mul, c1_mul, s)
        z_decoded_mul = decrypted_mul.to_complex_vector(self.delta)
        expected_mul = z1 * z2
        np.testing.assert_allclose(z_decoded_mul, expected_mul, atol=1e-5)
        
    def test_complex_vector(self):
        # test kodowania i dekodowania wektorow zespolonych
        m = PolyRing.from_complex_vector(self.z, self.delta)
        z_decoded = m.to_complex_vector(self.delta)
        np.testing.assert_allclose(z_decoded, self.z, atol=1e-5)

        # test szyfrowania i deszyfrowania z wektorami zespolonymi
        s, public_key = generate_keys(self.n)
        c0, c1 = encrypt(m, public_key, self.n)
        decrypted_poly = decrypt(c0, c1, s)
        z_decoded_encrypted = decrypted_poly.to_complex_vector(self.delta)
        np.testing.assert_allclose(z_decoded_encrypted, self.z, atol=1e-5)


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
# Test operations on PolyRing - multiplication left
# check if other tests are correct

if __name__ == '__main__':
    unittest.main()
    print("---")
    print("Testy zakonczone. Jesli widzisz 'OK', wszystkie testy przeszly.")
    print("Jesli widzisz blad, sprawdz szczegoly w komunikacie powyzej.")
    print("Mozliwe przyczyny bledow: bledna implementacja mnozenia w PolyRing lub problem z modulo q.")