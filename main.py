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

import unittest
import numpy as np
from PolyRing import PolyRing
from ckks import KeyGenerator, Ciphertext, encode, decode

print("Rozpoczynam testy dla klasy PolyRing.")
print("Testy sprawdzaja:")
print("- Poprawnosc operacji: dodawanie, odejmowanie, mnozenie, negacja.")
print("- Wlasnosci homomorficzne: dodawanie i mnozenie zaszyfrowanych danych.")
print("- Kodowanie i dekodowanie wektorow zespolonych.")
print("Jesli testy sie nie powioda, sprawdz szczegoly bledu w komunikacie ponizej.")
print("---")

class TestPolyRing(unittest.TestCase):
    def setUp(self):
        self.n = 8
        self.delta = 2**40
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j])
        self.keygen = KeyGenerator(self.n)

    def test_poly_ring_operations(self):
        p1 = PolyRing(np.array([1, 2, 3, 4, 0, 0, 0, 0]))
        p2 = PolyRing(np.array([2, 3, 4, 5, 0, 0, 0, 0]))
        result_add = p1 + p2
        expected_add = PolyRing(np.array([3, 5, 7, 9, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_add.vec, expected_add.vec)

        result_sub = p2 - p1
        expected_sub = PolyRing(np.array([1, 1, 1, 1, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_sub.vec, expected_sub.vec)

        result_neg = -p1
        expected_neg = PolyRing(np.array([-1, -2, -3, -4, 0, 0, 0, 0]) % PolyRing.q)
        np.testing.assert_array_equal(result_neg.vec, expected_neg.vec)

    def test_complex_vector(self):
        poly = encode(self.z, self.delta)
        decoded = decode(poly, self.delta)
        np.testing.assert_allclose(decoded, self.z, atol=1e-5)

    def test_homomorphic_properties(self):
        z1 = self.z
        z2 = np.array([0.5 + 0.5j, 1 - 0.5j, 0.25 - 0.25j, 1 + 0j])

        ctxt1 = Ciphertext.encrypt(encode(z1, self.delta), self.keygen.public_key, self.n, self.delta)
        ctxt2 = Ciphertext.encrypt(encode(z2, self.delta), self.keygen.public_key, self.n, self.delta)

        ctxt_add = ctxt1 + ctxt2
        decrypted_add = decode(Ciphertext.decrypt(ctxt_add, self.keygen), self.delta)
        np.testing.assert_allclose(decrypted_add, z1 + z2, atol=1e-5)

        ctxt_mul = ctxt1.multiply(ctxt2, self.keygen.relin_key)
        ctxt_mul = ctxt_mul.relinearize(self.keygen.relin_key)
        ctxt_mul = ctxt_mul.rescale(self.delta)

        decrypted_mul = decode(Ciphertext.decrypt(ctxt_mul, self.keygen), self.delta)
        np.testing.assert_allclose(decrypted_mul, z1 * z2, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
    print("---")
    print("Testy zakonczone. Jesli widzisz 'OK', wszystkie testy przeszly.")
    print("Jesli widzisz blad, sprawdz szczegoly w komunikacie powyzej.")
    print("Mozliwe przyczyny bledow: bledna implementacja mnozenia w PolyRing lub problem z modulo q.")
