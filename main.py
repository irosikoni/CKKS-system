import unittest
import numpy as np
from PolyRing import PolyRing
from ckks import KeyGenerator, Ciphertext, encode, decode

print("Rozpoczynam testy dla klasy PolyRing.")
print("Testy sprawdzają:")
print("- Poprawność operacji: dodawanie, odejmowanie, mnożenie, negacja.")
print("- Właściwości homomorficzne: dodawanie i mnożenie zaszyfrowanych danych.")
print("- Kodowanie i dekodowanie wektorów zespolonych.")
print("Jeśli testy się nie powiodą, sprawdź szczegóły błędu w komunikacie poniżej.")
print("---")

class TestPolyRing(unittest.TestCase):
    def setUp(self):
        # N musi być spójne z PolyRing.f (czyli 8 w tym przypadku)
        self.N = len(PolyRing.f) - 1
        self.delta = 2**40 # Przykładowa delta dla CKKS
        # Wektor liczb zespolonych do testów (długość N/2 = 4)
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j]) 
        # Inicjalizacja KeyGeneratora, potrzebna dla testów homomorficznych
        self.keygen = KeyGenerator(self.N) 

    def test_poly_ring_operations(self):
        """
        Testuje podstawowe operacje na wielomianach PolyRing:
        dodawanie, odejmowanie, negacja, mnożenie (w tym z redukcją modulo x^N+1).
        """
        N = len(PolyRing.f) - 1
        q = PolyRing.q # Modulo q z klasy PolyRing
        
        # Test dodawania
        p1 = PolyRing(np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=np.int64))
        p2 = PolyRing(np.array([2, 3, 4, 5, 0, 0, 0, 0], dtype=np.int64))
        
        result_add = p1 + p2
        expected_add = PolyRing(np.array([3, 5, 7, 9, 0, 0, 0, 0], dtype=np.int64))
        np.testing.assert_array_equal(result_add.vec, expected_add.vec, err_msg="Add failed")

        # Test odejmowania
        result_sub = p2 - p1
        expected_sub = PolyRing(np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64))
        np.testing.assert_array_equal(result_sub.vec, expected_sub.vec, err_msg="Sub failed")

        # Test negacji
        result_neg = -p1
        # Oczekiwana wartość dla -X mod Q to (Q-X) mod Q
        expected_neg = PolyRing((-np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=np.int64) % q + q) % q)
        np.testing.assert_array_equal(result_neg.vec, expected_neg.vec, err_msg="Negation failed")

        # Test mnożenia wielomianów niskiego stopnia (bez redukcji x^N)
        # (1 + 2x) * (2 + 3x) = 2 + 3x + 4x + 6x^2 = 2 + 7x + 6x^2
        p3 = PolyRing(np.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=np.int64))
        p4 = PolyRing(np.array([2, 3, 0, 0, 0, 0, 0, 0], dtype=np.int64))
        result_mul = p3 * p4
        expected_mul_coeffs = np.array([2, 7, 6, 0, 0, 0, 0, 0], dtype=np.int64)
        expected_mul = PolyRing(expected_mul_coeffs)
        np.testing.assert_array_equal(result_mul.vec, expected_mul.vec, err_msg="Multiplication failed for low degree")

        # Test mnożenia z redukcją x^N = -1 (mod x^N+1)
        # x^7 * x = x^8 = -1 (mod x^8+1)
        p_x7 = PolyRing(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)) # x^7
        p_x1 = PolyRing(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)) # x
        result_x8 = p_x7 * p_x1
        # Oczekiwana wartość dla -1 modulo q
        expected_x8 = PolyRing(np.array([(q - 1), 0, 0, 0, 0, 0, 0, 0], dtype=np.int64))
        np.testing.assert_array_equal(result_x8.vec, expected_x8.vec, err_msg="Multiplication with x^N reduction failed")
        
        # Test mnożenia z redukcją x^(N+k) = -x^k (mod x^N+1)
        # x^7 * x^2 = x^9 = x * x^8 = x * (-1) = -x (mod x^8+1)
        p_x2 = PolyRing(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)) # x^2
        result_x9 = p_x7 * p_x2
        # Oczekiwana wartość dla -x modulo q
        expected_x9 = PolyRing(np.array([0, (q - 1), 0, 0, 0, 0, 0, 0], dtype=np.int64))
        np.testing.assert_array_equal(result_x9.vec, expected_x9.vec, err_msg="Multiplication with x^N+k reduction failed")

    def test_complex_vector(self):
        """
        Testuje kodowanie wektora liczb zespolonych do wielomianu PolyRing i z powrotem.
        """
        poly = encode(self.z, self.delta)
        decoded = decode(poly, self.delta)
        np.testing.assert_allclose(decoded, self.z, atol=1e-5, err_msg="Encode/Decode failed")

    def test_homomorphic_properties(self):
        """
        Testuje homomorficzne dodawanie i mnożenie szyfrogramów.
        """
        z1 = self.z
        z2 = np.array([0.5 + 0.5j, 1 - 0.5j, 0.25 - 0.25j, 1 + 0j])

        # Kodowanie wiadomości jawnych
        m1 = encode(z1, self.delta)
        m2 = encode(z2, self.delta)
        # Szyfrowanie wiadomości
        ctxt1 = Ciphertext.encrypt(m1, self.keygen.public_key, self.N, self.delta)
        ctxt2 = Ciphertext.encrypt(m2, self.keygen.public_key, self.N, self.delta)

        # Homomorficzne Dodawanie
        ctxt_add = ctxt1 + ctxt2
        decrypted_add_poly = Ciphertext.decrypt(ctxt_add, self.keygen)
        decrypted_add = decode(decrypted_add_poly, self.delta)
        np.testing.assert_allclose(decrypted_add, z1 + z2, atol=1e-3, err_msg="Homomorphic addition failed")

        # Homomorficzne Mnożenie
        # Mnożenie szyfrogramów daje trzykomponentowy szyfrogram z podwojoną deltą.
        ctxt_mul_raw = ctxt1.multiply(ctxt2, self.keygen.relin_key)
        # Relinearyzacja redukuje szyfrogram do dwóch komponentów.
        ctxt_mul_relin = ctxt_mul_raw.relinearize(self.keygen.relin_key)
        
        # Reskalowanie normalizuje deltę szyfrogramu z powrotem do początkowej.
        ctxt_mul_rescaled = ctxt_mul_relin.rescale(self.delta)
        
        # Deszyfracja i dekodowanie wyniku
        decrypted_mul_poly = Ciphertext.decrypt(ctxt_mul_rescaled, self.keygen)
        decrypted_mul = decode(decrypted_mul_poly, self.delta) # Upewnij się, że ta linia jest obecna!
        np.testing.assert_allclose(decrypted_mul, z1 * z2, atol=1e-2, err_msg="Homomorphic multiplication failed")

    def test_rescale_after_scalar_multiplication(self):
        """
        Testuje mnożenie szyfrogramu przez skalar bez jawnej operacji rescale,
        ponieważ delta szyfrogramu nie zmienia się w tym przypadku.
        """
        z = self.z
        factor = 5 # Skalar

        m = encode(z, self.delta)
        ctxt = Ciphertext.encrypt(m, self.keygen.public_key, self.N, self.delta)

        # Mnożenie przez skalar: wartość zaszyfrowana się zmienia, delta szyfrogramu NIE zmienia się automatycznie.
        ctxt_scaled = ctxt * factor 
        
        # Deszyfrujemy używając ORYGINALNEJ delty, ponieważ nie było operacji rescale zmieniającej deltę szyfrogramu.
        decrypted = decode(Ciphertext.decrypt(ctxt_scaled, self.keygen), self.delta)
        
        # Oczekiwana wartość to oryginalny wektor pomnożony przez skalar.
        np.testing.assert_allclose(decrypted, z * factor, atol=1e-3, err_msg="Scalar multiplication failed without explicit rescale")


    def test_key_switch(self):
        """
        Testuje operację przełączania kluczy.
        UWAGA: Ta operacja jest skomplikowana w CKKS i uproszczona implementacja może być źródłem niedokładności.
        """
        N = len(PolyRing.f) - 1
        # Generujemy dwa różne KeyGeneratory, aby symulować przełączanie kluczy
        keygen1 = KeyGenerator(N)
        keygen2 = KeyGenerator(N) 

        z = self.z
        m = encode(z, self.delta)

        # Szyfrujemy wiadomość kluczem z keygen1
        ctxt = Ciphertext.encrypt(m, keygen1.public_key, N, self.delta)
        # Generujemy klucz do przełączania z secret_key keygen1 na secret_key keygen2
        ks_key = keygen2.generate_key_switch_key(keygen1.secret_key)

        # Wykonujemy przełączanie kluczy
        ctxt_switched = ctxt.key_switch(ks_key)
        
        # Deszyfrujemy przełączony szyfrogram, używając klucza z keygen2
        decrypted_poly = Ciphertext.decrypt(ctxt_switched, keygen2)
        z_decoded = decode(decrypted_poly, self.delta)

        # Sprawdzamy, czy odszyfrowana wartość jest bliska oryginalnej
        np.testing.assert_allclose(z_decoded, z, atol=1e-2, err_msg="Key switch failed")


class TestPolyRingEncoding(unittest.TestCase):
    def setUp(self):
        self.delta = 2**40
        self.N = len(PolyRing.f) - 1
        self.num_slots = self.N // 2
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j])

    def test_encode_decode_inverse(self):
        """
        Testuje inwersyjność operacji kodowania i dekodowania.
        """
        poly = PolyRing.from_complex_vector(self.z, self.delta)
        decoded = poly.to_complex_vector(self.delta)

        np.testing.assert_allclose(decoded, self.z, atol=1e-5, err_msg="Encoding/decoding inverse failed")

if __name__ == '__main__':
    unittest.main()
    print("---")
    print("Testy zakończone. Jeśli widzisz 'OK', wszystkie testy przeszły.")
    print("Jeśli widzisz błąd, sprawdź szczegóły błędu w komunikacie powyżej.")
    print("Możliwe przyczyny błędów: błędy implementacji lub problemy z akumulacją szumu.")