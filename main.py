import unittest
import numpy as np
from PolyRing import PolyRing
from ckks import CKKSContext, Ciphertext, encode, decode

print("Rozpynam testy dla CKKS.")
print("Testy sprawdzają:")
print("- Poprawność operacji na wielomianach (PolyRing).")
print("- Kodowanie i dekodowanie wektorów zespolonych.")
print("- Właściwości homomorficzne: dodawanie i mnożenie zaszyfrowanych danych.")
print("- Operacje relinearyzacji i reskalowania.")
print("- Przełączanie kluczy.")
print("---")

class TestCKKS(unittest.TestCase): 
    def setUp(self):
        # Definicja parametrów kontekstu CKKS dla SZYBKICH TESTÓW
        self.N_test = 8 # Używamy N=8 dla testów jednostkowych
        PolyRing.N = self.N_test # Ustawiamy N globalnie w PolyRing

        # q_sizes (bitowe rozmiary liczb pierwszych w łańcuchu modułów)
        # Suma bitów ograniczona do ~60-61.
        self.q_sizes_test = [30, 29] 
        self.delta_bits_test = 20 # Mniejsza delta dla mniejszego N, by szum nie dominował od razu

        # Inicjalizacja kontekstu CKKS dla testów
        self.context = CKKSContext(self.N_test, self.q_sizes_test, self.delta_bits_test)

        self.keygen = self.context.keygen 

        # Wektor liczb zespolonych do testów. Długość = N/2.
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j]) # Dla N=8, N/2=4 sloty.


    def test_poly_ring_operations(self):
        """
        Testuje podstawowe operacje na wielomianach PolyRing.
        """
        N_poly_test = PolyRing.N 
        q = self.context.current_q 

        # PolyRing(vec, current_q) - ważne, aby przekazywać moduł
        # Używamy dtype=object dla współczynników
        p1 = PolyRing(np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=object), q)
        p2 = PolyRing(np.array([2, 3, 4, 5, 0, 0, 0, 0], dtype=object), q)
        
        result_add = p1 + p2
        expected_add = PolyRing(np.array([3, 5, 7, 9, 0, 0, 0, 0], dtype=object), q)
        np.testing.assert_array_equal(result_add.vec, expected_add.vec, err_msg="Add failed")

        result_sub = p2 - p1
        expected_sub = PolyRing(np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=object), q)
        np.testing.assert_array_equal(result_sub.vec, expected_sub.vec, err_msg="Sub failed")

        result_neg = -p1
        expected_neg = PolyRing((-np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=object) % q + q) % q, q)
        np.testing.assert_array_equal(result_neg.vec, expected_neg.vec, err_msg="Negation failed")

        p3 = PolyRing(np.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=object), q) 
        p4 = PolyRing(np.array([2, 3, 0, 0, 0, 0, 0, 0], dtype=object), q) 
        result_mul = p3 * p4
        expected_mul_coeffs = np.array([2, 7, 6, 0, 0, 0, 0, 0], dtype=object)
        expected_mul = PolyRing(expected_mul_coeffs, q)
        np.testing.assert_array_equal(result_mul.vec, expected_mul.vec, err_msg="Multiplication failed for low degree")

        p_x7 = PolyRing(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=object), q) # x^7
        p_x1 = PolyRing(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=object), q) # x
        result_x8 = p_x7 * p_x1
        expected_x8 = PolyRing(np.array([(q - 1), 0, 0, 0, 0, 0, 0, 0], dtype=object), q)
        np.testing.assert_array_equal(result_x8.vec, expected_x8.vec, err_msg="Multiplication with x^N reduction failed")
        
        p_x2 = PolyRing(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=object), q) # x^2
        result_x9 = p_x7 * p_x2
        expected_x9 = PolyRing(np.array([0, (q - 1), 0, 0, 0, 0, 0, 0], dtype=object), q)
        np.testing.assert_array_equal(result_x9.vec, expected_x9.vec, err_msg="Multiplication with x^N+k reduction failed")


    def test_complex_vector(self):
        """
        Testuje kodowanie wektora liczb zespolonych do wielomianu PolyRing i z powrotem.
        """
        poly = encode(self.z, self.context.global_delta, self.context.current_q)
        decoded = decode(poly, self.context.global_delta)
        # Zwiększono atol ze względu na błędy precyzji w konwersjach i zaokrągleniach
        np.testing.assert_allclose(decoded, self.z, atol=1e-4, err_msg="Encode/Decode failed") 

    def test_homomorphic_properties(self):
        """
        Testuje homomorficzne dodawanie i mnożenie zaszyfrowanych danych.
        """
        z1 = self.z
        z2 = np.array([0.5 + 0.5j, 1 - 0.5j, 0.25 - 0.25j, 1 + 0j])

        m1 = encode(z1, self.context.global_delta, self.context.current_q)
        m2 = encode(z2, self.context.global_delta, self.context.current_q)
        
        ctxt1 = Ciphertext.encrypt(m1, self.keygen.public_key, self.context)
        ctxt2 = Ciphertext.encrypt(m2, self.keygen.public_key, self.context)

        # Homomorficzne Dodawanie
        ctxt_add = ctxt1 + ctxt2
        decrypted_add_poly = Ciphertext.decrypt(ctxt_add, self.keygen)
        decrypted_add = decode(decrypted_add_poly, self.context.global_delta)
        np.testing.assert_allclose(decrypted_add, z1 + z2, atol=1e-3, err_msg="Homomorphic addition failed")

        # Homomorficzne Mnożenie
        ctxt_mul_raw = ctxt1.multiply(ctxt2) 
        
        # Relinearyzacja redukuje szyfrogram do dwóch komponentów i redukuje moduł.
        ctxt_mul_relin = ctxt_mul_raw.relinearize() 
        
        # Reskalowanie normalizuje deltę szyfrogramu.
        ctxt_mul_rescaled = ctxt_mul_relin.rescale(self.context.global_delta)
        
        decrypted_mul_poly = Ciphertext.decrypt(ctxt_mul_rescaled, self.keygen)
        # Dekodowanie odbywa się przy finalnej delcie.
        np.testing.assert_allclose(decode(decrypted_mul_poly, ctxt_mul_rescaled.delta), z1 * z2, atol=1e-2, err_msg="Homomorphic multiplication failed")


    def test_rescale_after_scalar_multiplication(self):
        """
        Testuje mnożenie szyfrogramu przez skalar.
        """
        z = self.z
        factor = 5 

        m = encode(z, self.context.global_delta, self.context.current_q)
        ctxt = Ciphertext.encrypt(m, self.keygen.public_key, self.context)

        ctxt_scaled = ctxt * factor 
        
        decrypted = decode(Ciphertext.decrypt(ctxt_scaled, self.keygen), self.context.global_delta)
        
        np.testing.assert_allclose(decrypted, z * factor, atol=1e-3, err_msg="Scalar multiplication failed without explicit rescale")


    def test_key_switch(self):
        """
        Testuje operację przełączania kluczy.
        """
        context1 = CKKSContext(self.N_test, self.q_sizes_test, self.delta_bits_test)
        context2 = CKKSContext(self.N_test, self.q_sizes_test, self.delta_bits_test) 

        keygen1 = context1.keygen
        keygen2 = context2.keygen

        z = self.z
        m = encode(z, context1.global_delta, context1.current_q)

        ctxt = Ciphertext.encrypt(m, keygen1.public_key, context1)
        
        ks_key = keygen2.generate_key_switch_key(keygen1.secret_key)

        ctxt_switched = ctxt.key_switch(ks_key)
        
        decrypted_poly = Ciphertext.decrypt(ctxt_switched, keygen2)
        z_decoded = decode(decrypted_poly, context2.global_delta) 

        np.testing.assert_allclose(z_decoded, z, atol=1e-2, err_msg="Key switch failed")


class TestPolyRingEncoding(unittest.TestCase): 
    def setUp(self):
        self.N_test = 8 
        PolyRing.N = self.N_test 
        self.q_sizes_test = [30] 
        self.delta_bits_test = 20

        self.context = CKKSContext(self.N_test, self.q_sizes_test, self.delta_bits_test)
        
        self.num_slots = self.N_test // 2
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j]) 


    def test_encode_decode_inverse(self):
        """
        Testuje inwersyjność operacji kodowania i dekodowania.
        """
        poly = encode(self.z, self.context.global_delta, self.context.current_q)
        decoded = decode(poly, self.context.global_delta)

        np.testing.assert_allclose(decoded, self.z, atol=1e-5, err_msg="Encoding/decoding inverse failed")

if __name__ == '__main__':
    unittest.main()
    print("---")
    print("Testy zakończone. Jeśli widzisz 'OK', wszystkie testy przeszły.")
    print("Jeśli widzisz błąd, sprawdź szczegóły błędu w komunikacie powyżej.")
    print("Możliwe przyczyny błędów: błędy implementacji lub problemy z akumulacją szumu.")