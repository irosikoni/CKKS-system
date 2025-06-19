import unittest
import numpy as np
from PolyRing import PolyRing
from ckks import CKKSContext, Ciphertext, encode, decode

print("Rozpoczynam testy dla CKKS.")
print("Testy sprawdzają:")
print("- Poprawność operacji na wielomianach (PolyRing).")
print("- Kodowanie i dekodowanie wektorów zespolonych.")
print("- Właściwości homomorficzne: dodawanie i mnożenie zaszyfrowanych danych.")
print("- Operacje relinearyzacji i reskalowania.")
print("- Przełączanie kluczy.")
print("---")

class TestCKKS(unittest.TestCase): # Zmieniono nazwę klasy testowej na ogólniejszą
    def setUp(self):
        # Definicja parametrów kontekstu CKKS
        self.N = 1024 # Zgodnie z PolyRing.N
        self.q_sizes = [60, 40, 40, 60] # Rozmiary bitowe dla łańcucha modułów
        self.delta_bits = 40 # Rozmiar bitowy dla global_delta

        # Inicjalizacja kontekstu CKKS
        self.context = CKKSContext(self.N, self.q_sizes, self.delta_bits)

        # Klucze są teraz generowane przez kontekst
        self.keygen = self.context.keygen 

        # Wektor liczb zespolonych do testów (długość N/2 = 512 dla N=1024)
        # Zmniejszmy liczbę slotów, żeby test był szybszy.
        # N slotów = N/2.
        # Dla N=1024, slots=512. Trzeba dostosować self.z
        # Albo użyć mniejszego N dla testów jednostkowych.
        # Zostawmy N=8 dla prostych testów, żeby było szybciej.
        PolyRing.N = 8 # Tymczasowo ustaw PolyRing.N na 8 dla tego testu
        self.N = PolyRing.N
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j]) # Length N/2 = 4
        
        # Ponownie inicjujemy kontekst i keygen z nowym N
        # To jest hack, bo PolyRing.N powinno być ustawione raz.
        # Idealnie, PolyRing.N powinno być const i testy powinny używać tego.
        # W praktyce, dla unittestu, zmieniamy PolyRing.N na mniejsze, żeby testy były szybkie.
        # W prawdziwej aplikacji, N jest ustawione na początku i niezmienne.
        self.context = CKKSContext(self.N, self.q_sizes, self.delta_bits)
        self.keygen = self.context.keygen 


    def test_poly_ring_operations(self):
        """
        Testuje podstawowe operacje na wielomianach PolyRing.
        """
        N_poly_test = PolyRing.N # N dla tego konkretnego testu
        q = self.context.current_q # Aktualny moduł z kontekstu

        # Test dodawania
        p1 = PolyRing(np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=np.int64), q)
        p2 = PolyRing(np.array([2, 3, 4, 5, 0, 0, 0, 0], dtype=np.int64), q)
        
        result_add = p1 + p2
        expected_add = PolyRing(np.array([3, 5, 7, 9, 0, 0, 0, 0], dtype=np.int64), q)
        np.testing.assert_array_equal(result_add.vec, expected_add.vec, err_msg="Add failed")

        # Test odejmowania
        result_sub = p2 - p1
        expected_sub = PolyRing(np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64), q)
        np.testing.assert_array_equal(result_sub.vec, expected_sub.vec, err_msg="Sub failed")

        # Test negacji
        result_neg = -p1
        expected_neg = PolyRing((-np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=np.int64) % q + q) % q, q)
        np.testing.assert_array_equal(result_neg.vec, expected_neg.vec, err_msg="Negation failed")

        # Test mnożenia wielomianów niskiego stopnia
        p3 = PolyRing(np.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=np.int64), q)
        p4 = PolyRing(np.array([2, 3, 0, 0, 0, 0, 0, 0], dtype=np.int64), q)
        result_mul = p3 * p4
        expected_mul_coeffs = np.array([2, 7, 6, 0, 0, 0, 0, 0], dtype=np.int64)
        expected_mul = PolyRing(expected_mul_coeffs, q)
        np.testing.assert_array_equal(result_mul.vec, expected_mul.vec, err_msg="Multiplication failed for low degree")

        # Test mnożenia z redukcją x^N = -1 (mod x^N+1)
        p_x7 = PolyRing(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64), q) # x^7
        p_x1 = PolyRing(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64), q) # x
        result_x8 = p_x7 * p_x1
        expected_x8 = PolyRing(np.array([(q - 1), 0, 0, 0, 0, 0, 0, 0], dtype=np.int64), q)
        np.testing.assert_array_equal(result_x8.vec, expected_x8.vec, err_msg="Multiplication with x^N reduction failed")
        
        # Test mnożenia z redukcją x^(N+k) = -x^k (mod x^N+1)
        p_x2 = PolyRing(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64), q) # x^2
        result_x9 = p_x7 * p_x2
        expected_x9 = PolyRing(np.array([0, (q - 1), 0, 0, 0, 0, 0, 0], dtype=np.int64), q)
        np.testing.assert_array_equal(result_x9.vec, expected_x9.vec, err_msg="Multiplication with x^N+k reduction failed")


    def test_complex_vector(self):
        """
        Testuje kodowanie wektora liczb zespolonych do wielomianu PolyRing i z powrotem.
        """
        poly = encode(self.z, self.context.global_delta, self.context.current_q)
        decoded = decode(poly, self.context.global_delta)
        np.testing.assert_allclose(decoded, self.z, atol=1e-5, err_msg="Encode/Decode failed")

    def test_homomorphic_properties(self):
        """
        Testuje homomorficzne dodawanie i mnożenie zaszyfrowanych danych.
        """
        z1 = self.z
        z2 = np.array([0.5 + 0.5j, 1 - 0.5j, 0.25 - 0.25j, 1 + 0j])

        # Kodowanie wiadomości jawnych
        m1 = encode(z1, self.context.global_delta, self.context.current_q)
        m2 = encode(z2, self.context.global_delta, self.context.current_q)
        
        # Szyfrowanie wiadomości. Przekazujemy kontekst, nie tylko N i delta.
        ctxt1 = Ciphertext.encrypt(m1, self.keygen.public_key, self.context)
        ctxt2 = Ciphertext.encrypt(m2, self.keygen.public_key, self.context)

        # Homomorficzne Dodawanie
        ctxt_add = ctxt1 + ctxt2
        decrypted_add_poly = Ciphertext.decrypt(ctxt_add, self.keygen)
        decrypted_add = decode(decrypted_add_poly, self.context.global_delta)
        np.testing.assert_allclose(decrypted_add, z1 + z2, atol=1e-3, err_msg="Homomorphic addition failed")

        # Homomorficzne Mnożenie
        ctxt_mul_raw = ctxt1.multiply(ctxt2) # Mnożenie nie przyjmuje już relin_key
        
        # Relinearyzacja redukuje szyfrogram do dwóch komponentów i redukuje moduł.
        ctxt_mul_relin = ctxt_mul_raw.relinearize() # Relinearyzacja operuje na kontekście szyfrogramu
        
        # Reskalowanie normalizuje deltę szyfrogramu. Odbywa się po relinearyzacji.
        # Nowy `global_delta` szyfrogramu `ctxt_mul_relin` jest już zaktualizowany przez `relinearize`
        # Jeśli chcemy, aby wynik miał oryginalną skalę `self.context.global_delta`, musimy to przekazać.
        # Rescaling needs to divide by the original scaling factor or some intermediate one.
        # The result of relinearize already performed one modulus switch and scaling.
        # The delta in `ctxt_mul_relin` is now `self.context.global_delta / P_drop`.
        # To bring it back to `self.context.global_delta`, we multiply by `P_drop`.
        # No, the `relinearize` itself handles one scaling.
        # So, after relinearize, `ctxt_mul_relin.context.global_delta` is `original_delta * original_delta / P_drop`.
        # This is where `target_delta` is key.
        
        # After `relinearize`, the effective plaintext scale in `ctxt_mul_relin` is `initial_delta^2 / P_factor_for_rescale`.
        # If we want to bring it back to `initial_delta`, we need to divide by `initial_delta / P_factor_for_rescale`.
        # This is `ctxt_mul_relin.delta / target_delta`
        
        # Let's assume after relinearize, `ctxt_mul_relin.delta` holds `original_delta_for_plaintexts`.
        # So we just tell `rescale` to target that.
        ctxt_mul_rescaled = ctxt_mul_relin.rescale(self.context.global_delta)
        
        # Deszyfracja i dekodowanie wyniku
        decrypted_mul_poly = Ciphertext.decrypt(ctxt_mul_rescaled, self.keygen)
        # Dekodowanie odbywa się przy finalnej delcie.
        np.testing.assert_allclose(decrypted_mul_poly.to_complex_vector(ctxt_mul_rescaled.delta), z1 * z2, atol=1e-2, err_msg="Homomorphic multiplication failed")


    def test_rescale_after_scalar_multiplication(self):
        """
        Testuje mnożenie szyfrogramu przez skalar bez jawnej operacji rescale,
        ponieważ delta szyfrogramu nie zmienia się w tym przypadku.
        """
        z = self.z
        factor = 5 

        m = encode(z, self.context.global_delta, self.context.current_q)
        ctxt = Ciphertext.encrypt(m, self.keygen.public_key, self.context)

        # Mnożenie przez skalar: wartość zaszyfrowana się zmienia, delta szyfrogramu NIE zmienia się automatycznie.
        ctxt_scaled = ctxt * factor 
        
        # Deszyfrujemy używając ORYGINALNEJ delty, ponieważ nie było operacji rescale zmieniającej deltę szyfrogramu.
        decrypted = decode(Ciphertext.decrypt(ctxt_scaled, self.keygen), self.context.global_delta)
        
        np.testing.assert_allclose(decrypted, z * factor, atol=1e-3, err_msg="Scalar multiplication failed without explicit rescale")


    def test_key_switch(self):
        """
        Testuje operację przełączania kluczy.
        """
        # Użyjemy osobnych kontekstów dla keygen1 i keygen2, aby symulować przełączanie.
        # W praktyce, keygen2 mógłby być po prostu innym kluczem w tym samym kontekście.
        
        # PolyRing.N = 8 # Upewnij się, że PolyRing.N jest ustawione dla tej instancji
        
        context1 = CKKSContext(self.N, self.q_sizes, self.delta_bits)
        context2 = CKKSContext(self.N, self.q_sizes, self.delta_bits) # Nowy kontekst = nowe klucze

        keygen1 = context1.keygen
        keygen2 = context2.keygen

        z = self.z
        m = encode(z, context1.global_delta, context1.current_q)

        ctxt = Ciphertext.encrypt(m, keygen1.public_key, context1)
        
        # Generujemy klucz do przełączania z secret_key keygen1 na secret_key keygen2
        # Klucz przełączający powinien być generowany przez keygen2 dla s_old (z keygen1)
        ks_key = keygen2.generate_key_switch_key(keygen1.secret_key)

        # Wykonujemy przełączanie kluczy
        ctxt_switched = ctxt.key_switch(ks_key)
        
        # Deszyfrujemy przełączony szyfrogram, używając klucza z keygen2
        decrypted_poly = Ciphertext.decrypt(ctxt_switched, keygen2)
        z_decoded = decode(decrypted_poly, context2.global_delta) # Decode z deltą kontekstu docelowego

        # Sprawdzamy, czy odszyfrowana wartość jest bliska oryginalnej
        np.testing.assert_allclose(z_decoded, z, atol=1e-2, err_msg="Key switch failed")


class TestPolyRingEncoding(unittest.TestCase): # Nazwa pozostała stara dla testów tylko kodowania/dekodowania
    def setUp(self):
        # Parametry N, delta dla tego testu
        self.N = 8 # Użyjemy mniejszego N dla szybkości testu
        PolyRing.N = self.N # Ustawiamy N w klasie PolyRing
        self.q_sizes = [60] # Wystarczy jeden moduł dla prostego testu kodowania
        self.delta_bits = 40

        self.context = CKKSContext(self.N, self.q_sizes, self.delta_bits)
        
        self.num_slots = self.N // 2
        self.z = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j, 3 + 0j]) # Długość = num_slots


    def test_encode_decode_inverse(self):
        """
        Testuje inwersyjność operacji kodowania i dekodowania.
        """
        # Kodowanie używa global_delta i current_q z kontekstu
        poly = encode(self.z, self.context.global_delta, self.context.current_q)
        # Dekodowanie używa global_delta z kontekstu
        decoded = decode(poly, self.context.global_delta)

        np.testing.assert_allclose(decoded, self.z, atol=1e-5, err_msg="Encoding/decoding inverse failed")

if __name__ == '__main__':
    unittest.main()
    print("---")
    print("Testy zakończone. Jeśli widzisz 'OK', wszystkie testy przeszły.")
    print("Jeśli widzisz błąd, sprawdź szczegóły błędu w komunikacie powyżej.")
    print("Możliwe przyczyny błędów: błędy implementacji lub problemy z akumulacją szumu.")