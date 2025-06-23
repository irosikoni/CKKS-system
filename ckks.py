import numpy as np
import random 
from PolyRing import PolyRing 

# --- Globalne funkcje pomocnicze (generowanie szumu) ---
def _small_random_poly(n, bound=1, current_q=None): # BOUND=1 jest ok dla s
    if current_q is None: current_q = PolyRing.q
    # For debugging, set bound=0 (no noise)
    coeffs = np.random.randint(0, 1, size=n).tolist()
    return PolyRing(coeffs, current_q)

# ckks.py, funkcja _noise_poly
def _noise_poly(n, bound=1, current_q=None): # UPEWNIJ SIĘ, ŻE TO JEST bound=1 (lub testowo 0)
    if current_q is None: current_q = PolyRing.q
    # For debugging, set bound=0 (no noise)
    coeffs = np.zeros(n, dtype=int).tolist()
    return PolyRing(coeffs, current_q)

# --- Kodowanie i Dekodowanie ---
def encode(z: np.ndarray, delta: float, current_q: int) -> PolyRing:
    return PolyRing.from_complex_vector(z, delta, current_q)

def decode(poly: PolyRing, delta: float) -> np.ndarray:
    return poly.to_complex_vector(delta)

# --- Klasa CKKSContext ---
class CKKSContext:
    def __init__(self, N: int, q_sizes: list[int], delta_bits: int):
        # Ustawienie N w klasie PolyRing i zaktualizowanie PolyRing.f
        PolyRing.N = N 
        PolyRing.f = np.array([1] + [0]*(N-1) + [1], dtype=object)
        self.N = N

        self.q_sizes = q_sizes
        self.delta_bits = delta_bits
        self.global_delta = 2**delta_bits

        self.primes = [] 
        self.q_chain = [] 
        
        current_product = 1

        _known_primes = {
            60: 2**60 - 93,
            40: 2**40 - 87, # Przykład, należy zweryfikować czy jest pierwsza.
            30: 2**30 - 35, # Liczba pierwsza
            29: 2**29 - 3   # Liczba pierwsza
        }

        for i, bits in enumerate(q_sizes):
            if bits in _known_primes:
                prime_candidate = _known_primes[bits]
                # For multiple moduli of the same size, add small offsets
                if i > 0 and bits in [bits_prev for bits_prev in q_sizes[:i]]:
                    prime_candidate += 2 * i  # Small offset to make different primes
            else:
                print(f"Ostrzeżenie: Brak zdefiniowanej liczby pierwszej dla {bits} bitów. Używam losowej (może nie być pierwsza).")
                prime_candidate = random.getrandbits(bits)
                if prime_candidate < 2**(bits-1): prime_candidate += 2**(bits-1)
                if prime_candidate % 2 == 0: prime_candidate += 1
            
            self.primes.append(prime_candidate)
            current_product *= self.primes[-1]
            self.q_chain.append(current_product)

        self.current_modulus_idx = len(self.q_chain) - 1 
        self.current_q = self.q_chain[self.current_modulus_idx]
        PolyRing.q = self.current_q 

        self.keygen = KeyGenerator(self) 
        self.galois_keys = {} 

# --- Klasa Ciphertext ---
class Ciphertext:
    def __init__(self, c0: PolyRing, c1: PolyRing, context: CKKSContext, initial_delta: float, d2: PolyRing = None):
        self.c0 = c0
        self.c1 = c1
        self.context = context 
        self._d2 = d2
        self.delta = initial_delta # Dodano atrybut delta do obiektu Ciphertext

    def __add__(self, other):
        assert self.context.current_q == other.context.current_q, "Moduli Q must match for addition"
        return Ciphertext(self.c0 + other.c0, self.c1 + other.c1, self.context, self.delta)

    def __mul__(self, other):
        if isinstance(other, Ciphertext):
            raise ValueError("Użyj .multiply(other) dla mnożenia szyfrogram-szyfrogram.")
        elif isinstance(other, (int, float, np.integer, np.floating)):
            return Ciphertext(self.c0 * other, self.c1 * other, self.context, self.delta)
        else:
            raise TypeError("Nieobsługiwany typ mnożenia")

    def multiply(self, other):
        """
        Homomorficzne mnożenie dwóch szyfrogramów.
        Zwraca trzykomponentowy szyfrogram, który wymaga relinearyzacji i skalowania.
        """
        assert self.context.current_q == other.context.current_q, "Moduli Q must match for multiplication"
        
        d0 = self.c0 * other.c0
        d1 = self.c0 * other.c1 + self.c1 * other.c0
        d2 = self.c1 * other.c1
        
        new_effective_delta = self.delta * other.delta # Aktualizacja delty
        
        return Ciphertext(d0, d1, self.context, new_effective_delta, d2=d2)

    def relinearize(self):
        """
        Wykonuje relinearyzację szyfrogramu z trzech komponentów (c0,c1,d2) do dwóch (c0',c1').
        Łączy się z operacją Modulus Switching, redukując moduł szyfrogramu.
        """
        if self._d2 is None:
            raise ValueError("Brak składnika c2 (d2) – najpierw wykonaj multiply().")
        
        Q_current = self.context.current_q
        current_mod_idx = self.context.current_modulus_idx
        
        if current_mod_idx == 0:
            raise ValueError("Nie można przeprowadzić relinearyzacji: osiągnięto ostatni moduł w łańcuchu.")
        
        next_mod_idx = current_mod_idx - 1
        Q_next = self.context.q_chain[next_mod_idx]
        P_drop = self.context.primes[current_mod_idx] 

        relin_key_for_stage = self.context.keygen.relin_key[current_mod_idx - 1] 
        A_r, B_r = relin_key_for_stage

        temp_c0 = self.c0 + self._d2 * B_r
        temp_c1 = self.c1 + self._d2 * A_r

        c0_final = Ciphertext._rescale_poly_by_factor(temp_c0, P_drop)
        c1_final = Ciphertext._rescale_poly_by_factor(temp_c1, P_drop)

        # Stwórz nowy kontekst dla wynikowego szyfrogramu, aby odzwierciedlić zmianę modułu
        new_context = CKKSContext(self.context.N, self.context.q_sizes, self.context.delta_bits)
        new_context.primes = self.context.primes
        new_context.q_chain = self.context.q_chain
        new_context.current_modulus_idx = next_mod_idx
        new_context.current_q = Q_next
        PolyRing.q = new_context.current_q 

        new_effective_delta_after_relinearize = self.delta / P_drop 
        
        return Ciphertext(c0_final, c1_final, new_context, new_effective_delta_after_relinearize, d2=None)

    @staticmethod
    def _rescale_poly_by_factor(poly: PolyRing, factor: float) -> PolyRing:
        """ Skaluje współczynniki wielomianu przez podany współczynnik.
        Używa float() do konwersji współczynników, aby uniknąć przepełnienia.
        Współczynnik musi być różny od zera.
        """
        if factor == 0:
            raise ValueError("Współczynnik skalowania nie może być zerem.")

        # First reduce modulo the current modulus to ensure coefficients are in the correct range
        current_q = poly._current_q
        reduced_coeffs = np.array([(int(x) % current_q + current_q) % current_q for x in poly.vec], dtype=object)
        
        # Then scale by the factor
        scaled_coeffs_float = np.array([float(x) for x in reduced_coeffs], dtype=np.float64)
        scaled_coeffs_float = np.round(scaled_coeffs_float / factor)

        # Convert back to integers and reduce modulo the new modulus
        new_q = current_q // int(factor)  # The new modulus after dropping P_drop
        final_coeffs = np.array([(int(x) % new_q + new_q) % new_q for x in scaled_coeffs_float], dtype=object)

        return PolyRing(final_coeffs, new_q)

    def rescale(self, target_delta: float):
        """
        Reskaluje szyfrogram do nowego, pożądanego współczynnika delta.
        To jest oddzielna operacja od modulus switching w relinearyzacji.
        """
        rescale_factor = self.delta / target_delta 
        
        c0_rescaled = Ciphertext._rescale_poly_by_factor(self.c0, rescale_factor)
        c1_rescaled = Ciphertext._rescale_poly_by_factor(self.c1, rescale_factor)
        
        d2_rescaled = None
        if self._d2 is not None:
             d2_rescaled = Ciphertext._rescale_poly_by_factor(self._d2, rescale_factor)

        new_ciphertext_obj = Ciphertext(c0_rescaled, c1_rescaled, self.context, target_delta, d2=d2_rescaled)
        return new_ciphertext_obj

    def key_switch(self, key_switch_key):
        """
        Wykonuje operację przełączania kluczy szyfrogramu ze starego klucza tajnego na nowy.
        UWAGA: To jest uproszczona implementacja.
        """
        a_ks, b_ks = key_switch_key
        c0_new = self.c0 + self.c1 * b_ks
        c1_new = self.c1 * a_ks
        return Ciphertext(c0_new, c1_new, self.context, self.delta) 

    @staticmethod
    def encrypt(m: PolyRing, public_key, context: CKKSContext):
        """
        Szyfruje zaszyfrowaną wiadomość (w postaci wielomianu) przy użyciu klucza publicznego.
        """
        N_poly = context.N 
        current_q = context.current_q 
        
        a, b = public_key 
        
        u = _small_random_poly(N_poly, current_q=current_q)
        e1 = _noise_poly(N_poly, current_q=current_q)
        e2 = _noise_poly(N_poly, current_q=current_q)
        
        c0 = b * u + e1 + m
        c1 = a * u + e2
        
        return Ciphertext(c0, c1, context, context.global_delta, d2=None)

    @staticmethod
    def decrypt(ciphertext, keygen):
        """
        Deszyfruje szyfrogram przy użyciu klucza tajnego.
        Zwraca wielomian, który jest przybliżeniem oryginalnej wiadomości.
        """
        # Update the global modulus to match the ciphertext's modulus
        original_q = PolyRing.q
        PolyRing.q = ciphertext.c0._current_q
        
        # Create a temporary secret key with the correct modulus
        temp_secret_key = PolyRing(keygen.secret_key.vec, ciphertext.c0._current_q)
        
        # Perform decryption
        result = ciphertext.c0 + ciphertext.c1 * temp_secret_key
        
        # Restore the original global modulus
        PolyRing.q = original_q
        
        return result

# --- Klasa KeyGenerator ---

class KeyGenerator:
    def __init__(self, context: CKKSContext):
        self.context = context
        self.secret_key, self.public_key = self._generate_keys()
        self.relin_key = self._generate_relin_key(self.secret_key)
        self.galois_keys = {} 

    def _generate_keys(self):
        N = self.context.N
        current_q = self.context.current_q
        # Używamy float() do log2, aby uniknąć OverflowError dla bardzo dużego current_q
        max_q_bits = int(np.ceil(np.log2(float(current_q)))) 

        s = _small_random_poly(N, bound=1, current_q=current_q)
        
        a_coeffs = []
        for _ in range(N):
            random_large_int = random.getrandbits(max_q_bits + 10) 
            coeff = random_large_int % current_q
            a_coeffs.append(coeff)
        a = PolyRing(a_coeffs, current_q)
        
        e = _noise_poly(N, current_q=current_q)
        
        b = -a * s + e
        return s, (a, b)

    def _generate_relin_key(self, s: PolyRing):
        """
        Generuje klucze relinearyzacji.
        Zwraca listę kluczy (A_j, B_j) dla każdego etapu modulus chain.
        """
        relin_keys_list = []
        
        max_q_bits = int(np.ceil(np.log2(float(self.context.current_q)))) 

        for idx in range(len(self.context.q_chain) - 1, 0, -1): 
            Q_current_for_key = self.context.q_chain[idx] 
            P_drop_for_key = self.context.primes[idx] 
            
            a_r_coeffs = []
            for _ in range(self.context.N):
                rand_val = random.getrandbits(max_q_bits + 10)
                a_r_coeffs.append(rand_val % Q_current_for_key)
            A_r = PolyRing(a_r_coeffs, Q_current_for_key)
            
            E_r = _noise_poly(self.context.N, current_q=Q_current_for_key)
            
            s_temp = PolyRing(s.vec, Q_current_for_key)
            s_squared_temp = s_temp * s_temp
            s_squared_scaled = s_squared_temp * P_drop_for_key 
            
            B_r = -A_r * s_temp + E_r + s_squared_scaled
            
            relin_keys_list.append((A_r, B_r))
        
        return relin_keys_list[::-1] 

    def generate_key_switch_key(self, s_old: PolyRing):
        """
        Generuje klucz przełączania kluczy (key switch key) z s_old na self.secret_key (s_new).
        Uproszczona implementacja, bez dekompozycji bazowej.
        """
        N = self.context.N
        current_q = self.context.current_q
        max_q_bits = int(np.ceil(np.log2(float(current_q))))

        a_ks_coeffs = []
        for _ in range(N):
            rand_val = random.getrandbits(max_q_bits + 10)
            a_ks_coeffs.append(rand_val % current_q)
        a_ks = PolyRing(a_ks_coeffs, current_q)
        
        e_ks = _noise_poly(N, current_q=current_q)
        
        s_old_temp = PolyRing(s_old.vec, current_q)

        b_ks = -a_ks * self.secret_key + s_old_temp + e_ks
        return a_ks, b_ks