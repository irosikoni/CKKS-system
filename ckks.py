import numpy as np
import random 
from typing import Optional
from PolyRing import PolyRing 

# --- Globalne funkcje pomocnicze (generowanie szumu) ---
def _small_random_poly(n, bound=1, current_q=None): # BOUND=1 jest ok dla s
    if current_q is None: current_q = PolyRing.q 
    coeffs = np.random.randint(-bound, bound + 1, size=n).tolist()
    return PolyRing(coeffs, current_q)

# ckks.py, funkcja _noise_poly
def _noise_poly(n, bound=1, current_q=None): # UPEWNIJ SIĘ, ŻE TO JEST bound=1 (lub testowo 0)
    if current_q is None: current_q = PolyRing.q 
    coeffs = np.random.randint(-bound, bound + 1, size=n).tolist()
    return PolyRing(coeffs, current_q)

# --- Kodowanie i Dekodowanie ---
def encode(z_vec, delta, current_q):
    """
    Encode a complex vector into a polynomial.
    
    Args:
        z_vec: Complex vector to encode
        delta: Scale factor
        current_q: Current modulus
        
    Returns:
        PolyRing: Encoded polynomial
    """
    # Scale the input vector
    scaled_vec = z_vec * delta
    
    # Create polynomial
    poly = PolyRing.from_complex_vector(scaled_vec, current_q)
    poly._scale_factor = delta
    return poly

def decode(poly, delta):
    """
    Decode a polynomial back to a complex vector.
    
    Args:
        poly: PolyRing polynomial to decode
        delta: Scale factor
        
    Returns:
        numpy.ndarray: Decoded complex vector
    """
    # Get complex vector
    result = PolyRing.to_complex_vector(poly)
    
    # Rescale
    if delta > 0:
        result = result / delta
    
    return result


# --- Klasa CKKSContext ---
class CKKSContext:
    def __init__(self, N: int, q_sizes: list[int], delta_bits: int):
        # Verify N matches PolyRing.N instead of trying to assign it
        if N != PolyRing.N:
            raise ValueError(f"N must be {PolyRing.N}")
        self.N = N
        PolyRing.set_polynomial_modulus(N)

        self.q_sizes = q_sizes
        self.delta_bits = delta_bits
        self.global_delta = 2.0**delta_bits  # Changed to float for better precision

        self.primes = [] 
        self.q_chain = [] 
        
        current_product = 1

        _known_primes = {
            60: 2**60 - 93,
            40: 2**40 - 87,
            30: 2**30 - 35,
            29: 2**29 - 3
        }

        for bits in q_sizes:
            if bits in _known_primes:
                prime_candidate = _known_primes[bits]
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
    """A class representing an encrypted polynomial in the CKKS scheme."""
    
    def __init__(self, c0, c1, context, delta=None):
        """Initialize a ciphertext with two polynomials."""
        self.c0 = c0  # PolyRing
        self.c1 = c1  # PolyRing
        self.context = context
        self.delta = delta if delta is not None else context.global_delta
        self._d2 = None  # For storing the quadratic term in multiplication
        
    @staticmethod
    def encrypt_static(plain, public_key, context):
        """Encrypt a plaintext polynomial using the public key."""
        current_q = context.current_q
        
        # Generate random small polynomials with the correct modulus
        u = PolyRing.random_small(current_q)
        e1 = PolyRing.random_small(current_q)
        e2 = PolyRing.random_small(current_q)
        
        # Ensure all polynomials have the same modulus
        if plain._current_q != current_q:
            plain = plain.mod_switch_to(current_q)
        if public_key.p0._current_q != current_q:
            public_key.p0 = public_key.p0.mod_switch_to(current_q)
        if public_key.p1._current_q != current_q:
            public_key.p1 = public_key.p1.mod_switch_to(current_q)
        
        # Compute ciphertext components with modulus matching
        c0 = (public_key.p0 * u).mod_switch_to(current_q)
        c0 = (c0 + e1).mod_switch_to(current_q)
        c0 = (c0 + plain).mod_switch_to(current_q)
        
        c1 = (public_key.p1 * u).mod_switch_to(current_q)
        c1 = (c1 + e2).mod_switch_to(current_q)
        
        return Ciphertext(c0, c1, context, delta=plain._scale_factor if hasattr(plain, '_scale_factor') else None)
    
    def decrypt(self, secret_key):
        """Decrypt the ciphertext using the secret key."""
        current_q = self.context.current_q
        
        # Ensure all polynomials have the same modulus
        if self.c0._current_q != current_q:
            self.c0 = self.c0.mod_switch_to(current_q)
        if self.c1._current_q != current_q:
            self.c1 = self.c1.mod_switch_to(current_q)
        if secret_key.s._current_q != current_q:
            secret_key.s = secret_key.s.mod_switch_to(current_q)
        
        # Compute m + e = c0 + c1 * s with modulus matching
        temp = (self.c1 * secret_key.s).mod_switch_to(current_q)
        result = (self.c0 + temp).mod_switch_to(current_q)
        result._scale_factor = self.delta
        return result
    
    def __add__(self, other):
        """Add two ciphertexts homomorphically."""
        if not isinstance(other, Ciphertext):
            raise TypeError("Can only add two ciphertexts")
            
        current_q = self.context.current_q
        
        # Ensure all polynomials have the same modulus
        c0_self = self.c0.mod_switch_to(current_q) if self.c0._current_q != current_q else self.c0
        c1_self = self.c1.mod_switch_to(current_q) if self.c1._current_q != current_q else self.c1
        c0_other = other.c0.mod_switch_to(current_q) if other.c0._current_q != current_q else other.c0
        c1_other = other.c1.mod_switch_to(current_q) if other.c1._current_q != current_q else other.c1
        
        # Add corresponding polynomials with modulus matching
        c0 = (c0_self + c0_other).mod_switch_to(current_q)
        c1 = (c1_self + c1_other).mod_switch_to(current_q)
        
        # The scale factor should be the same for addition
        assert abs(self.delta - other.delta) < 1e-6, "Scales must match for addition"
        return Ciphertext(c0, c1, self.context, delta=self.delta)
    
    def __mul__(self, other):
        """Multiply two ciphertexts homomorphically."""
        if not isinstance(other, Ciphertext):
            raise TypeError("Can only multiply two ciphertexts")
            
        current_q = self.context.current_q
        
        # Ensure all polynomials have the same modulus
        c0_self = self.c0.mod_switch_to(current_q) if self.c0._current_q != current_q else self.c0
        c1_self = self.c1.mod_switch_to(current_q) if self.c1._current_q != current_q else self.c1
        c0_other = other.c0.mod_switch_to(current_q) if other.c0._current_q != current_q else other.c0
        c1_other = other.c1.mod_switch_to(current_q) if other.c1._current_q != current_q else other.c1
        
        # Compute all cross terms with modulus matching
        d0 = (c0_self * c0_other).mod_switch_to(current_q)
        d1 = ((c0_self * c1_other).mod_switch_to(current_q) + 
              (c1_self * c0_other).mod_switch_to(current_q)).mod_switch_to(current_q)
        d2 = (c1_self * c1_other).mod_switch_to(current_q)
        
        # The scale factor multiplies in multiplication
        new_delta = self.delta * other.delta
        
        result = Ciphertext(d0, d1, self.context, delta=new_delta)
        result._d2 = d2  # Store for later relinearization
        return result
    
    def add(self, other):
        """Legacy method for addition, use + operator instead."""
        return self.__add__(other)
    
    def multiply(self, other):
        """Legacy method for multiplication, use * operator instead."""
        return self.__mul__(other)

    def relinearize(self):
        """
        Performs relinearization and adjusts scale.
        The scale is divided by the dropped prime.
        """
        if self._d2 is None:
            raise ValueError("No c2 (d2) component - perform multiply() first.")

        current_mod_idx = self.context.current_modulus_idx
        if current_mod_idx == 0:
            raise ValueError("Cannot perform relinearization: reached last modulus in chain.")

        next_mod_idx = current_mod_idx - 1
        Q_next = self.context.q_chain[next_mod_idx]
        P_drop = self.context.primes[current_mod_idx]

        # Create new context with updated parameters
        new_context = CKKSContext(self.context.N, self.context.q_sizes, self.context.delta_bits)
        new_context.primes = self.context.primes
        new_context.q_chain = self.context.q_chain
        new_context.current_modulus_idx = next_mod_idx
        new_context.current_q = Q_next
        new_context.keygen = self.context.keygen
        PolyRing.q = Q_next

        # Get relinearization key for current level
        relin_key_for_stage = self.context.keygen.relin_key[current_mod_idx - 1]
        A_r, B_r = relin_key_for_stage

        # Scale down all components by P_drop
        d2_scaled = Ciphertext._rescale_poly_by_factor(self._d2, P_drop, Q_next)
        B_r_scaled = Ciphertext._rescale_poly_by_factor(B_r, P_drop, Q_next)
        A_r_scaled = Ciphertext._rescale_poly_by_factor(A_r, P_drop, Q_next)
        c0_scaled = Ciphertext._rescale_poly_by_factor(self.c0, P_drop, Q_next)
        c1_scaled = Ciphertext._rescale_poly_by_factor(self.c1, P_drop, Q_next)

        # Relinearization step with all components at Q_next
        c0_final = c0_scaled + d2_scaled * B_r_scaled
        c1_final = c1_scaled + d2_scaled * A_r_scaled

        # The scale is divided by P_drop
        new_delta = self.delta / P_drop

        return Ciphertext(c0_final, c1_final, new_context, new_delta)

    def rescale(self):
        """
        Performs rescaling and adjusts scale.
        The scale is divided by the dropped prime.
        """
        current_mod_idx = self.context.current_modulus_idx
        if current_mod_idx == 0:
            raise ValueError("Cannot perform rescaling: reached last modulus in chain.")

        next_mod_idx = current_mod_idx - 1
        Q_next = self.context.q_chain[next_mod_idx]
        P_drop = self.context.primes[current_mod_idx]

        # Create new context with updated parameters
        new_context = CKKSContext(self.context.N, self.context.q_sizes, self.context.delta_bits)
        new_context.primes = self.context.primes
        new_context.q_chain = self.context.q_chain
        new_context.current_modulus_idx = next_mod_idx
        new_context.current_q = Q_next
        new_context.keygen = self.context.keygen
        PolyRing.q = Q_next

        # Scale down ciphertext components
        c0_rescaled = Ciphertext._rescale_poly_by_factor(self.c0, P_drop, Q_next)
        c1_rescaled = Ciphertext._rescale_poly_by_factor(self.c1, P_drop, Q_next)

        # The scale is divided by P_drop
        new_delta = self.delta / P_drop

        return Ciphertext(c0_rescaled, c1_rescaled, new_context, new_delta)

    def key_switch(self, key_switch_key):
        """
        Wykonuje operację przełączania kluczy szyfrogramu ze starego klucza tajnego na nowy.
        """
        a_ks, b_ks = key_switch_key
        current_q = self.context.current_q

        # Scale down coefficients before key switching to reduce noise
        scale_factor = 2**4  # Reduced from 2**8 to control noise better

        # Scale down key switching keys
        a_ks_scaled = Ciphertext._rescale_poly_by_factor(a_ks, scale_factor, current_q)
        b_ks_scaled = Ciphertext._rescale_poly_by_factor(b_ks, scale_factor, current_q)

        # Scale down ciphertext
        c0_scaled = Ciphertext._rescale_poly_by_factor(self.c0, scale_factor, current_q)
        c1_scaled = Ciphertext._rescale_poly_by_factor(self.c1, scale_factor, current_q)

        # Key switching with scaled components
        c0_new = c0_scaled + c1_scaled * b_ks_scaled
        c1_new = c1_scaled * a_ks_scaled

        # Scale back up
        c0_final = c0_new * scale_factor
        c1_final = c1_new * scale_factor

        return Ciphertext(c0_final, c1_final, self.context, self.delta)

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

        return Ciphertext(c0, c1, context, context.global_delta)

    @staticmethod
    def decrypt_static(ciphertext, keygen):
        # Dopasuj secret_key do q ciphertext.c1
        if ciphertext.c1._current_q != keygen.secret_key._current_q:
            secret_key_scaled = ciphertext.rescale_secret_key(keygen.secret_key, ciphertext.c1._current_q)
        else:
            secret_key_scaled = keygen.secret_key

        # Dekodowanie: c0 + c1 * secret_key_scaled
        decrypted_poly = ciphertext.c0 + ciphertext.c1 * secret_key_scaled
        return decrypted_poly

    @staticmethod
    def rescale_secret_key(secret_key: PolyRing, target_q: int) -> PolyRing:
        """
        Skalowanie secret_key z jego aktualnego q do target_q.
        Przyjmujemy, że target_q < secret_key._current_q (czyli q się zmniejsza w łańcuchu).
        """
        current_q = secret_key._current_q
        if target_q == current_q:
            return secret_key  # nic nie zmieniamy

        factor = current_q / target_q
        if factor <= 0:
            raise ValueError("Niepoprawny współczynnik skalowania")

        # Zamieniamy współczynniki na float i dzielimy przez factor
        scaled_coeffs_float = np.array([float(x) for x in secret_key.vec], dtype=np.float64)
        scaled_coeffs_float = np.round(scaled_coeffs_float / factor)

        # Modular reduction w nowym q
        reduced = np.array([(int(x) % target_q + target_q) % target_q for x in scaled_coeffs_float], dtype=object)

        return PolyRing(reduced, target_q)

    @staticmethod
    def _rescale_poly_by_factor(poly: PolyRing, factor: int, new_q: int) -> PolyRing:
        """Helper method to rescale polynomial coefficients."""
        # Convert to centered representation
        coeffs = np.array([float(x) for x in poly.vec], dtype=np.float64)
        coeffs = np.where(coeffs > poly._current_q/2, coeffs - poly._current_q, coeffs)
        
        # Scale down by factor
        coeffs_scaled = coeffs / factor
        
        # Round to nearest integer
        coeffs_rounded = np.round(coeffs_scaled)
        
        # Center coefficients around zero
        coeffs_centered = np.where(coeffs_rounded > new_q/2, coeffs_rounded - new_q, coeffs_rounded)
        
        # Convert to integers modulo new_q
        coeffs_int = np.array([(int(x) % new_q + new_q) % new_q for x in coeffs_centered], dtype=object)
        
        return PolyRing(coeffs_int, new_q)


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