import numpy as np

class PolyRing:
    """
    Reprezentuje wielomiany w pierścieniu Z_Q[X]/(X^N+1).
    Moduł Q jest dynamiczny i będzie zarządzany przez modulus chain.
    """
    
    # N (poly_modulus_degree) musi być potęgą dwójki i znacznie większe niż 8.
    # Zacznijmy od 1024 dla rozsądnych czasów wykonania i lepszych wyników.
    # W prezentacji TenSEAL jest 8192, co jest dobre dla produkcji.
    N = 1024 
    f = np.array([1] + [0]*(N-1) + [1], dtype=np.int64) # X^N + 1

    # `q` nie będzie stałe w klasie, będzie przekazywane jako parametr w zależności od kontekstu.
    # Zostawiamy je jako klasowe pole dla kompatybilności z __init__ i starymi testami,
    # ale w rzeczywistości będzie dynamicznie zmieniane w Ciphertext/KeyGenerator.
    q = None # Będzie ustawiane dynamicznie przez Context/Ciphertext

    def __init__(self, vec, current_q=None):
        """
        Inicjalizuje wielomian.
        vec: Lista/array współczynników.
        current_q: Aktualny moduł Q dla tego wielomianu.
                   Jeśli None, używa PolyRing.q (dla wstecznej kompatybilności lub gdy q jest stałe).
        """
        if current_q is None and PolyRing.q is None:
            raise ValueError("Modulus q must be set either globally (PolyRing.q) or locally (current_q).")
        self._current_q = current_q if current_q is not None else PolyRing.q

        vec = np.array(vec, dtype=np.int64)
        
        # Redukcja wielomianu modulo X^N + 1 podczas inicjalizacji, jeśli jest zbyt długi.
        if len(vec) >= self.N: # Używamy >= N, bo np.polymul może dać stopień 2N-2
            self.vec = self._reduce_mod_f(vec)
        else:
            self.vec = np.resize(vec, self.N) # Dopasuj rozmiar do N
        
        # Zapewnij, że wszystkie współczynniki są w zakresie [0, self._current_q - 1]
        self.vec = (self.vec % self._current_q + self._current_q) % self.q

    def _reduce_mod_f(self, poly_coeffs):
        """
        Redukuje wielomian modulo x^N + 1.
        Wykorzystuje fakt, że x^N = -1 w tym pierścieniu.
        """
        N = self.N # Używamy N z instancji klasy
        current_q = self._current_q

        poly_coeffs_np = np.array(poly_coeffs, dtype=np.int64)
        reduced_coeffs = np.zeros(N, dtype=np.int64)

        for i in range(len(poly_coeffs_np)):
            target_idx = i % N
            sign = 1 if (i // N) % 2 == 0 else -1

            coeff_val = (poly_coeffs_np[i] % current_q + current_q) % current_q

            intermediate_sum = reduced_coeffs[target_idx] + sign * coeff_val
            reduced_coeffs[target_idx] = (intermediate_sum % current_q + current_q) % current_q
            
        return reduced_coeffs

    def __repr__(self):
        return str(self.vec)

    def __add__(self, other):
        # Sprawdź spójność modułów Q
        assert self._current_q == other._current_q, "Moduli Q must match for addition"
        return PolyRing((self.vec + other.vec) % self._current_q, self._current_q)

    def __sub__(self, other):
        # Sprawdź spójność modułów Q
        assert self._current_q == other._current_q, "Moduli Q must match for subtraction"
        return PolyRing((self.vec - other.vec) % self._current_q, self._current_q)

    def __neg__(self):
        return PolyRing((-self.vec) % self._current_q, self._current_q)

    def __mul__(self, other):
        # Mnożenie może zwiększyć stopień wielomianu i wymagac redukcji
        if isinstance(other, PolyRing):
            # Sprawdź spójność modułów Q
            assert self._current_q == other._current_q, "Moduli Q must match for multiplication"
            
            full = np.polymul(self.vec, other.vec)
            reduced = self._reduce_mod_f(full)
            return PolyRing(reduced, self._current_q)
        elif isinstance(other, (int, np.integer)):
            return PolyRing((self.vec * other) % self._current_q, self._current_q)
        else:
            raise TypeError("Nieprawidłowy typ mnożenia")

    def __rmul__(self, other):
        return self * other

    # --- Zmodyfikowane funkcje kodowania/dekodowania ---
    # Będą używać N z PolyRing.N

    @classmethod
    def from_complex_vector(cls, z_vec, delta, current_q):
        """
        Koduje wektor liczb zespolonych (slotów) do wielomianu PolyRing.
        Wykorzystuje kanoniczne osadzenie i niestandardowe IFFT dla X^N+1.
        """
        num_slots = cls.N // 2 # Używamy cls.N

        if len(z_vec) != num_slots:
            raise ValueError(f"Długość z_vec musi wynosić {num_slots} dla N={cls.N}")

        v_full = np.concatenate([z_vec, np.conj(z_vec[::-1])])

        xi_roots = np.exp(1j * np.pi * (2 * np.arange(cls.N) + 1) / cls.N) # Używamy cls.N

        coeffs_complex = np.zeros(cls.N, dtype=np.complex128)
        for j in range(cls.N):
            sum_val = 0 + 0j
            for k in range(cls.N):
                sum_val += v_full[k] * (xi_roots[k] ** (-j))
            coeffs_complex[j] = sum_val / cls.N

        scaled_coeffs_real = coeffs_complex.real * delta
        poly_coeffs = (np.round(scaled_coeffs_real) % current_q + current_q) % current_q

        return cls(poly_coeffs.astype(np.int64), current_q)

    def to_complex_vector(self, delta):
        """
        Dekoduje wielomian PolyRing z powrotem do wektora liczb zespolonych (slotów).
        Wykorzystuje kanoniczne osadzenie i niestandardowe DFT dla X^N+1.
        """
        num_slots = self.N // 2

        coeffs_centered = self.vec.astype(np.float64)
        coeffs_centered = np.where(coeffs_centered > self._current_q / 2, coeffs_centered - self._current_q, coeffs_centered)

        xi_roots = np.exp(1j * np.pi * (2 * np.arange(self.N) + 1) / self.N)

        slots = np.zeros(self.N, dtype=np.complex128)
        for k in range(self.N):
            sum_val = 0 + 0j
            for j in range(self.N):
                sum_val += coeffs_centered[j] * (xi_roots[k] ** j)
            slots[k] = sum_val
        
        return slots[:num_slots] / delta