import numpy as np
import random 

class PolyRing:
    """
    Reprezentuje wielomiany w pierścieniu Z_Q[X]/(X^N+1).
    Moduł Q jest dynamiczny i będzie zarządzany przez modulus chain.
    """
    
    # N zostanie nadpisane przez CKKSContext, ale inicjujemy dla spójności
    N = 1024 
    # f = X^N + 1. Będzie aktualizowane w CKKSContext, gdy N zostanie ustawione.
    f = np.array([1] + [0]*(N-1) + [1], dtype=object) 

    # `q` będzie globalnym modułem kontekstu, ustawianym przez CKKSContext.
    q = None 

    def __init__(self, vec, current_q=None):
        """
        Inicjalizuje wielomian.
        vec: Lista/array współczynników.
        current_q: Aktualny moduł Q dla tego wielomianu.
                   Jeśli None, używa PolyRing.q (dla wstecznej kompatybilności lub gdy q jest stałe).
        """
        if current_q is None and PolyRing.q is None:
            raise ValueError("Moduł q musi być ustawiony globalnie (PolyRing.q) lub lokalnie (current_q).")
        self._current_q = current_q if current_q is not None else PolyRing.q

        vec = np.array(vec, dtype=object) 
        
        # Upewniamy się, że PolyRing.f.size jest poprawne dla bieżącego PolyRing.N
        # To jest kluczowe, bo self.f jest atrybutem klasy i może być zmienione
        # przez CKKSContext. Niestety, numpy.array domyślnie tworzy raz,
        # więc musimy go zaktualizować, jeśli N się zmieni.
        if len(PolyRing.f) - 1 != self.N:
            PolyRing.f = np.array([1] + [0]*(self.N-1) + [1], dtype=object)

        # Redukcja wielomianu modulo X^N + 1 podczas inicjalizacji, jeśli jest zbyt długi.
        if len(vec) >= self.N:
            self.vec = self._reduce_mod_f(vec)
        else:
            self.vec = np.resize(vec, self.N) 
        
        self.vec = np.array([(int(x) % self._current_q + self._current_q) % self._current_q for x in self.vec], dtype=object)


    def _reduce_mod_f(self, poly_coeffs):
        """
        Redukuje wielomian modulo x^N + 1.
        Wykorzystuje fakt, że x^N = -1 w tym pierścieniu.
        Obsługuje duże liczby całkowite używając dtype=object.
        """
        N = self.N 
        current_q = self._current_q

        poly_coeffs_np = np.array(poly_coeffs, dtype=object) 
        reduced_coeffs = np.zeros(N, dtype=object) 

        for i in range(len(poly_coeffs_np)):
            target_idx = i % N
            sign = 1 if (i // N) % 2 == 0 else -1

            coeff_val = (int(poly_coeffs_np[i]) % current_q + current_q) % current_q

            intermediate_sum = int(reduced_coeffs[target_idx]) + sign * coeff_val
            reduced_coeffs[target_idx] = (intermediate_sum % current_q + current_q) % current_q
            
        return reduced_coeffs

    def __repr__(self):
        return str(self.vec)

    def __add__(self, other):
        """Homomorficzne dodawanie wielomianów."""
        assert self._current_q == other._current_q, "Moduli Q must match for addition"
        result_vec = np.array([(int(x) + int(y)) % self._current_q for x, y in zip(self.vec, other.vec)], dtype=object)
        return PolyRing(result_vec, self._current_q)

    def __sub__(self, other):
        """Homomorficzne odejmowanie wielomianów."""
        assert self._current_q == other._current_q, "Moduli Q must match for subtraction"
        result_vec = np.array([(int(x) - int(y)) % self._current_q for x, y in zip(self.vec, other.vec)], dtype=object)
        return PolyRing(result_vec, self._current_q)

    def __neg__(self):
        """Negacja wielomianu."""
        result_vec = np.array([(-int(x)) % self._current_q for x in self.vec], dtype=object)
        return PolyRing(result_vec, self._current_q)

    def __mul__(self, other):
        """Homomorficzne mnożenie wielomianów lub mnożenie przez skalar."""
        if isinstance(other, PolyRing):
            assert self._current_q == other._current_q, "Moduli Q must match for multiplication"
            
            # Ręczne mnożenie wielomianów
            coeffs1_list = [int(x) for x in self.vec]
            coeffs2_list = [int(x) for x in other.vec]

            deg1 = len(coeffs1_list) - 1
            deg2 = len(coeffs2_list) - 1
            result_deg = deg1 + deg2
            
            full_coeffs_list = [0] * (result_deg + 1)

            for i in range(deg1 + 1):
                for j in range(deg2 + 1):
                    full_coeffs_list[i + j] += coeffs1_list[i] * coeffs2_list[j]
            
            full = np.array(full_coeffs_list, dtype=object)
            
            reduced = self._reduce_mod_f(full)
            return PolyRing(reduced, self._current_q)
        elif isinstance(other, (int, np.integer)):
            result_vec = np.array([(int(x) * other) % self._current_q for x in self.vec], dtype=object)
            return PolyRing(result_vec, self._current_q)
        else:
            raise TypeError("Nieprawidłowy typ mnożenia")

    def __rmul__(self, other):
        return self * other

    @classmethod
    def from_complex_vector(cls, z_vec, delta, current_q):
        """
        Koduje wektor liczb zespolonych (slotów) do wielomianu PolyRing.
        Wykorzystuje kanoniczne osadzenie i niestandardowe IFFT dla X^N+1.
        """
        N = cls.N 
        num_slots = N // 2

        if len(z_vec) != num_slots:
            raise ValueError(f"Długość z_vec musi wynosić {num_slots} dla N={N}")

        v_full = np.concatenate([z_vec, np.conj(z_vec[::-1])])

        xi_roots = np.exp(1j * np.pi * (2 * np.arange(N) + 1) / N)

        coeffs_complex = np.zeros(N, dtype=np.complex128) 
        for j in range(N):
            sum_val = 0 + 0j 
            for k in range(N):
                sum_val += v_full[k] * (xi_roots[k] ** (-j))
            coeffs_complex[j] = sum_val / N 

        scaled_coeffs_real = coeffs_complex.real * delta
        
        poly_coeffs = np.array([(int(round(x)) % current_q + current_q) % current_q for x in scaled_coeffs_real], dtype=object)

        return cls(poly_coeffs, current_q)

    def to_complex_vector(self, delta):
        """
        Dekoduje wielomian PolyRing z powrotem do wektora liczb zespolonych (slotów).
        Wykorzystuje kanoniczne osadzenie i niestandardowe DFT dla X^N+1.
        """
        N = self.N
        num_slots = N // 2

        coeffs_centered_float = np.array([int(x) for x in self.vec], dtype=np.float64) 
        coeffs_centered_float = np.where(coeffs_centered_float > self._current_q / 2, coeffs_centered_float - self._current_q, coeffs_centered_float)

        xi_roots = np.exp(1j * np.pi * (2 * np.arange(N) + 1) / N)

        slots = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            sum_val = 0 + 0j 
            for j in range(N):
                sum_val += coeffs_centered_float[j] * (xi_roots[k] ** j)
            slots[k] = sum_val
        
        return slots[:num_slots] / delta