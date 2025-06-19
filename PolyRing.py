import numpy as np

class PolyRing:
    # q powinno być liczbą pierwszą lub iloczynem dużych liczb pierwszych,
    # w praktyce jest elementem łańcucha modułów.
    # Dla testów uproszczonych, używamy dużej liczby.
    q = 2**60 - 1
    # f = x^N + 1, gdzie N jest potęgą dwójki
    # Dla N=8: [1, 0, 0, 0, 0, 0, 0, 0, 1]
    f = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)

    def __init__(self, vec):
        N = len(self.f) - 1
        vec = np.array(vec, dtype=np.int64)
        
        # Jeśli wektor jest dłuższy niż N, zredukuj go modulo f
        if len(vec) > N:
            self.vec = self._reduce_mod_f(vec)
        else:
            # Dopasuj rozmiar do N, jeśli krótszy (wypełnij zerami)
            self.vec = np.resize(vec, N)
        
        # Zapewnij, że wszystkie współczynniki są w zakresie [0, q-1]
        self.vec = (self.vec % self.q + self.q) % self.q

    def _reduce_mod_f(self, poly_coeffs):
        """
        Redukuje wielomian modulo x^N + 1.
        Wykorzystuje fakt, że x^N = -1 w tym pierścieniu.
        """
        N = len(self.f) - 1
        poly_coeffs_np = np.array(poly_coeffs, dtype=np.int64)

        reduced_coeffs = np.zeros(N, dtype=np.int64)

        for i in range(len(poly_coeffs_np)):
            target_idx = i % N
            # Znak zależy od tego, ile razy i jest większe od N
            # np. x^N -> -1, x^(N+1) -> -x, x^(2N) -> 1
            sign = 1 if (i // N) % 2 == 0 else -1

            coeff_val = (poly_coeffs_np[i] % self.q + self.q) % self.q

            # Dodawanie z uwzględnieniem znaku i redukcja modulo q
            intermediate_sum = reduced_coeffs[target_idx] + sign * coeff_val
            reduced_coeffs[target_idx] = (intermediate_sum % self.q + self.q) % self.q
            
        return reduced_coeffs

    def __repr__(self):
        return str(self.vec)

    def __add__(self, other):
        """Homomorficzne dodawanie wielomianów."""
        return PolyRing((self.vec + other.vec) % self.q)

    def __sub__(self, other):
        """Homomorficzne odejmowanie wielomianów."""
        return PolyRing((self.vec - other.vec) % self.q)

    def __neg__(self):
        """Negacja wielomianu."""
        return PolyRing((-self.vec) % self.q)

    def __mul__(self, other):
        """Homomorficzne mnożenie wielomianów lub mnożenie przez skalar."""
        if isinstance(other, PolyRing):
            # Mnożenie wielomianów za pomocą np.polymul
            full = np.polymul(self.vec, other.vec)
            # Redukcja wyniku modulo x^N + 1
            reduced = self._reduce_mod_f(full)
            return PolyRing(reduced)
        elif isinstance(other, (int, np.integer)):
            # Mnożenie przez skalar
            return PolyRing((self.vec * other) % self.q)
        else:
            raise TypeError("Nieprawidłowy typ mnożenia")

    def __rmul__(self, other):
        """Obsługa mnożenia, gdy PolyRing jest po prawej stronie."""
        return self * other

    @classmethod
    def from_complex_vector(cls, z_vec, delta):
        """
        Koduje wektor liczb zespolonych (slotów) do wielomianu PolyRing.
        Wykorzystuje kanoniczne osadzenie i niestandardowe IFFT dla X^N+1.
        """
        N = len(cls.f) - 1
        num_slots = N // 2

        if len(z_vec) != num_slots:
            raise ValueError(f"Długość z_vec musi wynosić {num_slots} dla N={N}")

        # Rozszerzenie wektora slotów o symetrię Hermitowską
        # (z_0, ..., z_{N/2-1}, conj(z_{N/2-1}), ..., conj(z_0))
        v_full = np.concatenate([z_vec, np.conj(z_vec[::-1])])

        # Definicja N-tych pierwiastków jedności dla X^N + 1 (odd-frequency roots)
        # xi_k = exp(i * pi * (2k+1) / N)
        xi_roots = np.exp(1j * np.pi * (2 * np.arange(N) + 1) / N)

        # Implementacja odwrotnej transformaty DFT (interpolacji)
        # p_j = (1/N) * sum_{k=0}^{N-1} v_k * xi_k^(-j)
        coeffs_complex = np.zeros(N, dtype=np.complex128)
        for j in range(N):
            sum_val = 0 + 0j # Inicjalizacja sumy jako liczby zespolonej
            for k in range(N):
                sum_val += v_full[k] * (xi_roots[k] ** (-j))
            coeffs_complex[j] = sum_val / N # Normalizacja przez N

        # Skalowanie i zaokrąglanie do współczynników całkowitych modulo q
        scaled_coeffs_real = coeffs_complex.real * delta
        poly_coeffs = (np.round(scaled_coeffs_real) % cls.q + cls.q) % cls.q

        return cls(poly_coeffs.astype(np.int64))

    def to_complex_vector(self, delta):
        """
        Dekoduje wielomian PolyRing z powrotem do wektora liczb zespolonych (slotów).
        Wykorzystuje kanoniczne osadzenie i niestandardowe DFT dla X^N+1.
        """
        N = len(self.f) - 1
        num_slots = N // 2

        # Centrowanie współczynników wokół zera przed transformacją
        # (wartości > q/2 są traktowane jako ujemne)
        coeffs_centered = self.vec.astype(np.float64)
        coeffs_centered = np.where(coeffs_centered > self.q / 2, coeffs_centered - self.q, coeffs_centered)

        # Definicja N-tych pierwiastków jedności dla X^N + 1 (odd-frequency roots)
        # xi_k = exp(i * pi * (2k+1) / N)
        xi_roots = np.exp(1j * np.pi * (2 * np.arange(N) + 1) / N)

        # Implementacja transformaty DFT (ewaluacji)
        # v_k = sum_{j=0}^{N-1} p_j * xi_k^j
        slots = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            sum_val = 0 + 0j # Inicjalizacja sumy jako liczby zespolonej
            for j in range(N):
                sum_val += coeffs_centered[j] * (xi_roots[k] ** j)
            slots[k] = sum_val
        
        # Zwracamy tylko pierwszą połowę slotów i dzielimy przez delta
        return slots[:num_slots] / delta