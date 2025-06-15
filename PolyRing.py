import numpy as np

class PolyRing():
    q = 2**60 - 1
    f = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])

    def __init__(self, vec):
        N = len(self.f) - 1
        vec = np.array(vec)
        self.vec = (np.resize(vec, N) % self.q).astype(np.int64)

    def __repr__(self):
        return str(self.vec)

    def __add__(self, other):
        if isinstance(other, PolyRing):
            return PolyRing(np.polyadd(self.vec, other.vec))
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'PolyRing' and '{type(other).__name__}'")

    def __sub__(self, other):
        if isinstance(other, PolyRing):
            return PolyRing(np.polysub(self.vec, other.vec))
        else:
            raise TypeError(f"unsupported operand type(s) for -: 'PolyRing' and '{type(other).__name__}'")

    def __mul__(self, other):
        if isinstance(other, PolyRing):
            N = len(self.f) - 1
            prod = np.polymul(self.vec, other.vec)
            reduced_prod_coeffs = np.zeros(N, dtype=prod.dtype)
            for i in range(N):
                reduced_prod_coeffs[i] = prod[i]
            for i in range(N, len(prod)):
                exponent = i - N
                reduced_prod_coeffs[exponent] -= prod[i]
            return PolyRing(reduced_prod_coeffs)
        elif isinstance(other, int):
            return PolyRing(other * self.vec)
        else:
            raise TypeError(f"unsupported operand type(s) for *: 'PolyRing' and '{type(other).__name__}'")

    def __rmul__(self, other):
        if isinstance(other, int):
            return PolyRing(other * self.vec)
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(other).__name__}' and 'PolyRing'")


    @classmethod
    def from_complex_vector(cls, z_vec, delta):
        N = len(cls.f) - 1
        num_slots = N // 2

        if len(z_vec) != num_slots:
            raise ValueError(f"Length of z_vec must be {num_slots} for N={N}")

        v_prime_scaled = np.zeros(N, dtype=np.complex128)
        v_prime_scaled[:num_slots] = z_vec
        v_prime_scaled[num_slots:] = np.conj(z_vec[::-1])


        v_prime_scaled *= delta

        roots_of_unity_odd = np.exp(1j * np.pi * (2 * np.arange(N) + 1) / N)

        coeffs_complex = np.zeros(N, dtype=np.complex128)
        for j in range(N):
            sum_val = 0.0
            for k in range(N):
                term = v_prime_scaled[k] * (roots_of_unity_odd[k] ** (-j))
                sum_val += term
            coeffs_complex[j] = sum_val / N

        rounded_coeffs = np.round(coeffs_complex.real)

        reduced_coeffs = (rounded_coeffs % cls.q).astype(np.int64)
        return cls(reduced_coeffs)

    def to_complex_vector(self, delta):
            N = len(self.f) - 1
            num_slots = N // 2

            coeffs_raw = self.vec.astype(np.float64)
            coeffs = np.where(coeffs_raw > self.q / 2, coeffs_raw - self.q, coeffs_raw)


            roots_of_unity_odd = np.exp(1j * np.pi * (2 * np.arange(N) + 1) / N)

            v_prime_scaled_decoded = np.zeros(N, dtype=np.complex128)
            for k in range(N):
                sum_val = 0.0
                for j in range(N):
                    term = coeffs[j] * (roots_of_unity_odd[k] ** j)
                    sum_val += term
                v_prime_scaled_decoded[k] = sum_val

            z_approx_full = v_prime_scaled_decoded / delta

            return z_approx_full[:num_slots]