import numpy as np
import random 
from numpy.typing import NDArray
from typing import Optional

class PolyRing:
    """
    A class representing polynomials in a ring Z_q[X]/(X^N + 1).
    """
    N = 4  # Default ring degree, must be a power of 2
    q = None  # Current modulus
    f: Optional[NDArray[np.object_]] = None  # Polynomial modulus X^N + 1
    
    @classmethod
    def set_polynomial_modulus(cls, N: int) -> None:
        """Set the polynomial modulus X^N + 1."""
        cls.f = np.array([1] + [0]*(N-1) + [1], dtype=object)

    @classmethod
    def set_ring_degree(cls, value):
        """Set the ring degree N."""
        if value & (value - 1) != 0:  # Check if power of 2
            raise ValueError("Ring degree must be a power of 2")
        cls.N = value
    
    def __init__(self, coefficients, current_q):
        """
        Initialize a polynomial with given coefficients modulo current_q.
        """
        self.vec = np.array(coefficients, dtype=object)
        if len(self.vec) > self.N:
            raise ValueError(f"Polynomial degree must be less than {self.N}")
            
        # Pad with zeros if needed
        if len(self.vec) < self.N:
            self.vec = np.pad(self.vec, (0, self.N - len(self.vec)), constant_values=0)
            
        self._current_q = current_q
        self._scale_factor = 1  # Default scale factor
        
        # Ensure all coefficients are properly reduced modulo current_q
        self.vec = np.array([(int(x) % self._current_q + self._current_q) % self._current_q for x in self.vec], dtype=object)

    @classmethod
    def random_small(cls, current_q):
        """
        Generate a polynomial with small random coefficients.
        Used for error terms in encryption.
        """
        # Generate coefficients uniformly from {-1, 0, 1}
        coeffs = np.random.randint(-1, 2, cls.N)
        return cls(coeffs, current_q)

    def mod_switch_to(self, new_q):
        """
        Switch the modulus of the polynomial from current_q to new_q.
        Preserves the scale factor.
        """
        # Convert coefficients to centered representation
        coeffs = np.array([float(x) for x in self.vec], dtype=np.float64)
        coeffs = np.where(coeffs > self._current_q/2, coeffs - self._current_q, coeffs)
        
        # Scale coefficients with proper rounding
        scale = new_q / self._current_q
        new_coeffs = coeffs * scale
        
        # Round to nearest integer in centered representation
        new_coeffs = np.round(new_coeffs)
        
        # Center coefficients around zero before modulo operation
        new_coeffs = np.where(new_coeffs > new_q/2, new_coeffs - new_q, new_coeffs)
        new_coeffs = np.where(new_coeffs < -new_q/2, new_coeffs + new_q, new_coeffs)
        
        # Convert back to positive representation modulo new_q
        new_coeffs = np.array([(int(x) % new_q + new_q) % new_q for x in new_coeffs], dtype=object)
        
        # Create new polynomial with adjusted scale factor
        result = PolyRing(new_coeffs, new_q)
        if hasattr(self, '_scale_factor'):
            # Adjust scale factor to maintain the relative scale
            result._scale_factor = self._scale_factor * scale
        return result

    @classmethod
    def from_complex_vector(cls, z_vec, current_q):
        """
        Encodes a vector of complex numbers into a PolyRing polynomial.
        Uses proper scaling and rounding for better numerical stability.
        """
        num_slots = cls.N // 2

        if len(z_vec) != num_slots:
            raise ValueError(f"Length of z_vec must be {num_slots} for N={cls.N}")

        # Create conjugate-symmetric vector
        v_full = np.concatenate([z_vec, np.conj(z_vec[::-1])])
        
        # Create evaluation points using primitive 2N-th roots of unity
        k = np.arange(cls.N)
        roots = np.exp(2j * np.pi * (2*k + 1) / (2*cls.N))
        
        # Create Vandermonde matrix for the canonical embedding
        vand = np.vander(roots, cls.N, increasing=True)
        
        # Scale the input vector to use more of the available range
        # Use a very conservative scale factor for better precision in multiplication
        max_val = max(1.0, np.max(np.abs(v_full)))
        scale_factor = min(current_q // (1024 * max_val), current_q // 2048)
        v_scaled = v_full * scale_factor
        
        # Compute polynomial coefficients using least squares with better conditioning
        coeffs = np.linalg.lstsq(vand, v_scaled, rcond=1e-15)[0]
        
        # Round coefficients with proper scaling and ensure they're real
        coeffs = np.round(np.real(coeffs))
        
        # Center coefficients around zero before modulo operation
        coeffs = np.where(coeffs > current_q/2, coeffs - current_q, coeffs)
        coeffs = np.where(coeffs < -current_q/2, coeffs + current_q, coeffs)
        
        # Convert back to positive representation modulo current_q
        coeffs = np.array([(int(x) % current_q + current_q) % current_q for x in coeffs], dtype=object)
        
        result = cls(coeffs, current_q)
        result._scale_factor = float(scale_factor)  # Ensure scale_factor is float for better arithmetic
        return result

    @staticmethod
    def to_complex_vector(poly):
        """
        Decodes a PolyRing polynomial back to a vector of complex numbers.
        Uses proper scaling for better numerical stability.
        """
        N = poly.N
        num_slots = N // 2
        
        # Convert coefficients to centered representation
        coeffs = np.array([float(x) for x in poly.vec], dtype=np.float64)
        coeffs = np.where(coeffs > poly._current_q/2, coeffs - poly._current_q, coeffs)
        
        # Create evaluation points using primitive 2N-th roots of unity
        k = np.arange(N)
        roots = np.exp(2j * np.pi * (2*k + 1) / (2*N))
        
        # Create Vandermonde matrix for evaluation
        vand = np.vander(roots, N, increasing=True)
        
        # Evaluate polynomial at roots of unity
        result = vand @ coeffs
        
        # Rescale using stored scale factor
        if hasattr(poly, '_scale_factor') and poly._scale_factor > 0:
            result = result / float(poly._scale_factor)  # Ensure float division
        
        # Return first half (the rest is conjugate symmetric)
        return result[:num_slots]

    def __add__(self, other):
        """Add two polynomials."""
        if not isinstance(other, PolyRing):
            other = PolyRing(np.array([other] + [0] * (self.N - 1)), self._current_q)
            
        assert self._current_q == other._current_q, "Moduli Q must match for addition"
        
        # Handle different scale factors
        if hasattr(self, '_scale_factor') and hasattr(other, '_scale_factor'):
            if abs(self._scale_factor - other._scale_factor) > 1e-6:
                # Scale the polynomial with smaller scale factor up
                if self._scale_factor < other._scale_factor:
                    scale_ratio = other._scale_factor / self._scale_factor
                    scaled_vec = [(int(round(x * scale_ratio)) % self._current_q) for x in self.vec]
                    result_vec = [(x + y) % self._current_q for x, y in zip(scaled_vec, other.vec)]
                    result = PolyRing(result_vec, self._current_q)
                    result._scale_factor = other._scale_factor
                else:
                    scale_ratio = self._scale_factor / other._scale_factor
                    scaled_vec = [(int(round(x * scale_ratio)) % self._current_q) for x in other.vec]
                    result_vec = [(x + y) % self._current_q for x, y in zip(self.vec, scaled_vec)]
                    result = PolyRing(result_vec, self._current_q)
                    result._scale_factor = self._scale_factor
            else:
                # Same scale factor, just add
                result_vec = [(x + y) % self._current_q for x, y in zip(self.vec, other.vec)]
                result = PolyRing(result_vec, self._current_q)
                result._scale_factor = self._scale_factor
        else:
            # No scale factors, just add
            result_vec = [(x + y) % self._current_q for x, y in zip(self.vec, other.vec)]
            result = PolyRing(result_vec, self._current_q)
            
        return result

    def __mul__(self, other):
        """Multiply two polynomials."""
        if not isinstance(other, PolyRing):
            other = PolyRing(np.array([other] + [0] * (self.N - 1)), self._current_q)
            
        assert self._current_q == other._current_q, "Moduli Q must match for multiplication"
        
        # Convert to centered representation for better precision
        vec1 = np.array([float(x) for x in self.vec], dtype=np.float64)
        vec1 = np.where(vec1 > self._current_q/2, vec1 - self._current_q, vec1)
        
        vec2 = np.array([float(x) for x in other.vec], dtype=np.float64)
        vec2 = np.where(vec2 > self._current_q/2, vec2 - self._current_q, vec2)
        
        # Multiply coefficients in centered representation
        result_vec = np.zeros(2 * self.N, dtype=np.float64)
        for i in range(self.N):
            for j in range(self.N):
                result_vec[i + j] += vec1[i] * vec2[j]
        
        # Reduce modulo X^N + 1 in centered representation
        reduced_vec = np.zeros(self.N, dtype=np.float64)
        for i in range(len(result_vec)):
            if i < self.N:
                reduced_vec[i] += result_vec[i]
            else:
                reduced_vec[i - self.N] -= result_vec[i]
        
        # Round to nearest integer in centered representation
        reduced_vec = np.round(reduced_vec)
        
        # Center coefficients around zero before modulo operation
        reduced_vec = np.where(reduced_vec > self._current_q/2, reduced_vec - self._current_q, reduced_vec)
        reduced_vec = np.where(reduced_vec < -self._current_q/2, reduced_vec + self._current_q, reduced_vec)
        
        # Convert back to positive representation modulo current_q
        reduced_vec = np.array([(int(x) % self._current_q + self._current_q) % self._current_q 
                               for x in reduced_vec], dtype=object)
        
        result = PolyRing(reduced_vec, self._current_q)
        
        # Multiply scale factors
        if hasattr(self, '_scale_factor') and hasattr(other, '_scale_factor'):
            result._scale_factor = self._scale_factor * other._scale_factor
            
        return result

    def __repr__(self):
        return str(self.vec)

    def __sub__(self, other):
        """Subtract two polynomials."""
        if not isinstance(other, PolyRing):
            other = PolyRing(np.array([other] + [0] * (self.N - 1)), self._current_q)
            
        assert self._current_q == other._current_q, "Moduli Q must match for subtraction"
        result_vec = [(x - y) % self._current_q for x, y in zip(self.vec, other.vec)]
        return PolyRing(result_vec, self._current_q)

    def __neg__(self):
        """Negate a polynomial."""
        result_vec = [(-x) % self._current_q for x in self.vec]
        return PolyRing(result_vec, self._current_q)

    def __rmul__(self, other):
        return self * other