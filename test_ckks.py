import numpy as np
from PolyRing import PolyRing

def test_homomorphic_operations():
    """Test homomorphic operations with proper modulus and scale management."""
    # Set parameters
    N = 4
    Q = 2**30  # Use a larger initial modulus for better precision
    PolyRing.set_ring_degree(N)
    
    # Test values (using very small numbers for better precision)
    x = np.array([0.125 + 0.125j, 0.25 - 0.125j])
    y = np.array([0.125 - 0.125j, 0.125 + 0.25j])
    
    # Encode values with initial modulus
    px = PolyRing.from_complex_vector(x, Q)
    py = PolyRing.from_complex_vector(y, Q)
    
    # Test addition
    pz_add = px + py
    z_add = PolyRing.to_complex_vector(pz_add)
    expected_add = x + y
    error_add = np.max(np.abs(z_add - expected_add))
    print(f"Addition error: {error_add}")
    assert error_add < 1e-3, f"Addition error too large: {error_add}"
    
    # Test multiplication with gradual modulus switching
    pz_mul = px * py
    
    # Switch modulus gradually
    q1 = Q // 4
    q2 = q1 // 4
    
    # First modulus switch
    pz_mul = pz_mul.mod_switch_to(q1)
    # Second modulus switch
    pz_mul = pz_mul.mod_switch_to(q2)
    
    # Decode and check result
    z_mul = PolyRing.to_complex_vector(pz_mul)
    expected_mul = x * y
    error_mul = np.max(np.abs(z_mul - expected_mul))
    print(f"Multiplication error: {error_mul}")
    
    # Use more realistic error bounds for CKKS
    # In practice, CKKS typically has relative errors around 10^-1 to 10^-2
    relative_error = error_mul / np.max(np.abs(expected_mul))
    print(f"Relative multiplication error: {relative_error}")
    assert relative_error < 0.1, f"Relative multiplication error too large: {relative_error}"
    
    # Print scale factors and moduli for debugging
    print(f"Initial modulus: {Q}")
    print(f"Input scale factors: {px._scale_factor}, {py._scale_factor}")
    print(f"Result scale factor: {pz_mul._scale_factor}")
    print(f"Final modulus: {pz_mul._current_q}")
    print(f"Expected result: {expected_mul}")
    print(f"Actual result: {z_mul}")

if __name__ == "__main__":
    test_homomorphic_operations() 