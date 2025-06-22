import unittest
import numpy as np
from PolyRing import PolyRing
from ckks import CKKSContext, Ciphertext, encode, decode

class TestCKKS(unittest.TestCase):
    def setUp(self):
        self.N = 4  # Small N for testing
        self.q = 2**20  # 20-bit modulus
        PolyRing.set_ring_degree(self.N)
        
        # Use a longer moduli chain for multiplication operations
        self.q_sizes_test = [20, 20, 20, 20]  # Four levels with small primes
        self.delta_bits_test = 10  # Small scaling factor
        
        self.context = CKKSContext(self.N, self.q_sizes_test, self.delta_bits_test)
        self.keygen = self.context.keygen
        
        # Set random seed for reproducibility
        np.random.seed(42)

    def test_encode_decode_real(self):
        """Test encoding and decoding of real numbers"""
        x = np.array([1.0, -2.0])  # N/2 slots
        poly = PolyRing.from_complex_vector(x, self.q)
        y = PolyRing.to_complex_vector(poly)
        
        # Check relative error
        rel_error = np.abs(x - y) / (np.abs(x) + 1e-10)
        max_error = np.max(rel_error)
        self.assertLess(max_error, 0.1, f"Max relative error {max_error} exceeds threshold")
        
    def test_encode_decode_complex(self):
        """Test encoding and decoding of complex numbers"""
        x = np.array([1.0 + 2j, -2.0 - 1j])  # N/2 slots
        poly = PolyRing.from_complex_vector(x, self.q)
        y = PolyRing.to_complex_vector(poly)
        
        # Check relative error
        rel_error = np.abs(x - y) / (np.abs(x) + 1e-10)
        max_error = np.max(rel_error)
        self.assertLess(max_error, 0.1, f"Max relative error {max_error} exceeds threshold")
        
    def test_conjugate_symmetry(self):
        """Test that encoding preserves conjugate symmetry"""
        x = np.array([1.0 + 2j, -2.0 - 1j])  # N/2 slots
        poly = PolyRing.from_complex_vector(x, self.q)
        
        # Evaluate at all roots of unity
        k = np.arange(self.N)
        roots = np.exp(2j * np.pi * (2*k + 1) / (2*self.N))
        vand = np.vander(roots, self.N, increasing=True)
        evals = vand @ poly.vec
        
        # Check conjugate symmetry with a more realistic tolerance
        for i in range(self.N//2):
            conj_diff = np.abs(evals[i] - np.conj(evals[-(i+1)]))
            self.assertLess(conj_diff, 1e-9, f"Conjugate symmetry violated at index {i}")

    def test_encoding_decoding_simple(self):
        """Test encoding and decoding with simple values."""
        # Test with real numbers
        z_real = np.array([0.5, -0.5])  # Simple values for N=4
        
        poly_real = encode(z_real, self.context.global_delta, self.q)
        decoded_real = decode(poly_real, self.context.global_delta)
        
        np.testing.assert_allclose(np.real(decoded_real), z_real, rtol=1e-1, 
                                 err_msg="Real number encoding/decoding failed")
        np.testing.assert_allclose(np.imag(decoded_real), np.zeros_like(z_real), atol=1e-1,
                                 err_msg="Real number imaginary part should be zero")

        # Test with complex numbers
        z_complex = np.array([0.5+0.5j, -0.5-0.5j])  # Simple values for N=4
        
        poly_complex = encode(z_complex, self.context.global_delta, self.q)
        decoded_complex = decode(poly_complex, self.context.global_delta)
        
        np.testing.assert_allclose(decoded_complex, z_complex, rtol=1e-1,
                                 err_msg="Complex number encoding/decoding failed")

    def test_encoding_decoding_random(self):
        """Test encoding and decoding with random values."""
        num_slots = self.N // 2
        
        # Test with different magnitudes
        for scale in [0.1, 0.5]:  # Small scales for stability
            z = (np.random.randn(num_slots) + 1j * np.random.randn(num_slots)) * scale
            
            poly = encode(z, self.context.global_delta, self.q)
            decoded = decode(poly, self.context.global_delta)
            
            relative_error = np.linalg.norm(decoded - z) / np.linalg.norm(z)
            max_error = np.max(np.abs(decoded - z))
            
            print(f"\nEncoding/Decoding test with scale {scale}:")
            print(f"Relative error: {relative_error:.2e}")
            print(f"Maximum error: {max_error:.2e}")
            
            self.assertTrue(relative_error < 1e-1, 
                          f"High relative error {relative_error:.2e} for scale {scale}")

    def test_homomorphic_addition(self):
        """Test homomorphic addition with different patterns."""
        num_slots = self.N // 2
        
        # Test case 1: Simple addition
        z1 = np.array([0.5+0.5j, 0.5+0.5j])  # Simple values for N=4
        z2 = np.array([0.5-0.5j, 0.5-0.5j])  # Should sum to real values
        expected_sum = z1 + z2
        
        # Encrypt and add
        ctxt1 = Ciphertext.encrypt_static(
            encode(z1, self.context.global_delta, self.q),
            self.keygen.public_key, self.context
        )
        ctxt2 = Ciphertext.encrypt_static(
            encode(z2, self.context.global_delta, self.q),
            self.keygen.public_key, self.context
        )
        
        ctxt_sum = ctxt1 + ctxt2
        decrypted_sum = decode(
            Ciphertext.decrypt_static(ctxt_sum, self.keygen),
            self.context.global_delta
        )
        
        np.testing.assert_allclose(decrypted_sum, expected_sum, rtol=1e-1,
                                 err_msg="Homomorphic addition failed for simple case")
        
        # Test case 2: Random addition with small values
        z1 = np.random.randn(num_slots) + 1j * np.random.randn(num_slots)
        z2 = np.random.randn(num_slots) + 1j * np.random.randn(num_slots)
        z1 *= 0.1  # Scale down to improve stability
        z2 *= 0.1
        expected_sum = z1 + z2
        
        ctxt1 = Ciphertext.encrypt_static(
            encode(z1, self.context.global_delta, self.q),
            self.keygen.public_key, self.context
        )
        ctxt2 = Ciphertext.encrypt_static(
            encode(z2, self.context.global_delta, self.q),
            self.keygen.public_key, self.context
        )
        
        ctxt_sum = ctxt1 + ctxt2
        decrypted_sum = decode(
            Ciphertext.decrypt_static(ctxt_sum, self.keygen),
            self.context.global_delta
        )
        
        relative_error = np.linalg.norm(decrypted_sum - expected_sum) / np.linalg.norm(expected_sum)
        print(f"\nHomomorphic addition relative error: {relative_error:.2e}")
        self.assertTrue(relative_error < 1e-1, "High error in homomorphic addition")

    def test_homomorphic_multiplication(self):
        """Test homomorphic multiplication with proper rescaling."""
        num_slots = self.N // 2
        
        # Test case 1: Simple multiplication with small real numbers
        z1 = np.full(num_slots, 0.1)  # Use 0.1 for better stability
        z2 = np.full(num_slots, 0.1)
        expected_prod = z1 * z2
        
        ctxt1 = Ciphertext.encrypt_static(
            encode(z1, self.context.global_delta, self.q),
            self.keygen.public_key, self.context
        )
        ctxt2 = Ciphertext.encrypt_static(
            encode(z2, self.context.global_delta, self.q),
            self.keygen.public_key, self.context
        )
        
        # Multiply and manage scales
        ctxt_prod = ctxt1.multiply(ctxt2)
        ctxt_prod = ctxt_prod.relinearize()
        ctxt_prod = ctxt_prod.rescale()
        
        decrypted_prod = decode(
            Ciphertext.decrypt_static(ctxt_prod, self.keygen),
            ctxt_prod.delta
        )
        
        relative_error = np.linalg.norm(decrypted_prod - expected_prod) / np.linalg.norm(expected_prod)
        print(f"\nSimple multiplication relative error: {relative_error:.2e}")
        self.assertTrue(relative_error < 1e-1, "High error in simple homomorphic multiplication")
        
        # Test case 2: Complex number multiplication with small values
        z1 = (np.random.randn(num_slots) + 1j * np.random.randn(num_slots)) * 0.1
        z2 = (np.random.randn(num_slots) + 1j * np.random.randn(num_slots)) * 0.1
        expected_prod = z1 * z2
        
        ctxt1 = Ciphertext.encrypt_static(
            encode(z1, self.context.global_delta, self.q),
            self.keygen.public_key, self.context
        )
        ctxt2 = Ciphertext.encrypt(
            encode(z2, self.context.global_delta, self.q),
            self.keygen.public_key, self.context
        )
        
        # Multiply and manage scales
        ctxt_prod = ctxt1.multiply(ctxt2)
        ctxt_prod = ctxt_prod.relinearize()
        ctxt_prod = ctxt_prod.rescale()
        
        decrypted_prod = decode(
            Ciphertext.decrypt_static(ctxt_prod, self.keygen),
            ctxt_prod.delta
        )
        
        relative_error = np.linalg.norm(decrypted_prod - expected_prod) / np.linalg.norm(expected_prod)
        print(f"Complex multiplication relative error: {relative_error:.2e}")
        self.assertTrue(relative_error < 1e-1, "High error in complex homomorphic multiplication")

    def test_scale_management(self):
        """Test proper scale management through operations."""
        num_slots = self.N // 2
        initial_delta = self.context.global_delta
        
        # Use small values for better stability
        z1 = (np.random.randn(num_slots) + 1j * np.random.randn(num_slots)) * 0.1
        z2 = (np.random.randn(num_slots) + 1j * np.random.randn(num_slots)) * 0.1
        
        ctxt1 = Ciphertext.encrypt(
            encode(z1, initial_delta, self.q),
            self.keygen.public_key, self.context
        )
        ctxt2 = Ciphertext.encrypt(
            encode(z2, initial_delta, self.q),
            self.keygen.public_key, self.context
        )
        
        # Check scale after encryption
        np.testing.assert_allclose(ctxt1.delta, initial_delta, rtol=1e-1,
                                 err_msg="Scale changed after encryption")
        
        # Check scale after addition
        ctxt_sum = ctxt1 + ctxt2
        np.testing.assert_allclose(ctxt_sum.delta, initial_delta, rtol=1e-1,
                                 err_msg="Scale changed after addition")
        
        # Check scale after multiplication
        ctxt_prod = ctxt1.multiply(ctxt2)
        np.testing.assert_allclose(ctxt_prod.delta, initial_delta * initial_delta, rtol=1e-1,
                                 err_msg="Incorrect scale after multiplication")
        
        # Check scale after relinearization
        ctxt_relin = ctxt_prod.relinearize()
        self.assertTrue(ctxt_relin.delta < ctxt_prod.delta,
                       msg="Scale did not decrease after relinearization")
        
        # Check scale after rescaling
        ctxt_rescaled = ctxt_relin.rescale()
        self.assertTrue(ctxt_rescaled.delta < ctxt_relin.delta,
                       msg="Scale did not decrease after rescaling")
        np.testing.assert_allclose(ctxt_rescaled.delta, initial_delta, rtol=1e-1,
                                 err_msg="Final scale not close to initial scale")

if __name__ == '__main__':
    unittest.main(verbosity=2)
