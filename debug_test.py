import numpy as np
from PolyRing import PolyRing
from ckks import CKKSContext, Ciphertext, encode, decode

print("=== COMPREHENSIVE CKKS ERROR ANALYSIS ===")

# Test parameters (same as main.py)
N = 8
q_sizes = [30, 30]
delta_bits = 20

# Number of test examples
num_examples = 5

def test_poly_ring_operations(context):
    """Test polynomial ring operations"""
    print("\n--- Polynomial Ring Operations ---")
    q = context.current_q
    
    # Test addition
    p1 = PolyRing(np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=object), q)
    p2 = PolyRing(np.array([2, 3, 4, 5, 0, 0, 0, 0], dtype=object), q)
    result_add = p1 + p2
    expected_add = PolyRing(np.array([3, 5, 7, 9, 0, 0, 0, 0], dtype=object), q)
    add_error = np.mean(np.abs(np.array(result_add.vec) - np.array(expected_add.vec)))
    print(f"Addition mean absolute error: {add_error:.2e}")
    
    # Test multiplication
    p3 = PolyRing(np.array([1, 2, 0, 0, 0, 0, 0, 0], dtype=object), q)
    p4 = PolyRing(np.array([2, 3, 0, 0, 0, 0, 0, 0], dtype=object), q)
    result_mul = p3 * p4
    expected_mul_coeffs = np.array([2, 7, 6, 0, 0, 0, 0, 0], dtype=object)
    expected_mul = PolyRing(expected_mul_coeffs, q)
    mul_error = np.mean(np.abs(np.array(result_mul.vec) - np.array(expected_mul.vec)))
    print(f"Multiplication mean absolute error: {mul_error:.2e}")

def test_encode_decode(context):
    """Test encoding and decoding error"""
    print("\n--- Encode/Decode Error ---")
    abs_errors = []
    rel_errors = []
    
    for i in range(num_examples):
        # Generate random complex vector
        test_z = np.random.rand(4) * 10 + 1j * (np.random.rand(4) * 10)
        
        # Encode and decode
        m = encode(test_z, context.global_delta, context.current_q)
        decoded = decode(m, context.global_delta)
        
        # Calculate errors
        abs_error = np.abs(decoded - test_z)
        rel_error = abs_error / (np.abs(test_z) + 1e-10)
        
        abs_errors.append(np.mean(abs_error))
        rel_errors.append(np.mean(rel_error))
    
    mean_abs_error = np.mean(abs_errors)
    mean_rel_error = np.mean(rel_errors)
    print(f"Mean absolute error: {mean_abs_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    print(f"Absolute error range: {min(abs_errors):.2e} - {max(abs_errors):.2e}")

def test_homomorphic_addition(context, keygen):
    """Test homomorphic addition error"""
    print("\n--- Homomorphic Addition Error ---")
    abs_errors = []
    rel_errors = []
    
    for i in range(num_examples):
        # Generate random complex vectors
        z1 = np.random.rand(4) * 10 + 1j * (np.random.rand(4) * 10)
        z2 = np.random.rand(4) * 10 + 1j * (np.random.rand(4) * 10)
        
        # Encode and encrypt
        m1 = encode(z1, context.global_delta, context.current_q)
        m2 = encode(z2, context.global_delta, context.current_q)
        
        ctxt1 = Ciphertext.encrypt(m1, keygen.public_key, context)
        ctxt2 = Ciphertext.encrypt(m2, keygen.public_key, context)
        
        # Add ciphertexts
        ctxt_add = ctxt1 + ctxt2
        decrypted_add_poly = Ciphertext.decrypt(ctxt_add, keygen)
        decrypted_add = decode(decrypted_add_poly, context.global_delta)
        
        # Calculate errors
        expected = z1 + z2
        abs_error = np.abs(decrypted_add - expected)
        rel_error = abs_error / (np.abs(expected) + 1e-10)
        
        abs_errors.append(np.mean(abs_error))
        rel_errors.append(np.mean(rel_error))
    
    mean_abs_error = np.mean(abs_errors)
    mean_rel_error = np.mean(rel_errors)
    print(f"Mean absolute error: {mean_abs_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    print(f"Absolute error range: {min(abs_errors):.2e} - {max(abs_errors):.2e}")

def test_homomorphic_multiplication(context, keygen):
    """Test homomorphic multiplication error"""
    print("\n--- Homomorphic Multiplication Error ---")
    abs_errors = []
    rel_errors = []
    
    for i in range(num_examples):
        # Generate random complex vectors
        z1 = np.random.rand(4) * 5 + 1j * (np.random.rand(4) * 5)  # Smaller values for multiplication
        z2 = np.random.rand(4) * 5 + 1j * (np.random.rand(4) * 5)
        
        # Encode and encrypt
        m1 = encode(z1, context.global_delta, context.current_q)
        m2 = encode(z2, context.global_delta, context.current_q)
        
        ctxt1 = Ciphertext.encrypt(m1, keygen.public_key, context)
        ctxt2 = Ciphertext.encrypt(m2, keygen.public_key, context)
        
        # Multiply and relinearize
        ctxt_mul_raw = ctxt1.multiply(ctxt2)
        ctxt_mul_relin = ctxt_mul_raw.relinearize()
        
        # Decrypt and decode
        decrypted_mul_poly = Ciphertext.decrypt(ctxt_mul_relin, keygen)
        decrypted_mul = decode(decrypted_mul_poly, ctxt_mul_relin.delta)
        
        # Calculate errors
        expected = z1 * z2
        abs_error = np.abs(decrypted_mul - expected)
        rel_error = abs_error / (np.abs(expected) + 1e-10)
        
        abs_errors.append(np.mean(abs_error))
        rel_errors.append(np.mean(rel_error))
    
    mean_abs_error = np.mean(abs_errors)
    mean_rel_error = np.mean(rel_errors)
    print(f"Mean absolute error: {mean_abs_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    print(f"Absolute error range: {min(abs_errors):.2e} - {max(abs_errors):.2e}")

def test_scalar_multiplication(context, keygen):
    """Test scalar multiplication error"""
    print("\n--- Scalar Multiplication Error ---")
    abs_errors = []
    rel_errors = []
    
    for i in range(num_examples):
        # Generate random complex vector and scalar
        z = np.random.rand(4) * 10 + 1j * (np.random.rand(4) * 10)
        factor = np.random.randint(2, 10)
        
        # Encode and encrypt
        m = encode(z, context.global_delta, context.current_q)
        ctxt = Ciphertext.encrypt(m, keygen.public_key, context)
        
        # Scalar multiplication
        ctxt_scaled = ctxt * factor
        
        # Decrypt and decode
        decrypted = decode(Ciphertext.decrypt(ctxt_scaled, keygen), context.global_delta)
        
        # Calculate errors
        expected = z * factor
        abs_error = np.abs(decrypted - expected)
        rel_error = abs_error / (np.abs(expected) + 1e-10)
        
        abs_errors.append(np.mean(abs_error))
        rel_errors.append(np.mean(rel_error))
    
    mean_abs_error = np.mean(abs_errors)
    mean_rel_error = np.mean(rel_errors)
    print(f"Mean absolute error: {mean_abs_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    print(f"Absolute error range: {min(abs_errors):.2e} - {max(abs_errors):.2e}")

def test_key_switching(context1, context2, keygen1, keygen2):
    """Test key switching error"""
    print("\n--- Key Switching Error ---")
    abs_errors = []
    rel_errors = []
    
    for i in range(num_examples):
        # Generate random complex vector
        z = np.random.rand(4) * 10 + 1j * (np.random.rand(4) * 10)
        
        # Encode and encrypt with first key
        m = encode(z, context1.global_delta, context1.current_q)
        ctxt = Ciphertext.encrypt(m, keygen1.public_key, context1)
        
        # Generate key switch key and switch
        ks_key = keygen2.generate_key_switch_key(keygen1.secret_key)
        ctxt_switched = ctxt.key_switch(ks_key)
        
        # Decrypt with second key
        decrypted_poly = Ciphertext.decrypt(ctxt_switched, keygen2)
        z_decoded = decode(decrypted_poly, context2.global_delta)
        
        # Calculate errors
        abs_error = np.abs(z_decoded - z)
        rel_error = abs_error / (np.abs(z) + 1e-10)
        
        abs_errors.append(np.mean(abs_error))
        rel_errors.append(np.mean(rel_error))
    
    mean_abs_error = np.mean(abs_errors)
    mean_rel_error = np.mean(rel_errors)
    print(f"Mean absolute error: {mean_abs_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    print(f"Absolute error range: {min(abs_errors):.2e} - {max(abs_errors):.2e}")

# Main test execution
if __name__ == "__main__":
    print(f"Parameters: N={N}, q_sizes={q_sizes}, delta_bits={delta_bits}")
    print(f"Testing {num_examples} examples for each operation")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create contexts and keygens
    context = CKKSContext(N, q_sizes, delta_bits)
    keygen = context.keygen
    
    context1 = CKKSContext(N, q_sizes, delta_bits)
    context2 = CKKSContext(N, q_sizes, delta_bits)
    keygen1 = context1.keygen
    keygen2 = context2.keygen
    
    # Run all tests
    test_poly_ring_operations(context)
    test_encode_decode(context)
    test_homomorphic_addition(context, keygen)
    test_homomorphic_multiplication(context, keygen)
    test_scalar_multiplication(context, keygen)
    test_key_switching(context1, context2, keygen1, keygen2)
    
    print("\n=== TEST COMPLETED ===") 