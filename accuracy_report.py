import numpy as np
from PolyRing import PolyRing
from ckks import CKKSContext, Ciphertext, encode, decode

def generate_random_complex_vector(size, magnitude=1.0):
    """Generate random complex vector with controlled magnitude."""
    real = np.random.uniform(-magnitude, magnitude, size)
    imag = np.random.uniform(-magnitude, magnitude, size)
    return real + 1j * imag

def evaluate_operation(operation_name, inputs, expected_output, actual_output):
    """Calculate error metrics for an operation."""
    abs_error = np.abs(actual_output - expected_output)
    rel_error = np.abs(actual_output - expected_output) / (np.abs(expected_output) + 1e-10)
    
    metrics = {
        'max_abs_error': np.max(abs_error),
        'mean_abs_error': np.mean(abs_error),
        'std_abs_error': np.std(abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_rel_error': np.mean(rel_error),
        'std_rel_error': np.std(rel_error)
    }
    
    print(f"\n=== {operation_name} Accuracy ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2e}")
    return metrics

def run_accuracy_tests(num_tests=10, vector_size=4, magnitude=0.1):
    """Run accuracy tests for CKKS operations."""
    # Initialize CKKS context with conservative parameters
    PolyRing.set_ring_degree(vector_size * 2)  # N must be 2x vector_size
    context = CKKSContext(vector_size * 2, [30, 30, 30], 20)  # More conservative parameters
    keygen = context.keygen
    
    # Store results
    encode_decode_errors = []
    addition_errors = []
    multiplication_errors = []
    
    print(f"\nRunning {num_tests} tests with vectors of size {vector_size}")
    print(f"Input magnitude range: [-{magnitude}, {magnitude}]")
    print(f"Ring degree N: {vector_size * 2}")
    print(f"Initial modulus bits: {context.q_sizes}")
    print(f"Scale bits: {context.delta_bits}")
    
    for test_idx in range(num_tests):
        print(f"\nTest {test_idx + 1}/{num_tests}")
        
        # Generate random test vectors
        z1 = generate_random_complex_vector(vector_size, magnitude)
        z2 = generate_random_complex_vector(vector_size, magnitude)
        
        try:
            # Test encoding-decoding
            encoded = encode(z1, context.global_delta, context.current_q)
            decoded = decode(encoded, context.global_delta)
            encode_decode_metrics = evaluate_operation(
                "Encode-Decode",
                [z1],
                z1,
                decoded
            )
            encode_decode_errors.append(encode_decode_metrics['mean_rel_error'])
            
            # Test addition
            m1 = encode(z1, context.global_delta, context.current_q)
            m2 = encode(z2, context.global_delta, context.current_q)
            
            # Create ciphertexts manually
            u = PolyRing.random_small(context.current_q)
            e1 = PolyRing.random_small(context.current_q)
            e2 = PolyRing.random_small(context.current_q)
            
            a, b = keygen.public_key
            c0_1 = b * u + e1 + m1
            c1_1 = a * u + e2
            ctxt1 = Ciphertext(c0_1, c1_1, context)
            
            u = PolyRing.random_small(context.current_q)
            e1 = PolyRing.random_small(context.current_q)
            e2 = PolyRing.random_small(context.current_q)
            c0_2 = b * u + e1 + m2
            c1_2 = a * u + e2
            ctxt2 = Ciphertext(c0_2, c1_2, context)
            
            # Test addition
            ctxt_add = ctxt1 + ctxt2
            decrypted_add = Ciphertext.decrypt_static(ctxt_add, keygen)
            result_add = decode(decrypted_add, context.global_delta)
            
            addition_metrics = evaluate_operation(
                "Addition",
                [z1, z2],
                z1 + z2,
                result_add
            )
            addition_errors.append(addition_metrics['mean_rel_error'])
            
            # Test multiplication
            ctxt_mul = ctxt1 * ctxt2
            ctxt_mul = ctxt_mul.relinearize()
            ctxt_mul = ctxt_mul.rescale()
            
            decrypted_mul = Ciphertext.decrypt_static(ctxt_mul, keygen)
            result_mul = decode(decrypted_mul, ctxt_mul.delta)
            
            multiplication_metrics = evaluate_operation(
                "Multiplication",
                [z1, z2],
                z1 * z2,
                result_mul
            )
            multiplication_errors.append(multiplication_metrics['mean_rel_error'])
            
        except Exception as e:
            print(f"Error in test {test_idx + 1}: {str(e)}")
            continue
    
    if encode_decode_errors:
        print("\n=== Final Summary ===")
        print(f"Average Encode-Decode Relative Error: {np.mean(encode_decode_errors):.2e} ± {np.std(encode_decode_errors):.2e}")
        if addition_errors:
            print(f"Average Addition Relative Error: {np.mean(addition_errors):.2e} ± {np.std(addition_errors):.2e}")
        if multiplication_errors:
            print(f"Average Multiplication Relative Error: {np.mean(multiplication_errors):.2e} ± {np.std(multiplication_errors):.2e}")
    else:
        print("\nNo successful tests to report.")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    run_accuracy_tests(num_tests=10, vector_size=4, magnitude=0.1) 