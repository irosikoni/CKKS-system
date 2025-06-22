import numpy as np
from PolyRing import PolyRing
from ckks import CKKSContext, Ciphertext, encode, decode

print("=== CKKS MULTIPLICATION DEPTH TEST (ABSOLUTE ERROR) ===")

def test_multiplication_depth(context, keygen, max_depth=10):
    """Test how many consecutive multiplications can be performed before decryption fails"""
    print(f"\n--- Multiplication Depth Test (max {max_depth}) ---")
    
    # Create test vector (small values for multiplication)
    test_z = np.array([1.0 + 0.5j, 0.5 - 0.25j, 0.25 + 0.125j, 0.125 + 0j])
    
    # Encode and encrypt
    m = encode(test_z, context.global_delta, context.current_q)
    ctxt = Ciphertext.encrypt(m, keygen.public_key, context)
    
    print(f"Initial vector: {test_z}")
    print(f"Initial delta: {context.global_delta}")
    print(f"Initial modulus: {context.current_q}")
    print()
    
    # Test consecutive multiplications
    for depth in range(1, max_depth + 1):
        print(f"--- Depth {depth} ---")
        
        # Multiply by the same ciphertext
        ctxt_raw = ctxt.multiply(ctxt)
        
        # Check if we can still relinearize
        try:
            ctxt = ctxt_raw.relinearize()
        except ValueError as e:
            print(f"Relinearization failed: {e}")
            return depth - 1
        
        # Decrypt and decode
        try:
            decrypted_poly = Ciphertext.decrypt(ctxt, keygen)
            decrypted = decode(decrypted_poly, ctxt.delta)
            
            # Calculate expected result
            expected = test_z ** (2 ** depth)
            
            # Calculate absolute error only
            abs_error = np.abs(decrypted - expected)
            mean_abs_error = np.mean(abs_error)
            max_abs_error = np.max(abs_error)
            
            print(f"Expected: {expected}")
            print(f"Decrypted: {decrypted}")
            print(f"Mean absolute error: {mean_abs_error:.2e}")
            print(f"Max absolute error: {max_abs_error:.2e}")
            print(f"Current delta: {ctxt.delta:.2e}")
            print()
            
            # Check if absolute error is too large (threshold: 1.0)
            if mean_abs_error > 1.0:
                print(f"Multiplication depth limit reached at depth {depth} (mean abs_error > 1.0)")
                return depth - 1
                
        except Exception as e:
            print(f"Multiplication failed: {e}")
            return depth - 1
    
    print(f"Multiplication depth test completed successfully up to depth {max_depth}")
    return max_depth

def test_parameter_combinations():
    """Test different parameter combinations for multiplication depth"""
    print("\n--- Parameter Combinations Test ---")
    
    param_sets = [
        ("2 moduli", 8, [30, 30], 20),
        ("3 moduli", 8, [30, 30, 30], 20),
        ("4 moduli", 8, [30, 30, 30, 30], 20),
        ("5 moduli", 8, [30, 30, 30, 30, 30], 20),
        ("40-bit 3", 8, [40, 40, 40], 20),
        ("40-bit 4", 8, [40, 40, 40, 40], 20),
        ("Small delta", 8, [30, 30, 30], 15),
        ("Large delta", 8, [30, 30, 30], 25),
    ]
    
    results = {}
    
    for name, N, q_sizes, delta_bits in param_sets:
        print(f"\n{'='*60}")
        print(f"Testing: {name} - N={N}, q_sizes={q_sizes}, delta_bits={delta_bits}")
        print(f"{'='*60}")
        
        try:
            context = CKKSContext(N, q_sizes, delta_bits)
            keygen = context.keygen
            
            mul_depth = test_multiplication_depth(context, keygen, max_depth=6)
            results[name] = mul_depth
            
        except Exception as e:
            print(f"Failed: {e}")
            results[name] = -1
    
    # Print summary
    print(f"\n{'='*60}")
    print("MULTIPLICATION DEPTH SUMMARY (ABSOLUTE ERROR)")
    print(f"{'='*60}")
    print(f"{'Parameter Set':<20} {'Depth':<10}")
    print("-" * 60)
    
    for name, depth in results.items():
        if depth == -1:
            print(f"{name:<20} {'FAILED':<10}")
        else:
            print(f"{name:<20} {depth:<10}")
    
    # Find best
    valid_results = {name: depth for name, depth in results.items() if depth != -1}
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1])
        print(f"\nBest multiplication depth: {best[1]} with {best[0]} parameters")

if __name__ == "__main__":
    print("CKKS Multiplication Depth Analysis (Absolute Error)")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test default parameters first
    N = 8
    q_sizes = [30, 30, 30]
    delta_bits = 20
    
    print(f"Default parameters: N={N}, q_sizes={q_sizes}, delta_bits={delta_bits}")
    
    context = CKKSContext(N, q_sizes, delta_bits)
    keygen = context.keygen
    
    default_depth = test_multiplication_depth(context, keygen, max_depth=6)
    print(f"\nDefault parameters multiplication depth: {default_depth}")
    
    # Test different parameter combinations
    test_parameter_combinations() 