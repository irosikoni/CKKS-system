import numpy as np
from PolyRing import PolyRing
from ckks import CKKSContext, Ciphertext, encode, decode
import time

print("=== CKKS LARGE PARAMETERS TEST ===")

def test_large_parameters():
    """Test CKKS with larger, more realistic parameters"""
    
    # Large parameter sets to test
    param_sets = [
        ("N=8, q=40, d=40", 8, [40, 40, 40], 40),
        ("N=16, q=40, d=40", 16, [40, 40, 40], 40),
        ("N=16, q=60, d=60", 16, [60, 60, 60], 60),
        ("N=16, q=60, d=40", 16, [60, 60, 60], 40),
    ]
    
    results = {}
    
    for name, N, q_sizes, delta_bits in param_sets:
        print(f"\n{'='*80}")
        print(f"Testing: {name} - N={N}, q_sizes={q_sizes}, delta_bits={delta_bits}")
        print(f"{'='*80}")
        
        try:
            start_time = time.time()
            
            # Create context
            context = CKKSContext(N, q_sizes, delta_bits)
            keygen = context.keygen
            
            setup_time = time.time() - start_time
            print(f"Setup time: {setup_time:.2f} seconds")
            print(f"Modulus chain length: {len(context.q_chain)}")
            print(f"Global delta: {context.global_delta}")
            print(f"Current modulus: {context.current_q}")
            
            # Test vector (smaller values for larger N)
            test_z = np.array([1.0 + 0.5j, 0.5 - 0.25j] + [0.1 + 0.05j] * (N//2 - 2))
            
            # Encode and encrypt
            start_time = time.time()
            m = encode(test_z, context.global_delta, context.current_q)
            ctxt = Ciphertext.encrypt(m, keygen.public_key, context)
            encrypt_time = time.time() - start_time
            print(f"Encryption time: {encrypt_time:.2f} seconds")
            
            # Test basic operations
            print(f"\n--- Basic Operations Test ---")
            
            # Addition
            start_time = time.time()
            ctxt_add = ctxt + ctxt
            add_time = time.time() - start_time
            
            # Decrypt addition
            decrypted_poly = Ciphertext.decrypt(ctxt_add, keygen)
            decrypted_add = decode(decrypted_poly, context.global_delta)
            expected_add = test_z + test_z
            add_error = np.mean(np.abs(decrypted_add - expected_add))
            
            print(f"Addition time: {add_time:.2f} seconds")
            print(f"Addition error: {add_error:.2e}")
            
            # Multiplication
            start_time = time.time()
            ctxt_mul_raw = ctxt.multiply(ctxt)
            ctxt_mul = ctxt_mul_raw.relinearize()
            mul_time = time.time() - start_time
            
            # Decrypt multiplication
            decrypted_poly = Ciphertext.decrypt(ctxt_mul, keygen)
            decrypted_mul = decode(decrypted_poly, ctxt_mul.delta)
            expected_mul = test_z * test_z
            mul_error = np.mean(np.abs(decrypted_mul - expected_mul))
            
            print(f"Multiplication time: {mul_time:.2f} seconds")
            print(f"Multiplication error: {mul_error:.2e}")
            
            # Test multiplication depth
            print(f"\n--- Multiplication Depth Test ---")
            depth = 0
            current_ctxt = ctxt
            
            for i in range(1, 4):  # Test up to 3 multiplications
                try:
                    start_time = time.time()
                    ctxt_raw = current_ctxt.multiply(current_ctxt)
                    current_ctxt = ctxt_raw.relinearize()
                    mul_time = time.time() - start_time
                    
                    # Decrypt and check error
                    decrypted_poly = Ciphertext.decrypt(current_ctxt, keygen)
                    decrypted = decode(decrypted_poly, current_ctxt.delta)
                    expected = test_z ** (2 ** i)
                    error = np.mean(np.abs(decrypted - expected))
                    
                    print(f"Depth {i}: time={mul_time:.2f}s, error={error:.2e}")
                    
                    if error > 1.0:
                        print(f"Depth limit reached at {i} (error > 1.0)")
                        depth = i - 1
                        break
                    depth = i
                    
                except Exception as e:
                    print(f"Failed at depth {i}: {e}")
                    depth = i - 1
                    break
            
            results[name] = {
                'setup_time': setup_time,
                'encrypt_time': encrypt_time,
                'add_time': add_time,
                'mul_time': mul_time,
                'add_error': add_error,
                'mul_error': mul_error,
                'depth': depth,
                'N': N,
                'moduli': len(q_sizes)
            }
            
        except Exception as e:
            print(f"Failed: {e}")
            results[name] = {'error': str(e)}
    
    # Print summary
    print(f"\n{'='*80}")
    print("LARGE PARAMETERS TEST SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<12} {'N':<4} {'Moduli':<8} {'Setup(s)':<10} {'Encrypt(s)':<12} {'Add(s)':<8} {'Mul(s)':<8} {'AddErr':<10} {'MulErr':<10} {'Depth':<6}")
    print("-" * 80)
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name:<12} {'ERROR':<4} {'ERROR':<8} {'ERROR':<10} {'ERROR':<12} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10} {'ERROR':<6}")
        else:
            print(f"{name:<12} {result['N']:<4} {result['moduli']:<8} {result['setup_time']:<10.2f} {result['encrypt_time']:<12.2f} {result['add_time']:<8.2f} {result['mul_time']:<8.2f} {result['add_error']:<10.2e} {result['mul_error']:<10.2e} {result['depth']:<6}")

if __name__ == "__main__":
    print("CKKS Large Parameters Performance Test")
    print("Testing with realistic CKKS parameters...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    test_large_parameters() 