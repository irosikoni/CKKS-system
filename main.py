from PolyRing import PolyRing
import numpy as np

def test_poly_ring_operations():
    z = np.array([1.0, 2.0, -0.5, 3.0])
    delta = 2**40
    poly = PolyRing.from_complex_vector(z, delta)
    decoded = poly.to_complex_vector(delta)
    print("Odkodowany:", decoded)

def main():
    test_poly_ring_operations()


if __name__ == "__main__":
    main()