from PolyRing import PolyRing
import numpy as np

def test_poly_ring_operations():
    z = np.array([1 + 1j, 2 - 1j])
    delta = 2**10

    poly = PolyRing.from_complex_vector(z, delta)
    print("Zakodowany wielomian:", poly)

    decoded_z = poly.to_complex_vector(delta)
    print("Odkodowany wektor:", decoded_z)

def main():
    test_poly_ring_operations()


if __name__ == "__main__":
    main()