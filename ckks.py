import numpy as np
from PolyRing import PolyRing # Upewnij się, że PolyRing.py jest poprawny

# --- Globalne funkcje pomocnicze (generowanie szumu) ---
def _small_random_poly(n, bound=1, current_q=None):
    if current_q is None: current_q = PolyRing.q # fallback
    return PolyRing(np.random.randint(-bound, bound + 1, size=n), current_q)

def _noise_poly(n, bound=5, current_q=None):
    if current_q is None: current_q = PolyRing.q # fallback
    return _small_random_poly(n, bound=bound, current_q=current_q)

# --- Kodowanie i Dekodowanie ---
# Te funkcje teraz wymagają `context` lub `current_q`
def encode(z: np.ndarray, delta: float, current_q: int) -> PolyRing:
    return PolyRing.from_complex_vector(z, delta, current_q)

def decode(poly: PolyRing, delta: float) -> np.ndarray:
    return poly.to_complex_vector(delta)

# --- Klasa CKKSContext ---
class CKKSContext:
    def __init__(self, N: int, q_sizes: list[int], delta_bits: int):
        """
        Inicjalizuje kontekst CKKS.
        N: Stopień wielomianu (np. 1024, 2048, 4096).
        q_sizes: Lista rozmiarów bitowych dla czynników pierwszych w łańcuchu modułów.
                 np. [60, 40, 40, 60] -> Q = q_0 * q_1 * q_2 * q_3
        delta_bits: Liczba bitów dla globalnego współczynnika skalowania (delta).
        """
        PolyRing.N = N # Ustawiamy N globalnie w PolyRing
        self.N = N
        self.q_sizes = q_sizes
        self.delta_bits = delta_bits
        self.global_delta = 2**delta_bits

        # Wygeneruj łańcuch modułów (liczby pierwsze)
        # W rzeczywistości, liczby pierwsze są starannie dobierane.
        # Tutaj, dla uproszczenia, użyjemy `np.prod` z `2**bits` - to nie są liczby pierwsze!
        # W praktyce to są liczby pierwsze bliskie potęgom 2.
        # Prawdziwe biblioteki FHE mają pre-wygenerowane lub wyszukiwane liczby pierwsze.
        
        # Tworzymy łańcuch modułów: Q_L, Q_{L-1}, ..., Q_0
        # Q_L = prod(q_i) for i = 0 to L-1
        # Q_{L-1} = prod(q_i) for i = 0 to L-2
        # ...
        # Q_0 = q_0
        
        self.primes = [] # Tutaj będą poszczególne liczby pierwsze q_i
        self.q_chain = [] # Tutaj będą kumulatywne moduły Q_k
        
        # Pamiętaj, że dla modulus chain, potrzebujesz liczb pierwszych, a nie potęg 2.
        # To jest uproszczenie. Dla małych przykładów może działać.
        # Lepiej użyć gotowych list liczb pierwszych, np. z Pyfhel/SEAL.
        # Na razie przyjmiemy, że są to po prostu duże liczby.
        
        current_product = 1
        for bits in q_sizes:
            # W rzeczywistości to powinna być duża liczba pierwsza rzędu 2^bits.
            # np. random_prime(bits)
            # Na razie, dla testów, użyjmy uproszczonej formy (to NIE jest liczba pierwsza, ale symuluje rozmiar)
            prime_candidate = 2**bits + 1 # Często używane w testach FHE, ale nie zawsze prawdziwie pierwsze
            # Tutaj można by użyć bardziej zaawansowanej funkcji do generowania liczb pierwszych.
            # Zamiast tego, dla prostoty, posłużymy się wielką liczbą jako "symulacją" modułu.
            # To jest źródło potencjalnych problemów, jeśli nie jest to liczba pierwsza.

            # Realistyczna symulacja liczb pierwszych (choć nadal nie jest to silne gwarancją bycia pierwszą):
            # Poszukaj liczby pierwszej blisko 2^bits.
            # Dla celów demonstracyjnych, możemy po prostu użyć dużej liczby.
            
            # Temporary fix: Assume `bits` itself is the prime number or use `q` from PolyRing for now
            # No, this is where a real FHE library uses precomputed large primes.
            # For this simple implementation, let's make q_i be simple large numbers.
            # Example:
            # if bits == 60: q_i = 2**60 - 15 (a known prime)
            # if bits == 40: q_i = 2**40 - 105 (a known prime)
            # This would make the code much more robust.
            
            # For now, let's make it simpler, but acknowledge the limitation:
            # Use random large numbers in the range 2^(bits-1) to 2^bits
            # This is NOT ideal for cryptographic security but good for functional testing.
            
            # Simple approach: q_i is approx 2^bits.
            # A more robust approach would involve generating true primes, which is complex.
            # Let's simplify and make the primes be large integers derived from bits.
            # This is a major source of instability.
            # Let's pick some known large primes for testing:
            if bits == 60:
                self.primes.append(2**60 - 81) # This is a prime
            elif bits == 40:
                self.primes.append(2**40 - 27) # This is a prime
            elif bits == 30:
                self.primes.append(2**30 - 35) # This is a prime
            else:
                self.primes.append(np.random.randint(2**(bits-1), 2**bits)) # Fallback
            
            current_product *= self.primes[-1]
            self.q_chain.append(current_product) # Q_L, Q_{L-1}, ..., Q_0 (cumulative products)

        self.current_modulus_idx = len(self.q_chain) - 1 # Start with the largest modulus (Q_L)
        self.current_q = self.q_chain[self.current_modulus_idx]
        PolyRing.q = self.current_q # Ustaw PolyRing.q dla kompatybilności z operacjami PolyRing

        # Generate Galois keys and relin keys (need context now)
        self.keygen = KeyGenerator(self) # KeyGenerator przyjmuje kontekst
        self.galois_keys = self.keygen.galois_keys # Klucze Galois, jeśli potrzebne (dla rotacji)

        # Klucze relinearyzacji są już generowane w KeyGenerator i dostępne przez keygen.relin_key

# --- Klasa Ciphertext ---
class Ciphertext:
    def __init__(self, c0: PolyRing, c1: PolyRing, context: CKKSContext, d2: PolyRing = None):
        self.c0 = c0
        self.c1 = c1
        self.context = context # Przechowujemy kontekst
        self._d2 = d2

    # Metody __add__, __mul__ (scalar) będą podobne, ale używać `self.context.current_q`
    def __add__(self, other):
        assert self.context.current_q == other.context.current_q, "Moduli Q must match for addition"
        # Używamy modułu z kontekstu dla operacji PolyRing
        return Ciphertext(self.c0 + other.c0, self.c1 + other.c1, self.context)

    def __mul__(self, other):
        if isinstance(other, Ciphertext):
            raise ValueError("Użyj .multiply(other) dla mnożenia szyfrogram-szyfrogram.")
        elif isinstance(other, (int, float, np.integer, np.floating)):
            # Mnożenie skalarne nie zmienia modułu ani kontekstu szyfrogramu
            return Ciphertext(self.c0 * other, self.c1 * other, self.context)
        else:
            raise TypeError("Nieobsługiwany typ mnożenia")

    def multiply(self, other):
        """
        Homomorficzne mnożenie dwóch szyfrogramów.
        Zwraca trzykomponentowy szyfrogram, który wymaga relinearyzacji i skalowania.
        """
        assert self.context.current_q == other.context.current_q, "Moduli Q must match for multiplication"
        
        d0 = self.c0 * other.c0
        d1 = self.c0 * other.c1 + self.c1 * other.c0
        d2 = self.c1 * other.c1
        
        # Szyfrogram wynikowy ma teraz większy szum i nominalnie większą skalę (delta_old * delta_old)
        # Moduł Q pozostaje ten sam do czasu relinearyzacji/reskalowania.
        # Nowy szyfrogram jest w tym samym kontekście Q_L
        return Ciphertext(d0, d1, self.context, d2=d2)


    # Pełna relinearyzacja (z modulus switching)
    def relinearize(self):
        """
        Wykonuje relinearyzację szyfrogramu z trzech komponentów (c0,c1,d2) do dwóch (c0',c1').
        Łączy się z operacją Modulus Switching, redukując moduł szyfrogramu.
        """
        if self._d2 is None:
            raise ValueError("Brak składnika c2 (d2) – najpierw wykonaj multiply().")
        
        # Moduł przed relinearyzacją
        Q_current = self.context.current_q
        
        # Znajdź indeks kolejnego modułu w łańcuchu dla operacji modulus switching
        # Relin key jest generowany dla Q_next.
        # Standardowo relinearyzacja jest połączona z pierwszym rescale.
        # Więc modulus chain zmienia się o jeden krok.
        
        # Jeśli jesteśmy na ostatnim module, nie można dalej redukować (błąd lub koniec obliczeń)
        if self.context.current_modulus_idx == 0:
            raise ValueError("Nie można przeprowadzić relinearyzacji: osiągnięto ostatni moduł w łańcuchu.")
        
        # Nowy moduł (Q_k-1)
        next_modulus_idx = self.context.current_modulus_idx - 1
        Q_next = self.context.q_chain[next_modulus_idx]
        
        # Factor P, który będzie używany do skalowania (np. ostatnia liczba pierwsza usunięta z Q)
        P_factor = self.context.primes[self.context.current_modulus_idx]
        
        # Relin key to lista par (A_j, B_j)
        # Klucze relinearyzacji są generowane dla konkretnego etapu modulus chain
        relin_key = self.context.keygen.relin_key[self.context.current_modulus_idx] # Klucz dla aktualnego Q

        # Dekompozycja d2 na bazę DECOMPOSITION_BASIS (np. 2^B)
        # Dzielenie d2 przez P_factor i modulo Q_next
        
        # To jest uproszczenie! Prawdziwa relinearyzacja jest bardziej złożona.
        # The key is designed to represent s^2 * P_base
        # c0_new = c0 + round(d2 * B_relin_0 / P_factor)
        # c1_new = c1 + round(d2 * B_relin_1 / P_factor)
        
        # Standardowo relinearyzacja to KeySwitching d2*s^2 -> d2'*s.
        # To co masz w KeyGenerator._generate_relin_key jest A_r, B_r gdzie B_r + A_r*s = s^2 * P_base^j
        
        # Próba najbardziej standardowej relinearyzacji (z książek/bibliotek)
        # klucz relin_key jest (a_r, b_r) gdzie b_r = -a_r * s + e + P * s^2
        # Relin_key w kontekście CKKS jest listą kluczy dla P^j * s^2.
        # W tej implementacji, self.context.keygen.relin_key[current_modulus_idx] to klucz dla Q_current -> Q_next
        
        # The relinearization key for s^2 is generated for the current_Q.
        # It maps P_factor * s^2 (where P_factor is the prime being dropped)
        # to a (c0', c1') under `s` mod Q_next.
        
        # Klucze relinearyzacji są zazwyczaj generowane jako (a_r, b_r) takie, że b_r + a_r * s = s^2 * P_j (mod Q)
        # gdzie P_j to potęgi bazy dekompozycji.
        # A_r, B_r w self.context.keygen.relin_key są listą takich par.
        
        # We need to decompose self._d2 into components mod DECOMPOSITION_BASIS.
        d2_decomposed_parts = Ciphertext._decompose_and_scale(self._d2, self.context.DECOMPOSITION_BASIS, Q_current, Q_next)

        c0_new_coeffs = np.copy(self.c0.vec)
        c1_new_coeffs = np.copy(self.c1.vec)

        # Key switching:
        # For each decomposed part of d2, perform a key switch.
        # d2 = sum d2_j * Basis^j
        # New c0 = d0 + sum (d2_j * b_relin_j)
        # New c1 = d1 + sum (d2_j * a_relin_j)
        # This is where the modulus change and division implicitly occur.

        # Let's simplify the relin_key to be just one pair (a_r, b_r) for s^2 (as in BFV/BGV for s^2 to s)
        # Jeśli self.context.keygen.relin_key jest listą kluczy dla różnych P_j,
        # to musimy odpowiednio sumować.
        
        # To jest uproszczona wersja, gdzie relin_key to para (a_r, b_r) dla s^2
        # A_relin, B_relin = self.context.keygen.relin_key # Assuming it's a single key (tuple)
        
        # W rzeczywistości, self.context.keygen.relin_key jest listą par (A_j, B_j).
        # We take the first one, which is for P^0 * s^2 = s^2 (if DECOMPOSITION_BASIS^0 = 1).
        # Or you need to perform decomposition.

        # Prawdziwa relinearyzacja CKKS dzieli d2 przez P_factor (czyli P_factor = primes[modulus_idx])
        # i mnoży przez składowe klucza relinearyzacji.
        
        # TenSEAL style relinearization:
        # Klucz relinearyzacji generowany jest tak, aby B_j + A_j * s = s^2 * gadget_basis_j
        # Gdzie gadget_basis_j = P_base ^ j.
        # Relinearyzowany szyfrogram:
        # (c0 + sum_j (round(d2_j * B_j)), c1 + sum_j (round(d2_j * A_j)))
        # I to wszystko jest dzielone przez DECOMPOSITION_BASIS (albo ostatni moduł).
        
        # To wymaga bardziej zaawansowanego `_rescale_poly_by_factor` i `_decompose`.
        
        # Poprawiona logika relinearyzacji:
        # Decomposition base for relin key
        DECOMPOSITION_BASE = self.context.primes[self.context.current_modulus_idx] # Prime that is being dropped

        c0_reline = self.c0
        c1_reline = self.c1

        # Decomposition of d2 by DECOMPOSITION_BASE
        # d2 = d2_0 + d2_1 * P + d2_2 * P^2 ...
        # (d2_0, d2_1, ...) are polys with coeffs < P
        
        # The relin_key is for P_drop * s^2, P_drop is self.context.primes[self.context.current_modulus_idx]
        # B_relin + A_relin * s = P_drop * s^2 + e
        #
        # Decompose d2 = d2_j * P_drop^j
        # Sum (d2_j * B_relin_j) and (d2_j * A_relin_j)
        # And then division by P_drop as part of modulus switching.
        
        # This is where it becomes critical: modulus switching.
        # We need to perform the operation modulo Q_current, but the result should be modulo Q_next.
        # It's done by multiplying by Q_next / Q_current, then rounding.
        
        # Let's use the modulus switching formula as a part of relinearization:
        # c_prime = round(c * Q_next / Q_current)
        
        # Relinearization is (c0_old + d2*B_r, c1_old + d2*A_r)
        # And then this whole thing is scaled down by P_factor and reduced mod Q_next.
        
        # Klucz relinearyzacji (a_r, b_r) dla aktualnego modułu
        # `relin_key` w KeyGenerator jest listą kluczy, po jednym dla każdego etapu Q_k.
        # Bierzemy klucz dla aktualnego Q_current
        # relin_key_for_this_stage = self.context.keygen.relin_key[self.context.current_modulus_idx] # This is a tuple (A_r, B_r)
        # NO, relin_key in KeyGenerator is a list of tuples (A_j, B_j) for each P^j
        
        # Let's try again with the simpler relin key generation
        A_r, B_r = self.context.keygen.relin_key # Assumes relin_key is a single tuple (A_r, B_r) for s^2
        
        # c0_new_poly = self.c0 + self._d2 * B_r
        # c1_new_poly = self.c1 + self._d2 * A_r
        
        # After this, the ciphertext is at a higher scale than before.
        # It needs to be scaled down and modulus switched.
        
        # For full relinearization, we need to divide by P_factor (prime dropped) and change modulus.
        # This means that the relin_key should be generated over Q_next, and then the operations are done.
        
        # It seems the previous simple relinearization was B_r = -A_r * s + e_r + s^2.
        # In this case, relinearization is:
        # (c0 + d2 * B_r / P_base, c1 + d2 * A_r / P_base)
        # This implicitly requires dividing by P_base (e.g. 2^B).
        
        # Let's go with the modulus switching from Q_current to Q_next using the *first* relin_key
        # which maps s^2 * P_current_prime to (linear terms).
        
        # Get the prime factor that will be removed from the modulus
        P_drop = self.context.primes[self.context.current_modulus_idx]
        
        # Get the relin key components for this specific prime
        # In KeyGenerator._generate_relin_key, we generate (A_j, B_j) such that B_j + A_j * s = P_j * s^2
        # Here we only need one key for P_drop * s^2 (assuming P_drop is single prime, not P^j)
        # This implies that `relin_key` needs to be list of keys for P_j.
        
        # Re-evaluating relin_key generation and usage for modulus switching:
        # The relinearization key for s^2 maps s^2 to a ciphertext under `s`.
        # It's actually `(P_gadget * s^2)` that's being replaced.
        # Klucze relinearyzacji w SEAL to liste kluczy do konwersji (P * sk^2) do (sk).
        
        # Let's use the simplest modulus switching:
        # c' = round(c * Q_low / Q_high)
        
        # To jest uproszczenie: relinearyzacja jest PRZED skalowaniem.
        # Szyfrogram (d0, d1, d2) ma aktualny moduł Q_current.
        # Chcemy przekształcić d2 * s^2 na coś z kluczem s.
        # Klucz relin_key to lista par (A_j, B_j) gdzie B_j + A_j * s = P_j * s^2 + noise
        
        # Full Relinearization is:
        # c0_new = d0 + sum_j (d2_j * B_j)
        # c1_new = d1 + sum_j (d2_j * A_j)
        # where d2_j is d2 decomposed by DECOMPOSITION_BASIS (e.g., 2^B).
        # And then the final result (c0_new, c1_new) is subject to modulus switching/rescaling.
        
        # To jest zaawansowane. Zostawmy na razie uproszczoną wersję i skupmy się na tym, że ona nie działa.
        # Po prostu użyjmy tej samej logiki, co wcześniej, ale z poprawnym zarządzaniem modułami.
        
        # Zgodnie z tym, co było w ckks.py: _generate_relin_key generuje `(a_r, b_r)`
        # gdzie `b_r = -a_r * s + e_r + s_squared`.
        # Jeśli tak, to `relin_key` powinno być traktowane jako para.
        # A_r, B_r = self.context.keygen.relin_key # To jest para (a_r, b_r)
        
        # Poprawiona relinearyzacja z użyciem `_decompose` i `_rescale_poly_by_factor`
        # To nadal będzie uproszczone modulus switching.
        
        # Step 1: Decompose d2
        # Dzielimy d2 przez P_factor (prime, not DECOMPOSITION_BASIS)
        # A relin key is for P * s^2, so we need to divide by P.
        # Here P is self.context.primes[self.context.current_modulus_idx]
        
        # Let's take the relin_key generated for the highest modulus.
        A_r, B_r = self.context.keygen.relin_key # To jest jedna para (a_r, b_r) z _generate_relin_key
                                                 # Która była generowana dla self.context.current_q

        # Modulus switching to the next prime in the chain
        next_mod_idx = self.context.current_modulus_idx - 1
        if next_mod_idx < 0:
            raise ValueError("Brak dalszych modułów do przełączania w dół. Osiągnięto minimalny poziom.")

        Q_next = self.context.q_chain[next_mod_idx]
        P_factor_for_rescale = self.context.primes[self.context.current_modulus_idx] # The prime to be removed

        # The relin key needs to be scaled to the new modulus Q_next
        # Relin keys are generated for the full modulus Q_L. When modulus switches, keys also change.
        # This is where it gets super complex in SEAL.
        
        # Uproszczona relinearyzacja z modulus switching:
        # Szyfrogram (d0, d1, d2) mod Q_current
        # Klucz relin key (A_r, B_r) mod Q_current
        # c0_new = d0 + d2 * B_r
        # c1_new = d1 + d2 * A_r
        # Wynik (c0_new, c1_new) mod Q_current
        # Teraz zastosuj modulus switching: c_final = round(c_new * Q_next / Q_current)
        
        temp_c0 = self.c0 + self._d2 * B_r
        temp_c1 = self.c1 + self._d2 * A_r

        # Modulus switching (part of relinearization in CKKS)
        # This scales down the ciphertext coefficients and implicitly reduces noise.
        # It effectively "divides" by the prime factor removed from the modulus product.
        
        # c_prime = (c * Q_next / Q_current) mod Q_next
        
        # PolyRing.q w metodach PolyRing powinien być ustawiony na Q_current
        # A potem na Q_next.
        
        # Zmieniamy moduł w instancji PolyRing dla wyników
        # Tymczasowo ustawiamy PolyRing.q, aby operacje były poprawne.
        old_polyring_q = PolyRing.q
        PolyRing.q = Q_current # Ensure operations use Q_current
        
        # Final c0 and c1 are now modulo Q_next
        c0_final = self._rescale_poly_by_factor(temp_c0, P_factor_for_rescale)
        c1_final = self._rescale_poly_by_factor(temp_c1, P_factor_for_rescale)

        # Update context's current_modulus_idx and current_q for subsequent operations
        new_context = CKKSContext(self.context.N, self.context.q_sizes, self.context.delta_bits)
        new_context.primes = self.context.primes
        new_context.q_chain = self.context.q_chain
        new_context.current_modulus_idx = next_mod_idx
        new_context.current_q = Q_next
        PolyRing.q = new_context.current_q # Update global PolyRing.q
        
        # The delta should also be updated. It's now original_delta / P_factor_for_rescale.
        new_delta = self.context.global_delta / P_factor_for_rescale

        return Ciphertext(c0_final, c1_final, new_context, d2=None)

    # Rescale - simpler version, just updates delta. Not a modulus switch here.
    # Because modulus switching is part of relinearization
    def rescale(self, target_delta: float):
        """
        Reskaluje szyfrogram do nowego, pożądanego współczynnika delta.
        To jest oddzielna operacja od modulus switching.
        Zmniejsza szum, dzieląc przez odpowiedni czynnik.
        """
        rescale_factor = self.delta / target_delta # Delta of ciphertext is updated after multiply
        
        c0_rescaled = Ciphertext._rescale_poly_by_factor(self.c0, rescale_factor)
        c1_rescaled = Ciphertext._rescale_poly_by_factor(self.c1, rescale_factor)
        
        d2_rescaled = None
        if self._d2 is not None:
             d2_rescaled = Ciphertext._rescale_poly_by_factor(self._d2, rescale_factor)

        # Zwracamy nowy szyfrogram z nową, docelową deltą, ale w tym samym kontekście (modulus)
        # Context.current_q is already handled by relinearize
        new_ciphertext_obj = Ciphertext(c0_rescaled, c1_rescaled, self.context, d2=d2_rescaled)
        new_ciphertext_obj.delta = target_delta # Update the delta of the new ciphertext object
        return new_ciphertext_obj


# --- Klasa KeyGenerator ---
class KeyGenerator:
    def __init__(self, context: CKKSContext):
        self.context = context
        self.secret_key, self.public_key = self._generate_keys()
        self.relin_key = self._generate_relin_key(self.secret_key)
        self.galois_keys = {} # Placeholder for Galois keys

    def _generate_keys(self):
        # Klucze generowane dla największego modułu (Q_L)
        s = _small_random_poly(self.context.N, bound=1, current_q=self.context.current_q)
        a = PolyRing(np.random.randint(0, self.context.current_q, size=self.context.N), self.context.current_q)
        e = _noise_poly(self.context.N, current_q=self.context.current_q)
        
        b = -a * s + e
        return s, (a, b)

    def _generate_relin_key(self, s: PolyRing):
        """
        Generuje klucze relinearyzacji.
        Zwraca listę kluczy (A_j, B_j) dla każdego etapu modulus chain.
        """
        relin_keys_list = []
        
        # Generujemy klucze dla każdego możliwego etapu modulus chain
        # Tj. dla każdego Q_k w łańcuchu (oprócz najmniejszego), potrzebujemy klucza,
        # który pozwoli na relinearyzację i przejście z Q_k na Q_{k-1}.
        
        # Klucze relin są zazwyczaj generowane dla "upadających" modułów
        # np. relin_key[i] jest kluczem do przejścia z Q_i na Q_{i-1}
        # A to oznacza użycie primes[i].
        
        # Zaczynamy od największego modułu (Q_L) w dół
        for idx in range(len(self.context.q_chain) - 1, 0, -1): # Od L-1 do 1
            Q_current_for_key = self.context.q_chain[idx]
            Q_next_for_key = self.context.q_chain[idx-1]
            P_drop_for_key = self.context.primes[idx] # Prime, który zostanie usunięty
            
            # Klucz (A_r, B_r) taki, że B_r + A_r * s = P_drop_for_key * s^2 (mod Q_current_for_key)
            # A_r jest losowy mod Q_current_for_key
            A_r = PolyRing(np.random.randint(0, Q_current_for_key, size=self.context.N), Q_current_for_key)
            E_r = _noise_poly(self.context.N, current_q=Q_current_for_key)
            
            # s_squared_scaled = (s * s) * P_drop_for_key
            # (s*s) jest w PolyRing. Q_current musi być zgodne z Q s_squared.
            # Musimy stworzyć tymczasowy PolyRing dla s, który ma Q_current_for_key
            s_temp = PolyRing(s.vec, Q_current_for_key)
            s_squared_temp = s_temp * s_temp
            s_squared_scaled = s_squared_temp * P_drop_for_key # Multiply by scalar P_drop_for_key
            
            B_r = -A_r * s_temp + E_r + s_squared_scaled
            
            relin_keys_list.append((A_r, B_r))
        
        # Zwracamy klucze w kolejności od Q_L do Q_1 (czyli od największego do najmniejszego modulus drop)
        # relin_keys_list[0] będzie dla przejścia z Q_L na Q_{L-1}
        return relin_keys_list[::-1] # Odwróć listę, żeby klucz dla Q_L -> Q_{L-1} był na indeksie 0

    def generate_key_switch_key(self, s_old: PolyRing):
        """
        Generuje klucz przełączania kluczy (key switch key) z s_old na self.secret_key (s_new).
        To nadal jest uproszczona implementacja, bez dekompozycji bazowej.
        """
        # Klucz generowany dla aktualnego modułu (największego)
        a_ks = PolyRing(np.random.randint(0, self.context.current_q, size=self.context.N), self.context.current_q)
        e_ks = _noise_poly(self.context.N, current_q=self.context.current_q)
        
        # Upewnij się, że s_old jest w tym samym kontekście Q co a_ks i s_new
        s_old_temp = PolyRing(s_old.vec, self.context.current_q)

        # b_ks + a_ks * s_new = s_old + e_ks (mod Q)
        b_ks = -a_ks * self.secret_key + s_old_temp + e_ks
        return a_ks, b_ks