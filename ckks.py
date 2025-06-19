import numpy as np
from PolyRing import PolyRing # Upewnij się, że PolyRing.py jest poprawny

# --- Funkcje Pomocnicze (Generowanie Szumu) ---
# Przeniesione poza klasę Ciphertext, aby były dostępne globalnie
# Służą do generowania losowych wielomianów dla kluczy i szumu.

def _small_random_poly(n, bound=1):
    """
    Generuje wielomian o małych, losowych współczynnikach (np. {-1, 0, 1}).
    Używane do klucza tajnego i niektórych błędów.
    """
    return PolyRing(np.random.randint(-bound, bound + 1, size=n))

def _noise_poly(n, bound=5):
    """
    Generuje wielomian szumu o współczynnikach z małego zakresu.
    """
    return _small_random_poly(n, bound=bound)

# --- Kodowanie i Dekodowanie (funkcje fasadowe) ---

def encode(z: np.ndarray, delta: float) -> PolyRing:
    """
    Koduje wektor liczb zespolonych do wielomianu PolyRing.
    """
    return PolyRing.from_complex_vector(z, delta)

def decode(poly: PolyRing, delta: float) -> np.ndarray:
    """
    Dekoduje wielomian PolyRing z powrotem do wektora liczb zespolonych.
    """
    return poly.to_complex_vector(delta)

# --- Klasa Ciphertext (Szyfrogram) ---

class Ciphertext:
    def __init__(self, c0: PolyRing, c1: PolyRing, delta: float, d2: PolyRing = None):
        """
        Inicjalizuje obiekt szyfrogramu CKKS.
        c0, c1: komponenty szyfrogramu (wielomiany PolyRing)
        delta: aktualny współczynnik skalowania zaszyfrowanej wiadomości
        d2: opcjonalny trzeci komponent szyfrogramu, używany przed relinearyzacją po mnożeniu
        """
        self.c0 = c0
        self.c1 = c1
        self.delta = delta
        self._d2 = d2 # Przechowuje komponent d2 (c2) dla relinearyzacji

    def __add__(self, other):
        """
        Wykonuje homomorficzne dodawanie szyfrogramów.
        """
        # Sprawdzenie, czy delty są zgodne jest ważne dla zachowania poprawności skali
        assert self.delta == other.delta, "Deltas muszą być zgodne dla dodawania"
        return Ciphertext(self.c0 + other.c0, self.c1 + other.c1, self.delta)

    def __mul__(self, other):
        """
        Wykonuje homomorficzne mnożenie szyfrogramu przez skalar (int/float/numpy number).
        Dla mnożenia szyfrogram-szyfrogram użyj metody .multiply().
        """
        if isinstance(other, Ciphertext):
            raise ValueError("Użyj .multiply(other, relin_key) dla mnożenia szyfrogram-szyfrogram.")
        elif isinstance(other, (int, float, np.integer, np.floating)):
            # Mnożenie przez skalar zwiększa zaszyfrowaną wartość, ale NIE zmienia czynnika delta szyfrogramu.
            # Zmiana delta następuje po mnożeniu szyfrogram-szyfrogram, gdzie skala efektywnie się kwadratuje.
            return Ciphertext(self.c0 * other, self.c1 * other, self.delta)
        else:
            raise TypeError("Nieobsługiwany typ mnożenia")

    def multiply(self, other, relin_key):
        """
        Wykonuje homomorficzne mnożenie dwóch szyfrogramów.
        Generuje szyfrogram z trzema komponentami (c0, c1, d2), który wymaga relinearyzacji.
        """
        assert self.delta == other.delta, "Deltas muszą być zgodne dla mnożenia szyfrogram-szyfrogram."
        
        # Obliczenie komponentów (d0, d1, d2) nowego szyfrogramu
        d0 = self.c0 * other.c0
        d1 = self.c0 * other.c1 + self.c1 * other.c0
        d2 = self.c1 * other.c1
        
        # Efektywny współczynnik delta zaszyfrowanej wiadomości po mnożeniu podwaja się (delta * delta).
        # To jest kluczowe dla późniejszego skalowania.
        new_effective_delta = self.delta * other.delta
        
        # Zwracamy nowy szyfrogram z trzema komponentami i zaktualizowaną deltą.
        return Ciphertext(d0, d1, new_effective_delta, d2=d2)

    def relinearize(self, relin_key):
        """
        Redukuje szyfrogram z trzech komponentów (c0, c1, d2) do dwóch (c0_new, c1_new),
        przy użyciu klucza relinearyzacji.
        """
        if self._d2 is None:
            raise ValueError("Brak składnika c2 (d2) – najpierw wykonaj multiply().")
        
        a_r, b_r = relin_key # Klucz relinearyzacji
        
        # Relinearyzacja zgodnie z uproszczoną formą:
        # Nowe komponenty szyfrogramu powstają przez dodanie do c0 i c1 składowych d2*b_r i d2*a_r.
        # Ta forma działa, gdy klucz relinearyzacji (a_r, b_r) jest wygenerowany tak,
        # że b_r + a_r * s = s^2 + e (gdzie s to klucz tajny, e to szum).
        c0_new = self.c0 + (self._d2 * b_r)
        c1_new = self.c1 + (self._d2 * a_r)
        
        # Po relinearyzacji komponent d2 jest usuwany.
        result = Ciphertext(c0_new, c1_new, self.delta, d2=None)
        return result

    @staticmethod
    def _rescale_poly_by_factor(poly: PolyRing, factor: float) -> PolyRing:
        """
        Pomocnicza metoda statyczna do skalowania wielomianu przez podany współczynnik.
        Zapewnia poprawne zaokrąglanie i redukcję modulo q.
        """
        if factor == 0:
            raise ValueError("Współczynnik skalowania nie może być zerem.")
        scaled = np.round(poly.vec.astype(np.float64) / factor)
        # Zapewnienie, że współczynniki są w zakresie [0, q-1]
        reduced = (scaled % PolyRing.q + PolyRing.q) % PolyRing.q
        return PolyRing(reduced.astype(np.int64))

    def rescale(self, target_delta: float):
        """
        Skaluje szyfrogram do nowego, pożądanego współczynnika delta.
        Jest to operacja redukująca szum i normalizująca skalę po mnożeniu.
        """
        # Współczynnik, przez który dzielimy, to stosunek aktualnej delty do delty docelowej.
        # Self.delta w tym momencie to new_effective_delta z metody multiply().
        rescale_factor = self.delta / target_delta
        
        # Skalowanie komponentów c0 i c1
        c0_rescaled = Ciphertext._rescale_poly_by_factor(self.c0, rescale_factor)
        c1_rescaled = Ciphertext._rescale_poly_by_factor(self.c1, rescale_factor)
        
        # Jeśli komponent d2 nadal istnieje (co jest nietypowe po relinearyzacji, ale możliwe w pewnych sekwencjach),
        # również musi zostać przeskalowany.
        d2_rescaled = None
        if self._d2 is not None:
             d2_rescaled = Ciphertext._rescale_poly_by_factor(self._d2, rescale_factor)

        # Zwracamy nowy szyfrogram z nową, docelową deltą.
        return Ciphertext(c0_rescaled, c1_rescaled, target_delta, d2=d2_rescaled)

    def key_switch(self, key_switch_key):
        """
        Wykonuje operację przełączania kluczy szyfrogramu ze starego klucza tajnego na nowy.
        UWAGA: To jest uproszczona implementacja, która może być źródłem błędów
        w bardziej złożonych scenariuszach lub z dużym szumem.
        Prawdziwe key switching w CKKS jest bardziej skomplikowane i wymaga dekompozycji bazowej.
        """
        a_ks, b_ks = key_switch_key
        # Ta forma (c0 + c1 * b_ks, c1 * a_ks) jest uproszczoną wersją dla schematów typu BFV/BGV,
        # gdzie (a_ks, b_ks) jest kluczem dla P * s_old.
        c0_new = self.c0 + self.c1 * b_ks
        c1_new = self.c1 * a_ks
        return Ciphertext(c0_new, c1_new, self.delta)


    @staticmethod
    def encrypt(m: PolyRing, public_key, n, delta):
        """
        Szyfruje zaszyfrowaną wiadomość (w postaci wielomianu) przy użyciu klucza publicznego.
        """
        # n powinno być zgodne ze stopniem wielomianu PolyRing (N).
        N_poly = len(PolyRing.f) - 1 
        
        a, b = public_key # Klucz publiczny składa się z dwóch wielomianów (a, b)
        
        # Generowanie losowych wielomianów i szumu
        u = _small_random_poly(N_poly)
        e1 = _noise_poly(N_poly)
        e2 = _noise_poly(N_poly)
        
        # Konstrukcja szyfrogramu (c0, c1)
        c0 = b * u + e1 + m
        c1 = a * u + e2
        
        # Delta szyfrogramu to początkowa delta użyta do kodowania wiadomości.
        return Ciphertext(c0, c1, delta)

    @staticmethod
    def decrypt(ciphertext, keygen):
        """
        Deszyfruje szyfrogram przy użyciu klucza tajnego.
        Zwraca wielomian, który jest przybliżeniem oryginalnej wiadomości.
        """
        # Deszyfracja: m_poly = c0 + c1 * s (mod q), gdzie s to klucz tajny.
        return ciphertext.c0 + ciphertext.c1 * keygen.secret_key

# --- Klasa KeyGenerator ---

class KeyGenerator:
    def __init__(self, n: int):
        """
        Inicjalizuje generator kluczy.
        n: stopień wielomianu (N z PolyRing.f).
        """
        # Zapewnienie spójności: n powinno być równe N z PolyRing.f
        self.n = len(PolyRing.f) - 1 
        self.delta = 2**40 # Domyślna delta dla kodowania/dekodowania
        
        # Generowanie kluczy: tajnego, publicznego i relinearyzacji.
        self.secret_key, self.public_key = self._generate_keys(self.n)
        self.relin_key = self._generate_relin_key(self.secret_key, self.n)

    def _generate_keys(self, n: int):
        """
        Generuje parę kluczy: tajny (s) i publiczny (pk = (a, b)).
        """
        # Klucz tajny (s) jest zazwyczaj ternarny ({ -1, 0, 1 })
        s = _small_random_poly(n, bound=1) 
        # Wielomian 'a' jest losowany z całego zakresu modulo q
        a = PolyRing(np.random.randint(0, PolyRing.q, size=n)) 
        # Szum 'e'
        e = _noise_poly(n) 
        
        # Konstrukcja wielomianu 'b' dla klucza publicznego: b = -a*s + e (mod q)
        b = -a * s + e
        return s, (a, b)

    def _generate_relin_key(self, s: PolyRing, n: int):
        """
        Generuje klucz relinearyzacji dla kwadratu klucza tajnego (s^2).
        Uproszczona forma, która nie korzysta z dekompozycji bazowej.
        """
        s_squared = s * s # Oblicza s^2 jako wielomian PolyRing
        a_r = PolyRing(np.random.randint(0, PolyRing.q, size=n)) # Losowy wielomian dla klucza relinearyzacji
        e_r = _noise_poly(n) # Szum
        
        # b_r = -a_r * s + e_r + s^2. To jest typowa konstrukcja klucza relinearyzacji
        # dla schematów, które redukują s^2 do formy liniowej w s.
        b_r = -a_r * s + e_r + s_squared
        return a_r, b_r
    
    def generate_key_switch_key(self, s_old: PolyRing):
        """
        Generuje klucz przełączania kluczy (key switch key) z s_old na self.secret_key (s_new).
        UWAGA: To jest uproszczona implementacja, która może być źródłem błędów
        w bardziej złożonych scenariuszach lub z dużym szumem.
        Prawdziwe key switching w CKKS jest bardziej skomplikowane i wymaga dekompozycji bazowej.
        """
        N_poly = len(PolyRing.f) - 1
        a_ks = PolyRing(np.random.randint(0, PolyRing.q, size=N_poly)) # Losowy wielomian dla klucza
        e_ks = _noise_poly(N_poly) # Szum
        
        # Konstrukcja b_ks: b_ks + a_ks * s_new = s_old + e_ks (w przybliżeniu)
        # Czyli b_ks = -a_ks * s_new + s_old + e_ks.
        b_ks = -a_ks * self.secret_key + s_old + e_ks
        return a_ks, b_ks