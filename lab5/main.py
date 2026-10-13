import numpy as np
from scipy.special import erf


# 1. Підінтегральна функція згідно з варіантом
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


# 2. Точне значення інтеграла (через аналітичні інтеграли та функцію erf)
def exact_integral(a, b):
    int1 = 50 * (b - a)
    int2 = (-240 / np.pi) * (
        np.cos(np.pi * b / 12) - np.cos(np.pi * a / 12)
    )
    coeff = np.sqrt(0.2)
    int3 = (
        5
        * (np.pi / 0.2) ** 0.5
        * 0.5
        * (
            erf(coeff * (b - 12)) - erf(coeff * (a - 12))
        )
    )
    return int1 + int2 + int3


I_0 = exact_integral(0, 24)


# 3. Функція для складової формули Сімпсона
def simpson_composite(a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    integral = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * integral


# 4. Дослідження залежності точності від N (від 10 до 1000)
N_values = range(10, 1001, 2)
errors = [abs(simpson_composite(0, 24, n) - I_0) for n in N_values]

# Знаходження N_opt для заданої точності eps = 1e-12
target_eps = 1e-12
N_opt = None
for n in N_values:
    if abs(simpson_composite(0, 24, n) - I_0) <= target_eps:
        N_opt = n
        break
if N_opt is None:
    N_opt = 1000

eps_opt = abs(simpson_composite(0, 24, N_opt) - I_0)

# 5. Вибір N0 кратним 8 (беремо достатньо велике значення, щоб методи Рунге-Ромберга та Ейткена працювали в асимптотичній зоні)
N_0 = 24
eps_0 = abs(simpson_composite(0, 24, N_0) - I_0)

# 6. Метод Рунге-Ромберга (p = 4 для формули Сімпсона, q = 2 -> q^4 - 1 = 15)
I_N0 = simpson_composite(0, 24, N_0)
I_N0_2 = simpson_composite(0, 24, N_0 // 2)
I_R = I_N0 + (I_N0 - I_N0_2) / 15
eps_R = abs(I_R - I_0)

# 7. Метод Ейткена та оцінка порядку методу
I_N0_4 = simpson_composite(0, 24, N_0 // 4)
denominator = 2 * I_N0_2 - (I_N0 + I_N0_4)
if abs(denominator) > 1e-12:
    I_E = (I_N0_2**2 - I_N0 * I_N0_4) / denominator
else:
    I_E = I_N0

p_aitken = (1 / np.log(2)) * np.log(
    abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0))
)
eps_E = abs(I_E - I_0)

#Адаптивний алгоритм інтегрування
def adaptive_simpson(a, b, delta, f):
    def c_simpson(start, end):
        h = end - start
        mid = (start + end) / 2
        x = np.linspace(start, end, 5)
        y = f(x)
        i1 = (h / 6) * (y[0] + 4 * y[2] + y[4])
        i2 = (h / 12) * (y[0] + 4 * y[1] + y[2]) + (h / 12) * (
            y[2] + 4 * y[3] + y[4]
        )
        if abs(i1 - i2) <= 15 * delta:
            return i2 + (i2 - i1) / 15
        else:
            return c_simpson(start, mid) + c_simpson(mid, end)

    return c_simpson(a, b)


# Виведення результатів
print(f"Точне значення I_0: {I_0:.6f}")
print(f"N_opt для eps=1e-12: {N_opt}, досягнута похибка: {eps_opt:.2e}")
print(f"N_0: {N_0}, похибка eps_0: {eps_0:.2e}")
print(f"Метод Рунге-Ромберга: I_R = {I_R:.6f}, eps_R = {eps_R:.2e}")
print(
    f"Метод Ейткена: I_E = {I_E:.6f}, порядок p = {p_aitken:.2f}, eps_E = {eps_E:.2e}"
)
print("-" * 50)
print("Дослідження адаптивного алгоритму:")
for d in [1e-3, 1e-5, 1e-7]:
    res_ad = adaptive_simpson(0, 24, d, f)
    print(f"Delta: {d} -> Результат: {res_ad:.6f}, похибка: {abs(res_ad - I_0):.2e}")