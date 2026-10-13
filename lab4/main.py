import numpy as np

# 1. Задана функція та її аналітична похідна
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    # Аналітична похідна: M'(t) = -5e^(-0.1t) + 5cos(t)
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1.0
exact_val = dM_exact(t0)
print(f"1. Точне значення похідної в точці t0 = {t0}: {exact_val:.6f}\n")

# Центральна різницева формула
def central_diff(f, t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

# 2. Дослідження залежності похибки від кроку h
h_values = np.logspace(-20, 3, 1000)
errors = [abs(central_diff(M, t0, h) - exact_val) for h in h_values]

best_idx = np.argmin(errors)
h0 = h_values[best_idx]
R0 = errors[best_idx]

print(f"2. Дослідження кроку:")
print(f"   Оптимальний крок h0 = {h0:.2e}")
print(f"   Досягнута мінімальна похибка R0 = {R0:.2e}\n")

# 3, 4, 5. Обчислення з кроками h = 0.01 та 2h = 0.02 (або h=0.005 і h=0.01 за методичкою)
# Згідно з інструкцією в п.3-5 візьмемо h = 0.01
h_val = 0.01
d_h = central_diff(M, t0, h_val)
d_2h = central_diff(M, t0, 2 * h_val)

R1 = abs(d_h - exact_val)
print(f"3-5. Обчислення для h = {h_val}:")
print(f"   D(h) = {d_h:.6f}, Похибка R1 = {R1:.5f}")

# 6. Метод Рунге-Ромберга (для p = 2)
# p = 2, отже (2^p - 1) = 3
p = 2
d_rr = d_h + (d_h - d_2h) / (2**p - 1)
R2 = abs(d_rr - exact_val)
print(f"6. Метод Рунге-Ромберга:")
print(f"   Уточнене значення D_RR = {d_rr:.6f}")
print(f"   Похибка R2 = {R2:.2e}\n")

# 7. Метод Ейткена та оцінка порядку точності
# Використовуємо три кроки: h, 2h, 4h
h_e = 0.01
d1 = central_diff(M, t0, h_e)        # крок h
d2 = central_diff(M, t0, 2 * h_e)    # крок 2h
d3 = central_diff(M, t0, 4 * h_e)    # крок 4h

# Формула Ейткена
d_e = d1 - ((d2 - d1)**2) / (d3 - 2 * d2 + d1)
R3 = abs(d_e - exact_val)

# Оцінка порядку точності p
p_estimated = np.log(abs((d3 - d2) / (d2 - d1))) / np.log(2)

print(f"7. Метод Ейткена та порядок точності:")
print(f"   Уточнене значення за Ейткеном D_E = {d_e:.6f}")
print(f"   Похибка R3 = {R3:.2e}")
print(f"   Оцінений порядок точності p = {p_estimated:.4f}")