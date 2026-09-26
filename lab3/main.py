import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
csv_filename = "temperature_data.csv"
data = [
    (1, -2), (2, 0), (3, 5), (4, 10), (5, 15), (6, 20),
    (7, 23), (8, 22), (9, 17), (10, 10), (11, 5), (12, 0),
    (13, -10), (14, 3), (15, 7), (16, 13), (17, 19), (18, 20),
    (19, 22), (20, 21), (21, 18), (22, 15), (23, 10), (24, 3)
]

with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Month", "Temp"])
    writer.writerows(data)

def load_data(filename):
    df = pd.read_csv(filename)
    return df["Month"].values.astype(float), df["Temp"].values.astype(float)

x_data, y_data = load_data(csv_filename)
n_points = len(x_data)


def form_matrix(x, m):
    """Формування матриці B для системи МНК розміром (m+1, m+1)."""
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
    return A

def form_vector(x, y, m):
    """Формування вектора c для системи МНК розміром (m+1)."""
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))
    return b

def gauss_solve(A_in, b_in):
    """Розв'язання СЛАР методом Гауса з вибором головного елемента по стовпцях."""
    A = A_in.copy()
    b = b_in.copy()
    n = len(b)
    
    for k in range(n - 1):
        max_row = k + np.argmax(np.abs(A[k:n, k]))
        
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
            
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
            
    # Зворотний хід
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]
        
    return x_sol

def polynomial(x, coef):
    """Обчислення значення многочлена: P(x) = a0 + a1*x + a2*x^2 + ..."""
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * (x ** i)
    return y_poly

def calculate_variance(y_true, y_approx):
    """Обчислення дисперсії/середньоквадратичного відхилення."""
    return np.sqrt(np.mean((y_true - y_approx) ** 2))

max_m = 10
variances = []
coefficients_dict = {}

print("--- Результати обчислення дисперсії для m = 1..10 ---")
for m in range(1, max_m + 1):
    A = form_matrix(x_data, m)
    b_vec = form_vector(x_data, y_data, m)
    coef = gauss_solve(A, b_vec)
    coefficients_dict[m] = coef
    
    y_approx = polynomial(x_data, coef)
    var = calculate_variance(y_data, y_approx)
    variances.append(var)
    print(f"Степінь m = {m:2d} | Дисперсія (sigma) = {var:.4f}")


optimal_m = np.argmin(variances) + 1
print(f"\nОптимальний степінь полінома m = {optimal_m} (Дисперсія: {variances[optimal_m - 1]:.4f})")


x_fine = np.linspace(x_data[0], x_data[-1], 20 * n_points)
coef_opt = coefficients_dict[optimal_m]

# Значення апроксимації на густій сітці
y_fine_approx = polynomial(x_fine, coef_opt)

# Для похибки на вузлах x_data:
y_opt_approx = polynomial(x_data, coef_opt)
error_nodes = np.abs(y_data - y_opt_approx)


x_future = np.array([25, 26, 27], dtype=float)
y_future = polynomial(x_future, coef_opt)

print("\n--- Прогноз температури на наступні 3 місяці ---")
for m_num, temp in zip(x_future, y_future):
    print(f"Місяць {int(m_num)}: {temp:.2f} °C")

plt.figure(figsize=(14, 10))


plt.subplot(2, 2, 1)
plt.plot(range(1, max_m + 1), variances, 'o-', color='crimson', linewidth=2)
plt.axvline(optimal_m, color='gray', linestyle='--', label=f'Оптимальне m={optimal_m}')
plt.title('Залежність дисперсії від степеня m')
plt.xlabel('Степінь многочлена (m)')
plt.ylabel('Дисперсія δ')
plt.grid(True)
plt.legend()


plt.subplot(2, 2, 2)
plt.scatter(x_data, y_data, color='black', label='Фактичні дані', zorder=5)
plt.plot(x_fine, y_fine_approx, color='blue', label=f'Апроксимація (m={optimal_m})')
plt.plot(x_future, y_future, 'ro--', label='Прогноз (25-27 міс.)')
plt.title('Апроксимація МНК та прогноз')
plt.xlabel('Місяць')
plt.ylabel('Температура (°C)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, (3, 4))
plt.stem(x_data, error_nodes, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title(f'Абсолютна похибка ε(x) = |f(x) - φ(x)| для m={optimal_m}')
plt.xlabel('Місяць')
plt.ylabel('Модуль похибки')
plt.grid(True)

plt.tight_layout()
plt.show()