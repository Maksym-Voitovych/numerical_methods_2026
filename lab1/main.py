import requests
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Запит до Open-Elevation API 
# -------------------------------
url = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|"
    "48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|"
    "48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|"
    "48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|"
    "48.160250,24.500106"
)

response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)

with open("tabulation.txt", "w", encoding="utf-8") as f:
    f.write(f"Кількість вузлів: {n}\n")
    f.write("№  | Latitude  | Longitude | Elevation (m)\n")
    f.write("-" * 43 + "\n")
    for i, point in enumerate(results):
        f.write(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}\n")

# 4. Обчислення кумулятивної відстані
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

distances = [0.0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

x = np.array(distances)
y = np.array(elevations)

# 7. Метод прогонки
def sweep_method(alpha, beta, gamma, delta):
    size = len(delta)
    A = np.zeros(size - 1)
    B = np.zeros(size - 1)
    
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, size - 1):
        denom = alpha[i-1] * A[i-1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i-1] * B[i-1]) / denom
        
    sol = np.zeros(size)
    sol[-1] = (delta[-1] - alpha[-1] * B[-2]) / (alpha[-1] * A[-2] + beta[-1])
    for i in range(size - 2, -1, -1):
        sol[i] = A[i] * sol[i+1] + B[i]
        
    return sol

# 6, 8, 9. Побудова сплайна
def build_spline(x_nodes, y_nodes):
    num_pts = len(x_nodes)
    h_arr = np.diff(x_nodes)
    m_pts = num_pts - 2
    
    alpha = np.zeros(m_pts - 1)
    beta = np.zeros(m_pts)
    gamma = np.zeros(m_pts - 1)
    delta = np.zeros(m_pts)
    
    for i in range(m_pts):
        idx = i + 1  
        beta[i] = 2 * (h_arr[idx-1] + h_arr[idx])
        delta[i] = 3 * ((y_nodes[idx+1] - y_nodes[idx]) / h_arr[idx] - (y_nodes[idx] - y_nodes[idx-1]) / h_arr[idx-1])
        if i > 0:
            alpha[i-1] = h_arr[idx-1]
        if i < m_pts - 1:
            gamma[i] = h_arr[idx]

    c_int = sweep_method(alpha, beta, gamma, delta)
    c_arr = np.zeros(num_pts)
    c_arr[1:-1] = c_int

    a_arr = y_nodes[:-1]
    b_arr = np.zeros(num_pts - 1)
    d_arr = np.zeros(num_pts - 1)

    for i in range(num_pts - 1):
        d_arr[i] = (c_arr[i+1] - c_arr[i]) / (3 * h_arr[i])
        b_arr[i] = (y_nodes[i+1] - y_nodes[i]) / h_arr[i] - (h_arr[i] / 3) * (c_arr[i+1] + 2 * c_arr[i])

    return a_arr, b_arr, c_arr[:-1], d_arr

# Обчислення коефіцієнтів для повного набору
a, b, c, d = build_spline(x, y)

print("\n--- КОЕФІЦІЄНТИ КУБІЧНИХ СПЛАЙНІВ ---")
print(" i |      a_i     |      b_i     |      c_i     |      d_i")
print("-" * 62)
for i in range(len(a)):
    print(f"{i:2d} | {a[i]:12.4f} | {b[i]:12.6f} | {c[i]:12.6f} | {d[i]:12.8f}")

# Обчислення значень сплайна у точках
def evaluate_spline(x_grid, a_arr, b_arr, c_arr, d_arr, x_eval):
    y_eval = np.zeros_like(x_eval)
    dy_eval = np.zeros_like(x_eval)
    
    for k, x_val in enumerate(x_eval):
        if x_val <= x_grid[0]:
            i = 0
        elif x_val >= x_grid[-1]:
            i = len(a_arr) - 1
        else:
            i = np.searchsorted(x_grid, x_val) - 1
            i = min(i, len(a_arr) - 1)
            
        dx = x_val - x_grid[i]
        y_eval[k] = a_arr[i] + b_arr[i]*dx + c_arr[i]*(dx**2) + d_arr[i]*(dx**3)
        dy_eval[k] = b_arr[i] + 2*c_arr[i]*dx + 3*d_arr[i]*(dx**2)
        
    return y_eval, dy_eval

# 10, 11, 12. Побудова графіків
xx = np.linspace(x[0], x[-1], 500)
yy_full, dy_full = evaluate_spline(x, a, b, c, d, xx)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(x, y, 'ro', label='Вихідні вузли $y=f(x)$')
plt.plot(xx, yy_full, 'b-', label=r'Інтерполяція $y_{набл}$')
plt.title('Інтерполяція профілю висоти кубічним сплайном')
plt.xlabel('Відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
colors = ['green', 'orange', 'purple']
for num_nodes, col in zip([10, 15, 20], colors):
    indices = np.linspace(0, len(x) - 1, num_nodes, dtype=int)
    x_sub, y_sub = x[indices], y[indices]
    a_s, b_s, c_s, d_s = build_spline(x_sub, y_sub)
    yy_s, _ = evaluate_spline(x_sub, a_s, b_s, c_s, d_s, xx)
    plt.plot(xx, yy_s, label=f'Сплайн ({num_nodes} вузлів)', color=col)
    plt.plot(x_sub, y_sub, 'o', color=col)

plt.title('Порівняння інтерполяції для 10, 15 та 20 вузлів')
plt.xlabel('Відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
indices_10 = np.linspace(0, len(x) - 1, 10, dtype=int)
x_10, y_10 = x[indices_10], y[indices_10]
a_10, b_10, c_10, d_10 = build_spline(x_10, y_10)
yy_10, _ = evaluate_spline(x_10, a_10, b_10, c_10, d_10, xx)
error = np.abs(yy_full - yy_10)

plt.plot(xx, error, 'r-', label=r'Похибка $\epsilon = |y_{full} - y_{10}|$')
plt.title('Графік похибки інтерполяції при зменшенні деталізації до 10 вузлів')
plt.xlabel('Відстань (м)')
plt.ylabel('Абсолютна похибка (м)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Додатково: Характеристика маршруту
total_ascent = sum(max(y[i] - y[i-1], 0) for i in range(1, n))
total_descent = sum(max(y[i-1] - y[i], 0) for i in range(1, n))
grad_full = dy_full * 100
energy = 80 * 9.81 * total_ascent

print("\n--- ХАРАКТЕРИСТИКИ МАРШРУТУ ---")
print(f"Загальна довжина маршруту (м): {x[-1]:.2f}")
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")