import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)

def divided_differences(x, y):
    """Побудова таблиці розділених різниць."""
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
            
    return coef

def omega_k(x_val, x_nodes, k):
    """Обчислення w_k(x) = (x - x0)(x - x1)...(x - x_k)."""
    prod = 1.0
    for i in range(k + 1):
        prod *= (x_val - x_nodes[i])
    return prod

def newton_interpolation(x_val, x_nodes, coef):
    """Обчислення значення многочлена Ньютона у точці x_val."""
    n = len(x_nodes)
    result = coef[0, 0]
    for k in range(1, n):
        w = omega_k(x_val, x_nodes, k - 1)
        result += coef[0, k] * w
    return result

def factorial_polynomial_interpolation(x_val, x_nodes, y_nodes):
    """
    Інтерполяція факторіальними многочленними (через скінченні різниці).
    Для роботи робиться заміна t = (x - x0) / h.
    """
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    t = (x_val - x_nodes[0]) / h
    
    # Таблиця скінченних різниць Delta^k y0
    diffs = np.zeros((n, n))
    diffs[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            diffs[i][j] = diffs[i+1][j-1] - diffs[i][j-1]
            
    result = diffs[0, 0]
    t_factorial = 1.0
    fact = 1.0
    
    for k in range(1, n):
        t_factorial *= (t - (k - 1))
        fact *= k
        result += (diffs[0, k] / fact) * t_factorial
        
    return result


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data.csv")
    
    x_data, y_data = read_data(data_path)
    coef_matrix = divided_differences(x_data, y_data)
    
    print("--- Таблиця розділених різниць: ---")
    print(coef_matrix[0, :])
    target_x = 15000.0
    cost_newton = newton_interpolation(target_x, x_data, coef_matrix)
    
    print(f"\nПрогнозована вартість для {target_x:.0f} tasks (Метод Ньютона): {cost_newton:.4f} $")
    
    # Факторіальний метод 
    x_eq = np.linspace(x_data.min(), x_data.max(), len(x_data))
    y_eq = np.interp(x_eq, x_data, y_data) # апроксимація для рівномірної сітки
    cost_fact = factorial_polynomial_interpolation(target_x, x_eq, y_eq)
    print(f"Прогнозована вартість для {target_x:.0f} tasks (Факторіальний метод): {cost_fact:.4f} $")

    # Побудова графіка
    x_dense = np.linspace(x_data.min(), x_data.max(), 300)
    y_newton_dense = [newton_interpolation(x, x_data, coef_matrix) for x in x_dense]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, y_newton_dense, 'b-', label='Інтерполяційна крива Ньютона')
    plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label='Експериментальні точки')
    plt.scatter([target_x], [cost_newton], color='green', marker='*', s=150, zorder=6, label=f'Прогноз (15000 tasks) = ${cost_newton:.2f}')
    
    plt.title('Варіант 4: Модель Cost = f(tasks)')
    plt.xlabel('Tasks (Кількість задач)')
    plt.ylabel('Cost ($)')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nДослідження впливу кількості вузлів (n = 5, 10, 20)...")
    def true_cost_function(x):
        return 0.00035 * x + 0.00000001 * (x**1.3) + 0.1

    nodes_counts = [5, 10, 20]
    plt.figure(figsize=(12, 7))
    
    x_test_dense = np.linspace(1000, 20000, 500)
    y_true_dense = true_cost_function(x_test_dense)
    
    plt.plot(x_test_dense, y_true_dense, 'k--', label='Еталонна функція', linewidth=2)
    
    for n_nodes in nodes_counts:
        x_n = np.linspace(1000, 20000, n_nodes)
        y_n = true_cost_function(x_n)
        
        coef_n = divided_differences(x_n, y_n)
        y_poly = [newton_interpolation(x, x_n, coef_n) for x in x_test_dense]
        
        plt.plot(x_test_dense, y_poly, label=f'Ньютон ({n_nodes} вузлів)')
        
    plt.ylim(-2, 12) 
    plt.title('Дослідження точності та Ефекту Рунге при збільшенні кількості вузлів')
    plt.xlabel('Tasks')
    plt.ylabel('Cost ($)')
    plt.grid(True)
    plt.legend()
    plt.show()