"""Лабораторна робота №6. LU-розклад і уточнення розв'язку СЛАР."""

import math
import random
from pathlib import Path

N = 100
X_VALUE = 2.5
EPS = 1e-14
MAX_ITERATIONS = 20
FOLDER = Path(__file__).resolve().parent


def generate_matrix(n):
    """Генеруємо матрицю з діагональним переважанням для стійкого LU без перестановок."""
    a = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        a[i][i] = math.fsum(abs(a[i][j]) for j in range(n) if j != i) + 10
    return a


def save_matrix(filename, a):
    with open(FOLDER / filename, 'w', encoding='utf-8') as f:
        for row in a:
            f.write(' '.join(f'{value:.17g}' for value in row) + '\n')


def save_vector(filename, v):
    with open(FOLDER / filename, 'w', encoding='utf-8') as f:
        for value in v:
            f.write(f'{value:.17g}\n')


def read_matrix(filename):
    with open(FOLDER / filename, encoding='utf-8') as f:
        return [[float(x) for x in line.split()] for line in f if line.strip()]


def read_vector(filename):
    with open(FOLDER / filename, encoding='utf-8') as f:
        return [float(line) for line in f if line.strip()]


def matrix_vector(a, x):
    return [math.fsum(aij * xj for aij, xj in zip(row, x)) for row in a]


def vector_norm(v):
    return max(abs(value) for value in v)


def lu_decomposition(a):
    """LU за схемою Краута: U має одиниці на головній діагоналі."""
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    u = [[0.0] * n for _ in range(n)]
    for i in range(n):
        u[i][i] = 1.0
        for j in range(i, n):
            l[j][i] = a[j][i] - math.fsum(l[j][k] * u[k][i] for k in range(i))
        if abs(l[i][i]) < 1e-15:
            raise ValueError('Нульовий опорний елемент: LU без перестановок неможливий')
        for j in range(i + 1, n):
            u[i][j] = (a[i][j] - math.fsum(l[i][k] * u[k][j] for k in range(i))) / l[i][i]
    return l, u


def lu_solve(l, u, b):
    """Розв'язуємо Ly=b, потім Ux=y."""
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - math.fsum(l[i][j] * y[j] for j in range(i))) / l[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i] - math.fsum(u[i][j] * x[j] for j in range(i + 1, n))
    return x


def residual(a, x, b):
    # math.fsum обмежує похибку сумування довгих рядків.
    return [math.fsum([bi, *(-aij * xj for aij, xj in zip(row, x))])
            for row, bi in zip(a, b)]


def refine_solution(a, b, l, u, x, eps, max_iterations):
    history = []
    r = residual(a, x, b)
    err = vector_norm(r)
    history.append((0, err, None))
    best_err = err
    no_improvement = 0
    for iteration in range(1, max_iterations + 1):
        if err <= eps:
            break
        delta = lu_solve(l, u, r)
        new_x = [math.fsum((xi, dxi)) for xi, dxi in zip(x, delta)]
        new_r = residual(a, new_x, b)
        new_err = vector_norm(new_r)
        history.append((iteration, new_err, vector_norm(delta)))
        # Зупиняємося також при досягненні межі машинної точності.
        if new_err < best_err:
            best_err = new_err
            no_improvement = 0
        else:
            no_improvement += 1
        if new_x == x or no_improvement >= 2:
            if new_err <= err:
                x, err = new_x, new_err
            break
        x, r, err = new_x, new_r, new_err
    return x, history


def main():
    random.seed(42)  # однакові результати при повторному запуску
    a = generate_matrix(N)
    x_exact = [X_VALUE] * N
    b = matrix_vector(a, x_exact)

    save_matrix('A.txt', a)
    save_vector('B.txt', b)
    a = read_matrix('A.txt')
    b = read_vector('B.txt')

    l, u = lu_decomposition(a)
    save_matrix('L.txt', l)
    save_matrix('U.txt', u)

    x = lu_solve(l, u, b)
    save_vector('X_initial.txt', x)
    initial_error = vector_norm(residual(a, x, b))

    x_refined, history = refine_solution(a, b, l, u, x, EPS, MAX_ITERATIONS)
    save_vector('X_refined.txt', x_refined)

    reconstruction_error = max(
        abs(math.fsum(l[i][k] * u[k][j] for k in range(N)) - a[i][j])
        for i in range(N) for j in range(N)
    )
    exact_error = vector_norm([xi - X_VALUE for xi in x_refined])
    final_error = vector_norm(residual(a, x_refined, b))

    report = [
        'ЛАБОРАТОРНА РОБОТА №6',
        f'Розмірність матриці: {N} x {N}',
        f'Заданий розв’язок: x_i = {X_VALUE}',
        f'Задана точність: {EPS:.0e}',
        f'Похибка відновлення A = LU: {reconstruction_error:.6e}',
        f'Початкова нев’язка ||Ax-B||∞: {initial_error:.6e}',
        '', 'Ітераційне уточнення:',
    ]
    for iteration, err, correction in history:
        text = f'Ітерація {iteration}: нев’язка = {err:.6e}'
        if correction is not None:
            text += f', норма поправки = {correction:.6e}'
        report.append(text)
    report += [
        '', f'Кількість ітерацій уточнення: {len(history) - 1}',
        f'Кінцева нев’язка: {final_error:.6e}',
        f'Максимальне відхилення від x_i = 2.5: {exact_error:.6e}',
        f'Досягнуто точності: {"так" if final_error <= EPS else "ні (межа точності float)"}',
        'Перші 10 компонент розв’язку:',
        *[f'x[{i + 1}] = {x_refined[i]:.16f}' for i in range(min(N, 10))],
    ]
    text = '\n'.join(report)
    print(text)
    (FOLDER / 'results.txt').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
