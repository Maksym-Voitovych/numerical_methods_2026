"""Лабораторна №7. Проста ітерація, Якобі та Гаус–Зейдель (без NumPy)."""
import random
from pathlib import Path

N = 100
X_EXACT = 2.5
X_START = 1.0
EPS = 1e-14
MAX_ITER = 10000
SEED = 26
FOLDER = Path(__file__).resolve().parent


def generate_matrix(n):
    random.seed(SEED)
    a = []
    for i in range(n):
        row = [random.uniform(-0.01, 0.01) if j != i else 0.0 for j in range(n)]
        row[i] = 2.0 + sum(abs(v) for v in row)
        a.append(row)
    return a


def save_matrix(filename, a):
    with open(filename, 'w', encoding='utf-8') as f:
        for row in a:
            f.write(' '.join(f'{v:.17g}' for v in row) + '\n')


def save_vector(filename, values):
    with open(filename, 'w', encoding='utf-8') as f:
        for v in values:
            f.write(f'{v:.17g}\n')


def read_matrix(filename):
    with open(filename, encoding='utf-8') as f:
        return [[float(v) for v in line.split()] for line in f if line.strip()]


def read_vector(filename):
    with open(filename, encoding='utf-8') as f:
        return [float(line) for line in f if line.strip()]


def matvec(a, x):
    return [sum(v * xj for v, xj in zip(row, x)) for row in a]


def vector_norm(x):
    return max(abs(v) for v in x)


def matrix_norm(a):
    return max(sum(abs(v) for v in row) for row in a)


def residual(a, b, x):
    ax = matvec(a, x)
    return vector_norm([ax_i - b_i for ax_i, b_i in zip(ax, b)])


def simple_iteration(a, b, x0, eps, tau, max_iter=MAX_ITER):
    x = x0.copy()
    for k in range(1, max_iter + 1):
        ax = matvec(a, x)
        new_x = [xi + tau * (bi - axi) for xi, bi, axi in zip(x, b, ax)]
        delta = vector_norm([u - v for u, v in zip(new_x, x)])
        r = residual(a, b, new_x)
        if delta <= eps and r <= eps:
            return new_x, k, r, True
        if new_x == x:
            return new_x, k, r, False
        x = new_x
    return x, max_iter, residual(a, b, x), False


def jacobi(a, b, x0, eps, max_iter=MAX_ITER):
    n = len(b)
    x = x0.copy()
    for k in range(1, max_iter + 1):
        new_x = [(b[i] - sum(a[i][j] * x[j] for j in range(n) if j != i)) / a[i][i]
                 for i in range(n)]
        delta = vector_norm([u - v for u, v in zip(new_x, x)])
        r = residual(a, b, new_x)
        if delta <= eps and r <= eps:
            return new_x, k, r, True
        if new_x == x:
            return new_x, k, r, False
        x = new_x
    return x, max_iter, residual(a, b, x), False


def gauss_seidel(a, b, x0, eps, max_iter=MAX_ITER):
    n = len(b)
    x = x0.copy()
    for k in range(1, max_iter + 1):
        new_x = x.copy()
        for i in range(n):
            left = sum(a[i][j] * new_x[j] for j in range(i))
            right = sum(a[i][j] * x[j] for j in range(i + 1, n))
            new_x[i] = (b[i] - left - right) / a[i][i]
        delta = vector_norm([u - v for u, v in zip(new_x, x)])
        r = residual(a, b, new_x)
        if delta <= eps and r <= eps:
            return new_x, k, r, True
        if new_x == x:
            return new_x, k, r, False
        x = new_x
    return x, max_iter, residual(a, b, x), False


def main():
    a = generate_matrix(N)
    x_exact = [X_EXACT] * N
    b = matvec(a, x_exact)
    save_matrix(FOLDER / 'matrix_A.txt', a)
    save_vector(FOLDER / 'vector_B.txt', b)
    a = read_matrix(FOLDER / 'matrix_A.txt')
    b = read_vector(FOLDER / 'vector_B.txt')
    x0 = [X_START] * N

    min_dominance = min(abs(row[i]) - sum(abs(v) for j, v in enumerate(row) if i != j)
                        for i, row in enumerate(a))
    # Для A з додатними діагональними елементами вибираємо tau = 1 / ||A||inf.
    # Оцінка ||I - tau*A||inf < 1 гарантує збіжність простої ітерації.
    tau = 1.0 / matrix_norm(a)
    q = max(abs(1 - tau * row[i]) + tau * sum(abs(v) for j, v in enumerate(row) if j != i)
            for i, row in enumerate(a))

    print('ЛАБОРАТОРНА РОБОТА №7')
    print(f'Розмірність: {N} x {N}')
    print(f'Відомий розв’язок: x_i = {X_EXACT}')
    print(f'Початкове наближення: x_i^(0) = {X_START}')
    print(f'Точність: {EPS:.0e}')
    print(f'Мінімальний запас діагонального переважання: {min_dominance:.6f}')
    print(f'Норма матриці ||A||∞: {matrix_norm(a):.6f}')
    print(f'Параметр простої ітерації tau: {tau:.6f}')
    print(f'Оцінка норми матриці ітерації ||I - tau*A||∞: {q:.6f}')

    methods = [
        ('Проста ітерація', simple_iteration(a, b, x0, EPS, tau)),
        ('Якобі', jacobi(a, b, x0, EPS)),
        ('Зейдель', gauss_seidel(a, b, x0, EPS)),
    ]
    lines = []
    for name, (x, k, r, success) in methods:
        err = vector_norm([v - X_EXACT for v in x])
        print(f'\nМетод: {name}')
        print(f'Кількість ітерацій: {k}')
        print(f'Нев’язка ||Ax-B||∞: {r:.6e}')
        print(f'Максимальна похибка: {err:.6e}')
        print(f'Досягнута задана точність: {"так" if success else "ні (межа float або ліміт)"}')
        print('Перші 10 компонент:', ', '.join(f'{v:.12f}' for v in x[:10]))
        save_vector(FOLDER / {'Проста ітерація':'solution_simple.txt', 'Якобі':'solution_jacobi.txt',
                              'Зейдель':'solution_seidel.txt'}[name], x)
        lines.append(f'{name}: ітерацій = {k}; нев’язка = {r:.17g}; похибка = {err:.17g}; точність = {success}')
    (FOLDER / 'results.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
