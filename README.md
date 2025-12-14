# SLAE Solver & Comparison / Решатель СЛАУ и Сравнение Методов

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English Documentation

### Description
This project implements Algorithm for solving Systems of Linear Algebraic Equations (SLAE). It specifically focuses on the **Square Root Method (Cholesky Decomposition)** for Symmetric Positive Definite (SPD) matrices. It also implements other direct and iterative methods for comparison purposes.

### Features
1.  **Direct Methods:**
    *   **Square Root Method (Cholesky):** $A = L L^T$. Efficient for SPD matrices.
    *   **LU Decomposition:** With partial pivoting ($PA = LU$). General purpose direct solver.
    *   **Library Baseline:** Uses `numpy.linalg.solve` for performance benchmarking.
2.  **Iterative Methods:**
    *   **Jacobi Method:** Splits matrix into Diagonal and Remainder.
    *   **Gauss-Seidel Method:** Uses updated values immediately during iteration.
3.  **Analysis:**
    *   Compares methods based on **execution time**, **residual norm** ($||Ax - b||_2$), and **number of iterations**.
    *   Validates results against a reference solution.

### Usage
#### Requirements
*   Python 3.x
*   Numpy

#### Running the script
The script can be run from the terminal:

```bash
# 1. Run with default demo parameters (Random 5x5 SPD matrix)
python Square-Root-Method.py

# 2. Run with a specific matrix size for the demo
python Square-Root-Method.py --n 10

# 3. Generate a diagonally dominant matrix for the demo (better for iterative convergence)
python Square-Root-Method.py --demo diagdom

# 4. Load matrix A and vector b from a JSON file
python Square-Root-Method.py --json input.json

# 5. Show help message
python Square-Root-Method.py --help
```

### Input Format (JSON)
If you use the `--json` flag, the file must be in the following format:

```json
{
    "A": [
        [4, 1, 2],
        [1, 5, 1],
        [2, 1, 3]
    ],
    "b": [4, 6, 7]
}
```
*   `A`: Square matrix (list of lists).
*   `b`: Right-hand side vector (list).

---

<a name="russian"></a>
## Документация на Русском

### Описание
Этот проект реализует алгоритмы для решения Систем Линейных Алгебраических Уравнений (СЛАУ). Основное внимание уделяется **Методу Квадратного Корня (Разложение Холецкого)** для симметричных положительно определенных матриц. Также реализованы другие прямые и итерационные методы для сравнения.

### Возможности
1.  **Прямые методы:**
    *   **Метод Квадратного Корня (Холецкого):** $A = L L^T$. Эффективен для SPD-матриц.
    *   **LU-разложение:** С частичным выбором ведущего элемента ($PA = LU$). Универсальный прямой метод.
    *   **Библиотечное решение:** Использует `numpy.linalg.solve` в качестве эталона.
2.  **Итерационные методы:**
    *   **Метод Якоби:** Разделяет матрицу на диагональную и остаточную части.
    *   **Метод Гаусса-Зейделя:** Использует обновленные значения переменных немедленно.
3.  **Анализ:**
    *   Сравнение методов по **времени выполнения**, **норме невязки** ($||Ax - b||_2$) и **количеству итераций**.
    *   Проверка результатов относительно эталонного решения.

### Использование
#### Требования
*   Python 3.x
*   Numpy

#### Запуск скрипта
Скрипт запускается из терминала:

```bash
# 1. Запуск с параметрами демо по умолчанию (Случайная SPD матрица 5x5)
python Square-Root-Method.py

# 2. Запуск с указанным размером матрицы для демо
python Square-Root-Method.py --n 10

# 3. Генерация матрицы с диагональным преобладанием (лучше для сходимости итерационных методов)
python Square-Root-Method.py --demo diagdom

# 4. Загрузка матрицы A и вектора b из JSON файла
python Square-Root-Method.py --json input.json

# 5. Показать справку
python Square-Root-Method.py --help
```

### Формат входных данных (JSON)
Если вы используете флаг `--json`, файл должен иметь следующий формат:

```json
{
    "A": [
        [4, 1, 2],
        [1, 5, 1],
        [2, 1, 3]
    ],
    "b": [4, 6, 7]
}
```
*   `A`: Квадратная матрица (список списков).
*   `b`: Вектор правой части (список).