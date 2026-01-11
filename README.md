# SLAE Solver & Comparison / Решатель СЛАУ и Сравнение Методов

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English Documentation

### Description
This project implements the **Square Root Method (Cholesky Decomposition)** for solving Systems of Linear Algebraic Equations (SLAE). The method is specifically designed for Symmetric Positive Definite (SPD) matrices and decomposes the matrix as $A = L L^T$, where $L$ is a lower triangular matrix.

### Features
1.  **Square Root Method (Cholesky):** 
    *   Decomposes SPD matrix $A$ as $A = L L^T$
    *   Solves the system using forward and backward substitution
    *   Efficient for SPD matrices with $O(n^3/3)$ operations
2.  **Analysis:**
    *   Reports **execution time** and **residual norm** ($||Ax - b||_2$)
    *   Validates that the input matrix is symmetric and positive definite

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

# 3. Load matrix A and vector b from a JSON file
python Square-Root-Method.py --json input.json

# 4. Show help message
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
Этот проект реализует **Метод Квадратного Корня (Разложение Холецкого)** для решения Систем Линейных Алгебраических Уравнений (СЛАУ). Метод предназначен для симметричных положительно определенных матриц и разлагает матрицу как $A = L L^T$, где $L$ — нижняя треугольная матрица.

### Возможности
1.  **Метод Квадратного Корня (Холецкого):**
    *   Разлагает SPD-матрицу $A$ как $A = L L^T$
    *   Решает систему с помощью прямой и обратной подстановки
    *   Эффективен для SPD-матриц с $O(n^3/3)$ операциями
2.  **Анализ:**
    *   Отображает **время выполнения** и **норму невязки** ($||Ax - b||_2$)
    *   Проверяет, что входная матрица симметрична и положительно определена

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

# 3. Загрузка матрицы A и вектора b из JSON файла
python Square-Root-Method.py --json input.json

# 4. Показать справку
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