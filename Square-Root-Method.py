#!/usr/bin/env python3
"""
SLAE Course Project
Implements:
  1) Direct method (assigned): Square Root Method (Cholesky) for SPD matrices
  2) Alternative direct method: LU decomposition with partial pivoting
  3) Library baseline ("Gauss from a suitable library"): numpy.linalg.solve
  4) Iterative method (proposed): Jacobi
  5) Alternative iterative method: Gauss-Seidel
  6) Comparison by criteria: runtime, residual norm, iterations, convergence status

Input options:
  - Load A and b from a JSON file
  - Generate demo matrices (SPD, diagonally dominant)

Output:
  - Solutions from each method
  - Comparison report

Author: (you)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np


# ----------------------------
# Utility structures
# ----------------------------

@dataclass
class SolveResult:
    name: str
    x: Optional[np.ndarray]
    ok: bool
    message: str
    time_sec: float
    residual_norm: Optional[float] = None
    iterations: Optional[int] = None


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    r = A @ x - b
    return float(np.linalg.norm(r, ord=2))


def is_symmetric(A: np.ndarray, tol: float = 1e-12) -> bool:
    return float(np.linalg.norm(A - A.T, ord=np.inf)) <= tol


def is_spd(A: np.ndarray) -> bool:
    """
    SPD check via attempting Cholesky factorization.
    This is a practical check for the project.
    """
    if not is_symmetric(A):
        return False
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


# ----------------------------
# 1) Direct method: Square Root (Cholesky)
# ----------------------------

def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Computes lower triangular L such that A = L L^T
    for symmetric positive definite matrices A.

    This is the "square root method" used in many courses.
    """
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]

            if i == j:
                val = A[i, i] - s
                if val <= 0.0:
                    raise ValueError("Matrix is not positive definite (breakdown in Cholesky).")
                L[i, j] = math.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]
    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]
    return x


def solve_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b using square root method:
      A = L L^T
      L y = b
      L^T x = y
    """
    L = cholesky_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(L.T, y)
    return x


# ----------------------------
# 2) Alternative direct method: LU with partial pivoting
# ----------------------------

def lu_decomposition_partial_pivot(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes P, L, U such that P A = L U using partial pivoting.
    """
    A = A.astype(float).copy()
    n = A.shape[0]

    P = np.eye(n)
    L = np.zeros((n, n), dtype=float)
    U = A

    for k in range(n):
        # Pivot selection
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < 1e-15:
            raise ValueError("Matrix is singular or nearly singular (LU pivot too small).")

        # Swap rows in U
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            # swap already built part of L
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        L[k, k] = 1.0

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            U[i, k] = 0.0

    return P, L, U


def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    P, L, U = lu_decomposition_partial_pivot(A)
    # Solve P A x = P b  -> L U x = P b
    pb = P @ b
    y = forward_substitution(L, pb)
    x = backward_substitution(U, y)
    return x


# ----------------------------
# 4) Iterative method: Jacobi
# ----------------------------

def solve_jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 10_000
) -> Tuple[np.ndarray, int, bool, str]:
    """
    Jacobi iteration:
      x^{k+1} = D^{-1} (b - (L+U) x^k)

    Convergence is not guaranteed unless conditions hold
    (for example, A diagonally dominant or SPD in some cases).
    """
    n = A.shape[0]
    x = np.zeros(n, dtype=float) if x0 is None else x0.astype(float).copy()

    D = np.diag(A)
    if np.any(np.abs(D) < 1e-15):
        return x, 0, False, "Jacobi failed: zero on diagonal."

    R = A - np.diagflat(D)

    for k in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x, ord=np.inf) <= tol:
            return x_new, k, True, "Converged."
        x = x_new

    return x, max_iter, False, "Max iterations reached."


# ----------------------------
# 5) Alternative iterative method: Gauss-Seidel
# ----------------------------

def solve_gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 10_000
) -> Tuple[np.ndarray, int, bool, str]:
    """
    Gauss-Seidel iteration:
      update x in-place using the latest available values.
    """
    n = A.shape[0]
    x = np.zeros(n, dtype=float) if x0 is None else x0.astype(float).copy()

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                return x, k - 1, False, "Gauss-Seidel failed: zero on diagonal."

            s1 = float(A[i, :i] @ x[:i])
            s2 = float(A[i, i + 1:] @ x_old[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) <= tol:
            return x, k, True, "Converged."

    return x, max_iter, False, "Max iterations reached."


# ----------------------------
# 3) Library baseline: numpy.linalg.solve
# ----------------------------

def solve_library(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, b)


# ----------------------------
# Comparison runner
# ----------------------------

def run_method(name: str, func, A: np.ndarray, b: np.ndarray) -> SolveResult:
    t0 = time.perf_counter()
    try:
        out = func(A, b)
        t1 = time.perf_counter()

        # handle iterative return shape
        if isinstance(out, tuple):
            x = out[0]
            iters = int(out[1])
            ok = bool(out[2])
            msg = str(out[3])
            resn = residual_norm(A, x, b) if ok else None
            return SolveResult(name=name, x=x, ok=ok, message=msg, time_sec=t1 - t0,
                               residual_norm=resn, iterations=iters)

        x = out
        resn = residual_norm(A, x, b)
        return SolveResult(name=name, x=x, ok=True, message="OK", time_sec=t1 - t0,
                           residual_norm=resn)
    except Exception as e:
        t1 = time.perf_counter()
        return SolveResult(name=name, x=None, ok=False, message=f"Error: {e}", time_sec=t1 - t0)


def print_report(results: List[SolveResult], x_ref: Optional[np.ndarray]) -> None:
    print("\n=== Solutions and Comparison ===")
    for r in results:
        print(f"\n[{r.name}]")
        print(f"Status: {'OK' if r.ok else 'FAIL'}")
        print(f"Message: {r.message}")
        print(f"Time (s): {r.time_sec:.6f}")

        if r.iterations is not None:
            print(f"Iterations: {r.iterations}")

        if r.residual_norm is not None:
            print(f"Residual norm ||Ax-b||_2: {r.residual_norm:.3e}")

        if r.ok and r.x is not None:
            print(f"x: {np.array2string(r.x, precision=6, suppress_small=False)}")
            if x_ref is not None:
                err = float(np.linalg.norm(r.x - x_ref, ord=2))
                print(f"Error vs reference ||x - x_ref||_2: {err:.3e}")

    print("\n=== Ranking by time (fastest first) ===")
    ok_results = [r for r in results if r.ok]
    ok_results.sort(key=lambda r: r.time_sec)
    for r in ok_results:
        print(f"{r.name:20s} {r.time_sec:.6f} s")


# ----------------------------
# Input helpers
# ----------------------------

def load_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    JSON format:
      {
        "A": [[...], [...], ...],
        "b": [...]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    A = np.array(data["A"], dtype=float)
    b = np.array(data["b"], dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must be a vector with same size as A.")

    return A, b


def make_spd_demo(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a random SPD matrix A and random b.
    A = M^T M + alpha I
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    A = M.T @ M + (n * 0.5) * np.eye(n)
    b = rng.normal(size=n)
    return A, b


def make_diag_dominant_demo(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonally dominant matrices are good for iterative convergence.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + 1.0
    b = rng.normal(size=n)
    return A, b


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SLAE course project solver and comparer.")
    parser.add_argument("--json", type=str, default=None, help="Path to JSON with A and b.")
    parser.add_argument("--demo", type=str, choices=["spd", "diagdom"], default="spd",
                        help="Which demo matrix to generate if --json is not provided.")
    parser.add_argument("--n", type=int, default=5, help="Size for demo matrix.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for demo matrix.")
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance for iterative methods.")
    parser.add_argument("--max-iter", type=int, default=10000, help="Max iterations for iterative methods.")
    args = parser.parse_args()

    if args.json:
        A, b = load_json(args.json)
    else:
        if args.demo == "spd":
            A, b = make_spd_demo(args.n, args.seed)
        else:
            A, b = make_diag_dominant_demo(args.n, args.seed)

    print("=== Input ===")
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")
    print(f"Symmetric: {is_symmetric(A)}")
    print(f"SPD (Cholesky applicable): {is_spd(A)}")

    # Reference solution from library (baseline)
    ref = run_method("Library solve (numpy.linalg.solve)", solve_library, A, b)
    x_ref = ref.x if ref.ok else None

    # 1) Proposed direct method: square root method (Cholesky)
    chol = run_method("Direct: Square Root (Cholesky)", solve_cholesky, A, b)

    # 2) Alternative direct method: LU with partial pivoting
    lu = run_method("Direct: LU (partial pivoting)", solve_lu, A, b)

    # 4) Proposed iterative method: Jacobi
    jacobi = run_method(
        "Iterative: Jacobi",
        lambda AA, bb: solve_jacobi(AA, bb, tol=args.tol, max_iter=args.max_iter),
        A,
        b
    )

    # 5) Alternative iterative method: Gauss-Seidel
    gs = run_method(
        "Iterative: Gauss-Seidel",
        lambda AA, bb: solve_gauss_seidel(AA, bb, tol=args.tol, max_iter=args.max_iter),
        A,
        b
    )

    results = [ref, chol, lu, jacobi, gs]
    print_report(results, x_ref)


if __name__ == "__main__":
    main()
