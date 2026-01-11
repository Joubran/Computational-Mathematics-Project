#!/usr/bin/env python3
"""
SLAE Course Project
Implements:
  Direct method: Square Root Method (Cholesky) for SPD matrices

Input options:
  - Load A and b from a JSON file
  - Generate demo matrices (SPD)

Output:
  - Solution from square root method
  - Runtime and residual norm

Author: (you)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

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
# Method runner
# ----------------------------

def run_method(name: str, func, A: np.ndarray, b: np.ndarray) -> SolveResult:
    t0 = time.perf_counter()
    try:
        x = func(A, b)
        t1 = time.perf_counter()
        resn = residual_norm(A, x, b)
        return SolveResult(name=name, x=x, ok=True, message="OK", time_sec=t1 - t0,
                           residual_norm=resn)
    except Exception as e:
        t1 = time.perf_counter()
        return SolveResult(name=name, x=None, ok=False, message=f"Error: {e}", time_sec=t1 - t0)


def print_report(result: SolveResult) -> None:
    print("\n=== Solution ===")
    print(f"\n[{result.name}]")
    print(f"Status: {'OK' if result.ok else 'FAIL'}")
    print(f"Message: {result.message}")
    print(f"Time (s): {result.time_sec:.6f}")

    if result.residual_norm is not None:
        print(f"Residual norm ||Ax-b||_2: {result.residual_norm:.3e}")

    if result.ok and result.x is not None:
        print(f"x: {np.array2string(result.x, precision=6, suppress_small=False)}")


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


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SLAE course project: Square Root Method solver.")
    parser.add_argument("--json", type=str, default=None, help="Path to JSON with A and b.")
    parser.add_argument("--n", type=int, default=5, help="Size for demo matrix.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for demo matrix.")
    args = parser.parse_args()

    if args.json:
        A, b = load_json(args.json)
    else:
        A, b = make_spd_demo(args.n, args.seed)

    print("=== Input ===")
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")
    print(f"Symmetric: {is_symmetric(A)}")
    print(f"SPD (Cholesky applicable): {is_spd(A)}")

    # Square root method (Cholesky)
    result = run_method("Square Root Method (Cholesky)", solve_cholesky, A, b)
    print_report(result)


if __name__ == "__main__":
    main()
