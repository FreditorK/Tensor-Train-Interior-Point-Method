import numpy as np
import scipy.sparse as sp
import scs
import sdpap

_SQRT2 = np.sqrt(2.0)


def _tril_index(i, j, n):
    return (j * (2 * n - j + 1)) // 2 + (i - j)


def pack_scs_symmetric(matrix):
    n = matrix.shape[0]
    packed = np.zeros(n * (n + 1) // 2)
    for j in range(n):
        for i in range(j, n):
            idx = _tril_index(i, j, n)
            val = matrix[i, j]
            packed[idx] = val if i == j else _SQRT2 * val
    return packed


def unpack_scs_symmetric(packed, n):
    matrix = np.zeros((n, n))
    for j in range(n):
        for i in range(j, n):
            idx = _tril_index(i, j, n)
            val = packed[idx] if i == j else packed[idx] / _SQRT2
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix


def scs_row_from_entries(size, entries):
    row = {}
    for i, j, coef in entries:
        if i < j:
            i, j = j, i
        idx = _tril_index(i, j, size)
        val = float(coef) if i == j else float(coef) / _SQRT2
        row[idx] = row.get(idx, 0.0) + val
    return row


def solve_scs_psd_max(c_matrix, eq_rows, eq_rhs, ineq_rows=None, ineq_rhs=None, eps=1e-5, verbose=True):
    size = c_matrix.shape[0]
    nvar = size * (size + 1) // 2
    eq_rows = eq_rows or []
    eq_rhs = np.asarray(eq_rhs if eq_rhs is not None else np.zeros(len(eq_rows)), dtype=float)
    ineq_rows = ineq_rows or []
    ineq_rhs = np.asarray(ineq_rhs if ineq_rhs is not None else np.zeros(len(ineq_rows)), dtype=float)

    m_eq = len(eq_rows)
    m_ineq = len(ineq_rows)
    m = m_eq + m_ineq + nvar

    rows = []
    cols = []
    vals = []
    b = np.zeros(m)

    for r, row in enumerate(eq_rows):
        for c_idx, coef in row.items():
            rows.append(r)
            cols.append(c_idx)
            vals.append(coef)
        b[r] = eq_rhs[r]

    for r0, row in enumerate(ineq_rows):
        r = m_eq + r0
        for c_idx, coef in row.items():
            rows.append(r)
            cols.append(c_idx)
            vals.append(coef)
        b[r] = ineq_rhs[r0]

    for k in range(nvar):
        rows.append(m_eq + m_ineq + k)
        cols.append(k)
        vals.append(-1.0)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(m, nvar)).tocsc()
    c = -pack_scs_symmetric(c_matrix)
    cone = {"z": m_eq, "l": m_ineq, "s": [size]}
    sol = scs.solve(
        {"A": A, "b": b, "c": c},
        cone,
        eps_abs=eps,
        eps_rel=eps,
        verbose=verbose,
        use_indirect=True,
    )

    x = sol.get("x")
    y = sol.get("y")
    if x is None or y is None:
        raise RuntimeError(f"SCS did not return primal/dual solution (status={sol.get('info', {}).get('status')})")

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    x_matrix = unpack_scs_symmetric(x, size)
    z_packed = y[m_eq + m_ineq:]
    z_matrix = unpack_scs_symmetric(z_packed, size)

    return {
        "A": A,
        "b": b,
        "c": c,
        "x_matrix": x_matrix,
        "y_eq": y[:m_eq],
        "y_ineq": y[m_eq:m_eq + m_ineq],
        "z_matrix": z_matrix,
        "sol": sol,
    }


def _full_idx(i, j, n):
    return i + j * n


def sdpa_row_from_entries(size, entries):
    row = {}
    for i, j, coef in entries:
        idx = _full_idx(i, j, size)
        row[idx] = row.get(idx, 0.0) + float(coef)
    return row


def solve_sdpa_psd_max(c_matrix, eq_rows, eq_rhs, ineq_rows=None, ineq_rhs=None, option=None):
    size = c_matrix.shape[0]
    nvar = size * size
    eq_rows = eq_rows or []
    eq_rhs = np.asarray(eq_rhs if eq_rhs is not None else np.zeros(len(eq_rows)), dtype=float)
    ineq_rows = ineq_rows or []
    ineq_rhs = np.asarray(ineq_rhs if ineq_rhs is not None else np.zeros(len(ineq_rows)), dtype=float)

    m_eq = len(eq_rows)
    m_ineq = len(ineq_rows)
    m = m_eq + m_ineq

    rows = []
    cols = []
    vals = []
    b = np.zeros(m)

    for r, row in enumerate(eq_rows):
        for c_idx, coef in row.items():
            rows.append(r)
            cols.append(c_idx)
            vals.append(coef)
        b[r] = eq_rhs[r]

    for r0, row in enumerate(ineq_rows):
        r = m_eq + r0
        for c_idx, coef in row.items():
            rows.append(r)
            cols.append(c_idx)
            vals.append(coef)
        b[r] = ineq_rhs[r0]

    A = sp.coo_matrix((vals, (rows, cols)), shape=(m, nvar)).tocsc()
    c = -c_matrix.reshape(-1, order="F")

    K = sdpap.SymCone(s=(size,))
    J = sdpap.SymCone(f=m_eq, l=m_ineq)

    sdpa_option = {
        "print": "display",
        "epsilonDash": 1e-8,
        "epsilonStar": 1e-7,
        "domainMethod": "none",
        "rangeMethod": "none",
    }
    if option:
        sdpa_option.update(option)

    x, y, sdpapinfo, timeinfo, sdpainfo = sdpap.solve(A, b, c, K, J, option=sdpa_option)

    x_vec = np.asarray(x.todense()).reshape(-1)
    y_vec = np.asarray(y.todense()).reshape(-1)
    x_matrix = x_vec.reshape((size, size), order="F")

    dual_slack = c - A.T @ y_vec
    z_matrix = dual_slack.reshape((size, size), order="F")

    return {
        "A": A,
        "b": b,
        "c": c,
        "x_matrix": x_matrix,
        "y_eq": y_vec[:m_eq],
        "y_ineq": y_vec[m_eq:m_eq + m_ineq],
        "z_matrix": z_matrix,
        "sdpapinfo": sdpapinfo,
        "timeinfo": timeinfo,
        "sdpainfo": sdpainfo,
    }
