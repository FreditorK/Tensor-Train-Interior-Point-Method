import os
import re

import numpy as np
import scipy.sparse as sp

_SQRT2 = np.sqrt(2.0)


def _require_sdpap():
    try:
        import scipy

        if not hasattr(scipy, "ix_"):
            scipy.ix_ = np.ix_
        import sdpap
    except ImportError as exc:
        raise ImportError(
            "SDPA baseline requires sdpa-python (sdpap). Install sdpa-python in "
            "this environment, or run sdpa.sh from an environment where sdpap is available."
        ) from exc
    return sdpap


def _sdpa_info_dicts(result):
    for key in ("sdpapinfo", "sdpainfo", "timeinfo"):
        info = result.get(key)
        if isinstance(info, dict):
            yield info


def _sdpa_info_texts(result):
    for key in ("sdpapinfo", "sdpainfo", "timeinfo"):
        info = result.get(key)
        if isinstance(info, bytes):
            try:
                info = info.decode("utf-8", errors="ignore")
            except Exception:
                info = str(info)
        if isinstance(info, str):
            yield info
        elif isinstance(info, (list, tuple)):
            joined = "\n".join(str(item) for item in info)
            if joined:
                yield joined


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def _sdpa_info_value(result, keys, default=np.nan):
    for info in _sdpa_info_dicts(result):
        for key in keys:
            if key in info:
                value = _safe_float(info[key])
                if np.isfinite(value):
                    return value
        lower_map = {str(k).lower(): k for k in info.keys()}
        for key in keys:
            match_key = lower_map.get(str(key).lower())
            if match_key is not None:
                value = _safe_float(info[match_key])
                if np.isfinite(value):
                    return value
    return default


def _sdpa_text_value(result, patterns, default=np.nan):
    last_value = None
    for text in _sdpa_info_texts(result):
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                last_value = match.group(1)
    value = _safe_float(last_value)
    return value if np.isfinite(value) else default


def _sdpa_info_str(result, keys, default=""):
    for info in _sdpa_info_dicts(result):
        for key in keys:
            if key in info:
                value = str(info[key]).strip()
                if value:
                    return value
        lower_map = {str(k).lower(): k for k in info.keys()}
        for key in keys:
            match_key = lower_map.get(str(key).lower())
            if match_key is not None:
                value = str(info[match_key]).strip()
                if value:
                    return value
    return default


def _neg_part_norm_sq(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    neg = np.minimum(values, 0.0)
    return float(np.sum(neg ** 2))


def _sdpa_psd_violation(matrix):
    if matrix is None:
        return np.nan
    sym = 0.5 * (matrix + matrix.T)
    eigs = np.linalg.eigvalsh(sym)
    return _neg_part_norm_sq(eigs)


def sdpa_phase(result):
    value = _sdpa_info_str(result, ("phase", "SDPA.phase", "phasevalue"), default="")
    return value


def require_sdpa_optimal(result):
    phase = sdpa_phase(result)
    if phase != "pdOPT":
        raise RuntimeError(f"SDPA did not converge to pdOPT (phase={phase or 'unknown'})")


def sdpa_dual_error_reported(result):
    value = _sdpa_info_value(
        result,
        (
            "dualError",
            "dual_error",
            "err_d",
            "dualFeas",
            "dual_feas",
            "d.feas",
            "dfeas",
            "SDPA.dualError",
            "SDPA.dual_error",
            "SDPA.err_d",
            "SDPA.dfeas",
        ),
    )
    if not np.isfinite(value):
        value = _sdpa_text_value(
            result,
            [
                r"\bdual\s*error\s*[:=]\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
                r"\bdualerror\s*[:=]\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
                r"\berr[_\s]*d\s*[:=]\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
                r"\bdual\s*feas(?:ibility)?\s*[:=]\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
            ],
        )
    return float(value) if np.isfinite(value) else np.nan


def sdpa_dual_error(result):
    reported = sdpa_dual_error_reported(result)
    if np.isfinite(reported):
        return float(reported) ** 2

    theta_d = _sdpa_info_value(
        result,
        ("thetaD", "theta_d", "SDPA.thetaD", "SDPA.theta_d"),
    )
    if not np.isfinite(theta_d):
        theta_d = _sdpa_text_value(
            result,
            [
                r"^\s*\d+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)",
            ],
        )
    if np.isfinite(theta_d) and theta_d > 0.0:
        return float(theta_d) ** 2

    psd_violation = _sdpa_psd_violation(result.get("z_matrix"))
    ineq_violation = np.nan
    if "y_ineq" in result:
        ineq_violation = _neg_part_norm_sq(result["y_ineq"])

    if np.isfinite(psd_violation) or np.isfinite(ineq_violation):
        total = 0.0
        if np.isfinite(psd_violation):
            total += psd_violation
        if np.isfinite(ineq_violation):
            total += ineq_violation
        return float(total)

    return np.nan


def sdpa_num_iters(result):
    value = _sdpa_info_value(
        result,
        (
            "iter",
            "iteration",
            "numIter",
            "numIterations",
            "nIter",
            "iters",
            "SDPA.iter",
            "SDPA.iteration",
        ),
    )
    if not np.isfinite(value):
        value = _sdpa_text_value(
            result,
            [
                r"\biter(?:ation)?\s*[:=]\s*(\d+)",
                r"\bnum(?:ber)?\s*iter(?:ation)?s?\s*[:=]\s*(\d+)",
            ],
        )
    if not np.isfinite(value):
        value = _sdpa_text_value(
            result,
            [
                r"^\s*(\d+)\s+[-+0-9.]+(?:[eE][+-]?[0-9]+)?\s+[-+0-9.]+(?:[eE][+-]?[0-9]+)?\s+[-+0-9.]+(?:[eE][+-]?[0-9]+)?",
            ],
        )
    return float(value) if np.isfinite(value) else np.nan


def sdpa_duality_gap(result):
    value = _sdpa_info_value(result, ("dualityGap", "duality_gap", "gap"))
    if np.isfinite(value):
        return value
    return float(abs(np.trace(result["x_matrix"] @ result["z_matrix"])))


def _sdpa_num_threads(config):
    for key in ("sdpa_num_threads", "numThreads"):
        if key in config:
            return int(config[key])
    for env_name in ("SDPA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(env_name)
        if value:
            return int(value)
    return None


def sdpa_solver_options(config, gamma_star=None, domain_method="basis", range_method=None):
    option = {
        "print": "display",
        "epsilonDash": float(config.get("sdpa_feas_tol", config.get("sdpa_gap_tol", 1e-5))),
        "epsilonStar": float(config.get("sdpa_gap_tol", 1e-5)),
    }
    num_threads = _sdpa_num_threads(config)
    if num_threads is not None:
        option["numThreads"] = num_threads
    if gamma_star is not None:
        option["gammaStar"] = gamma_star
    if domain_method is not None:
        option["domainMethod"] = domain_method
    if range_method is not None:
        option["rangeMethod"] = range_method
    return option


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
    import scs

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
    sdpap = _require_sdpap()
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
        "epsilonDash": 1e-5,
        "epsilonStar": 1e-5,
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
