"""Analyzes memory access patterns within loops to classify stride and locality behavior."""

import operator
from dataclasses import dataclass
from typing import Dict, List, Optional

from numba.core import ir, types

_AFFINE_BINOPS = {operator.add, operator.sub, operator.mul, operator.iadd, operator.isub, operator.imul}


@dataclass
class ArrayAccess:
    """One array reference inside a loop: its per-dimension affine coefficients and stride classification."""

    array: str
    kind: str  # "read" or "write"
    is_affine: bool
    coefficients: Optional[Dict[str, int]]  # innermost-loop-var coeff per dimension, if affine
    classification: str  # "contiguous", "strided-constant", or "irregular"
    dim_coeffs: Optional[List[Dict[str, int]]] = None  # full {loop_var: coeff, "const": c} per dimension
    unit_dim: Optional[int] = None  # index of the array's unit-stride dimension, if the layout is known
    itemsize: Optional[int] = None  # bytes per element, used by cache_model for byte-accurate estimates
    array_shape: Optional[tuple] = None  # concrete (dim0, dim1, ...) extents, when statically known


def _resolve_affine(var_name, loop_vars, func_ir):
    """Returns {loop_var: coeff, ..., 'const': c} if var_name is affine in loop_vars, else None."""
    if var_name in loop_vars:
        return {var_name: 1}
    try:
        defn = func_ir.get_definition(var_name)
    except Exception:
        return None
    if isinstance(defn, ir.Const):
        return {"const": defn.value} if isinstance(defn.value, int) else None
    if isinstance(defn, ir.Expr):
        if defn.op == "cast":
            return _resolve_affine(defn.value.name, loop_vars, func_ir)
        if defn.op == "unary" and defn.fn is operator.neg:
            inner = _resolve_affine(defn.value.name, loop_vars, func_ir)
            return _scale(inner, -1) if inner is not None else None
        if defn.op in ("binop", "inplace_binop") and defn.fn in _AFFINE_BINOPS:
            lhs = _resolve_affine(defn.lhs.name, loop_vars, func_ir)
            rhs = _resolve_affine(defn.rhs.name, loop_vars, func_ir)
            if lhs is None or rhs is None:
                return None
            if defn.fn in (operator.add, operator.iadd):
                return _add(lhs, rhs)
            if defn.fn in (operator.sub, operator.isub):
                return _add(lhs, _scale(rhs, -1))
            if defn.fn in (operator.mul, operator.imul):
                if set(lhs) <= {"const"}:
                    return _scale(rhs, lhs.get("const", 0))
                if set(rhs) <= {"const"}:
                    return _scale(lhs, rhs.get("const", 0))
                return None  # non-linear: variable * variable
    return None  # anything else (getitem, calls, ...) is not statically affine


def _add(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result


def _scale(d, factor):
    return {k: v * factor for k, v in d.items()}


def _index_dims(index_var, func_ir):
    """Splits a (possibly tuple-valued) index variable into one IR var name per array dimension."""
    try:
        defn = func_ir.get_definition(index_var)
    except Exception:
        return [index_var]
    if isinstance(defn, ir.Expr) and defn.op == "build_tuple":
        return [item.name for item in defn.items]
    return [index_var]


def _unit_stride_dim(array_type):
    if getattr(array_type, "layout", None) == "C":
        return array_type.ndim - 1
    if getattr(array_type, "layout", None) == "F":
        return 0
    return None  # unknown/mixed layout: cannot statically prove unit stride


def _classify(dim_coeffs, innermost_var, unit_dim):
    """Classifies an access given each dimension's affine coefficients for the innermost loop var."""
    innermost_coeffs = [d.get(innermost_var, 0) for d in dim_coeffs]
    if unit_dim is not None:
        unit_coeff = innermost_coeffs[unit_dim]
        other_nonzero = any(c != 0 for i, c in enumerate(innermost_coeffs) if i != unit_dim)
        if unit_coeff == 1 and not other_nonzero:
            return "contiguous"
    if any(c != 0 for c in innermost_coeffs):
        return "strided-constant"
    return "strided-constant"  # loop-invariant access: constant (zero) stride


def classify_accesses(loop_nest, func_ir, typemap):
    """Returns one ArrayAccess record per array reference statically found in loop_nest's own body."""
    loop_vars = set(loop_nest.induction_vars)
    innermost_var = loop_nest.induction_vars[-1] if loop_nest.induction_vars else None
    records = []

    for label in loop_nest.own_blocks:
        block = func_ir.blocks[label]
        for stmt in block.body:
            array_name = None
            index_var = None
            kind = None

            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and stmt.value.op == "getitem":
                array_name = stmt.value.value.name
                index_var = stmt.value.index.name
                kind = "read"
            elif isinstance(stmt, ir.SetItem):
                array_name = stmt.target.name
                index_var = stmt.index.name
                kind = "write"

            if array_name is None:
                continue
            array_type = typemap.get(array_name)
            if not isinstance(array_type, types.Array):
                continue

            dim_vars = _index_dims(index_var, func_ir)
            dim_coeffs = [_resolve_affine(v, loop_vars, func_ir) for v in dim_vars]
            bitwidth = getattr(array_type.dtype, "bitwidth", None)
            itemsize = bitwidth // 8 if bitwidth else None

            if any(c is None for c in dim_coeffs) or innermost_var is None:
                records.append(ArrayAccess(array_name, kind, False, None, "irregular", itemsize=itemsize))
                continue

            unit_dim = _unit_stride_dim(array_type)
            classification = _classify(dim_coeffs, innermost_var, unit_dim)
            coefficients = {f"dim{i}": d.get(innermost_var, 0) for i, d in enumerate(dim_coeffs)}
            records.append(
                ArrayAccess(
                    array_name,
                    kind,
                    True,
                    coefficients,
                    classification,
                    dim_coeffs=dim_coeffs,
                    unit_dim=unit_dim,
                    itemsize=itemsize,
                )
            )

    return records
