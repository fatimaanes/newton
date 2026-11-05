# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental compact Delassus matrices and Cholesky solves."""

from functools import cache
from typing import Any

import numpy as np
import warp as wp

from ..kinematics.constraints import get_max_constraints_per_world
from ..linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LinearSolverNameToType
from .delassus import DelassusOperator

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class CompactDelassusData:
    """Device arrays shared by compact assembly, factorization and solves."""

    dim: wp.array[wp.int32]
    vio: wp.array[wp.int32]
    support: wp.array3d[wp.uint32]
    graph: wp.array3d[wp.uint32]
    filled: wp.array3d[wp.uint32]
    degree: wp.array2d[wp.int32]
    seen: wp.array2d[wp.int32]
    queue: wp.array2d[wp.int32]
    permutation: wp.array2d[wp.int32]
    row_ptr: wp.array2d[wp.int32]
    counts: wp.array[wp.int32]
    scratch: wp.array2d[wp.float32]
    values: wp.array2d[wp.float32]
    factors: wp.array2d[wp.float32]
    columns: wp.array2d[wp.int32]
    entry_rows: wp.array2d[wp.int32]
    diagonal: wp.array[wp.float32]
    P: wp.array[wp.float32]
    sigma: wp.array[wp.vec2f]


@wp.func
def has_edge(graph: wp.array3d[wp.uint32], world: int, row: int, col: int):
    return (graph[world, row, col // 32] & (wp.uint32(1) << wp.uint32(col % 32))) != wp.uint32(0)


@cache
def make_jacobian_kernels(ndim):
    JType = wp.array(dtype=wp.float32, ndim=ndim)

    @wp.kernel(module="unique", module_options={"enable_backward": False, "default_grid_stride": False})
    def body_support(storage: CompactDelassusData, bio: wp.array[wp.int32], jio: wp.array[wp.int32], J_storage: JType):
        w, row, word = wp.tid()
        bits = wp.uint32(0)
        nb = bio[w + 1] - bio[w]
        J = J_storage[w] if wp.static(ndim == 2) else J_storage
        if row < storage.dim[w]:
            for body in range(word * 32, wp.min(word * 32 + 32, nb)):
                for axis in range(6):
                    if J[jio[w] + row * nb * 6 + body * 6 + axis] != 0.0:
                        bits = bits | (wp.uint32(1) << wp.uint32(body % 32))
                        break
        storage.support[w, row, word] = bits

    @wp.kernel(module="unique", module_options={"enable_backward": False, "default_grid_stride": False})
    def assemble(
        storage: CompactDelassusData,
        bio: wp.array[wp.int32],
        jio: wp.array[wp.int32],
        J_storage: JType,
        inv_m: wp.array[wp.float32],
        inv_I: wp.array[wp.mat33f],
        ndyn: wp.array[wp.int32],
        dyn_offset: wp.array[wp.int32],
        inv_m_j: wp.array[wp.float32],
        neffort: wp.array[wp.int32],
        effort_offset: wp.array[wp.int32],
        bcgo: wp.array[wp.int32],
        nfriction: wp.array[wp.int32],
        inv_m_a: wp.array[wp.float32],
    ):
        w, pos = wp.tid()
        if pos >= storage.counts[w]:
            return
        i0 = storage.permutation[w, storage.entry_rows[w, pos]]
        j0 = storage.permutation[w, storage.columns[w, pos]]
        i = wp.min(i0, j0)
        j = wp.max(i0, j0)
        nb = bio[w + 1] - bio[w]
        J = J_storage[w] if wp.static(ndim == 2) else J_storage
        value = wp.float32(0.0)
        for body in range(nb):
            ii = jio[w] + i * nb * 6 + body * 6
            jj = jio[w] + j * nb * 6 + body * 6
            vi = wp.vec3f(J[ii], J[ii + 1], J[ii + 2])
            vj = wp.vec3f(J[jj], J[jj + 1], J[jj + 2])
            wi = wp.vec3f(J[ii + 3], J[ii + 4], J[ii + 5])
            wj = wp.vec3f(J[jj + 3], J[jj + 4], J[jj + 5])
            value += inv_m[bio[w] + body] * wp.dot(vi, vj) + wp.dot(wi, inv_I[bio[w] + body] @ wj)
        if i == j:
            if i < ndyn[w]:
                value += inv_m_j[dyn_offset[w] + i]
            effort = i - bcgo[w] - nfriction[w]
            if effort >= 0 and effort < neffort[w]:
                value += inv_m_a[effort_offset[w] + effort]
            storage.diagonal[storage.vio[w] + i] = value
        storage.values[w, pos] = value

    return body_support, assemble


@wp.kernel
def build_adjacency(storage: CompactDelassusData):
    w, row, word = wp.tid()
    bits = wp.uint32(0)
    if row < storage.dim[w]:
        for col in range(word * 32, wp.min(word * 32 + 32, storage.dim[w])):
            connected = row == col
            for bodyword in range(storage.support.shape[2]):
                shared = storage.support[w, row, bodyword] & storage.support[w, col, bodyword]
                connected = connected or shared != wp.uint32(0)
            if connected:
                bits = bits | (wp.uint32(1) << wp.uint32(col % 32))
    storage.graph[w, row, word] = bits


@wp.kernel
def rcm_order(storage: CompactDelassusData):
    w = wp.tid()
    n = storage.dim[w]
    for row in range(n):
        count = int(0)
        for col in range(n):
            if row != col and has_edge(storage.graph, w, row, col):
                count += 1
        storage.degree[w, row] = count
        storage.seen[w, row] = 0
    tail = int(0)
    head = int(0)
    while tail < n:
        seed = int(-1)
        best = n + 1
        for row in range(n):
            if storage.seen[w, row] == 0 and storage.degree[w, row] < best:
                seed = row
                best = storage.degree[w, row]
        storage.queue[w, tail] = seed
        storage.seen[w, seed] = 1
        tail += 1
        while head < tail:
            row = storage.queue[w, head]
            head += 1
            start = tail
            for col in range(n):
                if storage.seen[w, col] == 0 and has_edge(storage.graph, w, row, col):
                    storage.seen[w, col] = 1
                    # Insert by degree; ties keep column order.
                    pos = tail
                    while pos > start and storage.degree[w, storage.queue[w, pos - 1]] > storage.degree[w, col]:
                        storage.queue[w, pos] = storage.queue[w, pos - 1]
                        pos -= 1
                    storage.queue[w, pos] = col
                    tail += 1
    for row in range(n):
        storage.permutation[w, row] = storage.queue[w, n - 1 - row]


@wp.kernel
def permute_graph(storage: CompactDelassusData):
    w, row, word = wp.tid()
    bits = wp.uint32(0)
    if row < storage.dim[w]:
        for col in range(word * 32, wp.min(word * 32 + 32, storage.dim[w])):
            if has_edge(storage.graph, w, storage.permutation[w, row], storage.permutation[w, col]):
                bits = bits | (wp.uint32(1) << wp.uint32(col % 32))
    storage.filled[w, row, word] = bits


@wp.kernel
def symbolic_cholesky(storage: CompactDelassusData):
    w = wp.tid()
    n = storage.dim[w]
    for pivot in range(n):
        for row in range(pivot + 1, n):
            if has_edge(storage.filled, w, pivot, row):
                for word in range(pivot // 32, storage.filled.shape[2]):
                    bits = storage.filled[w, pivot, word]
                    if word == pivot // 32:
                        low = pivot % 32
                        bits = bits & (~wp.uint32(0) << wp.uint32(low)) & ~(wp.uint32(1) << wp.uint32(low))
                    storage.filled[w, row, word] = storage.filled[w, row, word] | bits
    count = int(0)
    storage.row_ptr[w, 0] = 0
    for row in range(n):
        for col in range(row + 1):
            if has_edge(storage.filled, w, row, col):
                count += 1
        storage.row_ptr[w, row + 1] = count
    storage.counts[w] = count


@wp.kernel
def write_columns(storage: CompactDelassusData):
    w, row = wp.tid()
    if row < storage.dim[w]:
        pos = storage.row_ptr[w, row]
        for col in range(row + 1):
            if has_edge(storage.filled, w, row, col):
                storage.columns[w, pos] = col
                storage.entry_rows[w, pos] = row
                pos += 1


@wp.kernel
def copy_diagonal(storage: CompactDelassusData, target: wp.array[wp.float32]):
    w, row = wp.tid()
    if row < storage.dim[w]:
        target[storage.vio[w] + storage.permutation[w, row]] = storage.values[w, storage.row_ptr[w, row + 1] - 1]


@wp.kernel
def regularize_diagonal(storage: CompactDelassusData, eta: wp.array[wp.float32]):
    w, row = wp.tid()
    if row < storage.dim[w]:
        storage.values[w, storage.row_ptr[w, row + 1] - 1] += eta[storage.vio[w] + storage.permutation[w, row]]


@wp.kernel
def scale_values(storage: CompactDelassusData):
    w, pos = wp.tid()
    if pos < storage.counts[w]:
        i0 = storage.permutation[w, storage.entry_rows[w, pos]]
        j0 = storage.permutation[w, storage.columns[w, pos]]
        i = storage.vio[w] + wp.max(i0, j0)
        j = storage.vio[w] + wp.min(i0, j0)
        storage.values[w, pos] = storage.P[i] * (storage.P[j] * storage.values[w, pos])


@wp.kernel
def add_proximal(storage: CompactDelassusData, status: wp.array[Any]):
    w, row = wp.tid()
    if row < storage.dim[w] and status[w].converged == 0:
        storage.values[w, storage.row_ptr[w, row + 1] - 1] += storage.sigma[w][0] - storage.sigma[w][1]


@wp.kernel
def factorize(storage: CompactDelassusData, dim: wp.array[wp.int32]):
    w = wp.tid()
    for row in range(dim[w]):
        start = storage.row_ptr[w, row]
        end = storage.row_ptr[w, row + 1]
        for pos in range(start, end):
            col = storage.columns[w, pos]
            value = storage.values[w, pos]
            left = start
            right = storage.row_ptr[w, col]
            right_end = storage.row_ptr[w, col + 1] - 1
            while left < pos and right < right_end:
                a = storage.columns[w, left]
                b = storage.columns[w, right]
                if a == b:
                    value -= storage.factors[w, left] * storage.factors[w, right]
                    left += 1
                    right += 1
                elif a < b:
                    left += 1
                else:
                    right += 1
            if col == row:
                storage.factors[w, pos] = wp.sqrt(value)
            else:
                storage.factors[w, pos] = value / storage.factors[w, right_end]


@wp.kernel
def flag_failed(storage: CompactDelassusData, retry: wp.array[wp.int32]):
    w, row = wp.tid()
    if row < storage.dim[w] and (not wp.isfinite(storage.factors[w, storage.row_ptr[w, row + 1] - 1])):
        retry[w] = storage.dim[w]


@wp.kernel
def shift_diagonal(storage: CompactDelassusData, dim: wp.array[wp.int32], shift: wp.float32):
    w, row = wp.tid()
    if row < dim[w]:
        storage.values[w, storage.row_ptr[w, row + 1] - 1] += shift


@wp.kernel
def solve_csr(storage: CompactDelassusData, b: wp.array[wp.float32], x: wp.array[wp.float32]):
    w = wp.tid()
    n = storage.dim[w]
    for row in range(n):
        diag = storage.row_ptr[w, row + 1] - 1
        value = b[storage.vio[w] + storage.permutation[w, row]]
        for pos in range(storage.row_ptr[w, row], diag):
            value -= storage.factors[w, pos] * storage.scratch[w, storage.columns[w, pos]]
        storage.scratch[w, row] = value / storage.factors[w, diag]
    for rev in range(n):
        row = n - 1 - rev
        diag = storage.row_ptr[w, row + 1] - 1
        value = storage.scratch[w, row] / storage.factors[w, diag]
        storage.scratch[w, row] = value
        for pos in range(storage.row_ptr[w, row], diag):
            storage.scratch[w, storage.columns[w, pos]] -= storage.factors[w, pos] * value
        x[storage.vio[w] + storage.permutation[w, row]] = value


@wp.kernel
def multiply_symmetric(
    storage: CompactDelassusData,
    physical: bool,
    alpha: wp.float32,
    beta: wp.float32,
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
):
    w = wp.tid()
    n = storage.dim[w]
    for row in range(n):
        storage.scratch[w, row] = 0.0
    for row in range(n):
        oi = storage.vio[w] + storage.permutation[w, row]
        for pos in range(storage.row_ptr[w, row], storage.row_ptr[w, row + 1]):
            col = storage.columns[w, pos]
            oj = storage.vio[w] + storage.permutation[w, col]
            value = storage.values[w, pos]
            if col == row:
                value -= storage.sigma[w][0]
            if physical:
                value /= storage.P[oi] * storage.P[oj]
            storage.scratch[w, row] += value * x[oj]
            if row != col:
                storage.scratch[w, col] += value * x[oi]
    for row in range(n):
        oi = storage.vio[w] + storage.permutation[w, row]
        value = alpha * storage.scratch[w, row]
        if beta != 0.0:
            value += beta * y[oi]
        y[oi] = value


class CompactCholeskySolver:
    """Solve using the operator's stored matrix entries."""

    def __init__(self, operator, failed_pivot_shift=0.0, **kwargs):
        self.operator = operator
        self._failed_pivot_shift = failed_pivot_shift
        self._retry_dim = wp.zeros(operator.num_worlds, dtype=wp.int32, device=operator._device)

    def reset(self):
        """The next factorization overwrites its stored values."""

    def compute(self, A=None):
        op = self.operator
        op._launch(factorize, op.num_worlds, op.info.dim, block_dim=32)
        if self._failed_pivot_shift > 0.0:
            self._retry_dim.zero_()
            shape = (op.num_worlds, op.max_dimension)
            op._launch(flag_failed, shape, self._retry_dim)
            op._launch(shift_diagonal, shape, self._retry_dim, self._failed_pivot_shift)
            op._launch(factorize, op.num_worlds, self._retry_dim, block_dim=32)
            op._launch(shift_diagonal, shape, self._retry_dim, -self._failed_pivot_shift)

    def solve(self, b, x):
        op = self.operator
        op._launch(solve_csr, op.num_worlds, b, x, block_dim=32)

    def solve_inplace(self, x):
        self.solve(x, x)


class CompactDelassusOperator(DelassusOperator):
    """Store needed matrix entries and rebuild them each step."""

    compact = True

    def finalize(self, model, data, limits=None, contacts=None, solver=None, solver_kwargs=None):
        self._size = model.size
        self._device = model.device
        self._num_worlds = model.size.num_worlds
        self._world_maxdims = get_max_constraints_per_world(model, limits, contacts)
        self._model_maxdims = sum(self._world_maxdims)
        self._world_maxsize = [n * n for n in self._world_maxdims]
        self._model_maxsize = sum(self._world_maxsize)
        self.max_dimension = max(self._world_maxdims, default=0)
        self._max_of_max_total_D_size = self.max_dimension * self.max_dimension
        info = DenseSquareMultiLinearInfo()
        info.assign(
            maxdim=model.info.max_total_cts,
            dim=data.info.num_total_cts,
            vio=model.info.total_cts_offset,
            mio=wp.zeros(self._num_worlds, dtype=wp.int32, device=self._device),
            dtype=wp.float32,
            device=self._device,
        )
        self._operator = DenseLinearOperatorData(info=info)
        storage = self._data = CompactDelassusData()
        storage.dim = info.dim
        storage.vio = info.vio
        nw, n = self._num_worlds, self.max_dimension
        words = (n + 31) // 32
        bodies = np.diff(model.info.bodies_offset.numpy())
        bodywords = (int(bodies.max(initial=0)) + 31) // 32
        self.support = storage.support = wp.zeros((nw, n, bodywords), dtype=wp.uint32, device=self._device)
        self.graph = storage.graph = wp.zeros((nw, n, words), dtype=wp.uint32, device=self._device)
        self.filled = storage.filled = wp.zeros_like(self.graph)
        self.degree = storage.degree = wp.zeros((nw, n), dtype=wp.int32, device=self._device)
        self.seen = storage.seen = wp.zeros_like(self.degree)
        self.queue = storage.queue = wp.zeros_like(self.degree)
        self.permutation = storage.permutation = wp.zeros_like(self.degree)
        self.row_ptr = storage.row_ptr = wp.zeros((nw, n + 1), dtype=wp.int32, device=self._device)
        self.counts = storage.counts = wp.zeros(nw, dtype=wp.int32, device=self._device)
        self.scratch = storage.scratch = wp.zeros((nw, n), dtype=wp.float32, device=self._device)
        self.original_diagonal = storage.diagonal = wp.zeros(self._model_maxdims, dtype=wp.float32, device=self._device)
        self._P = storage.P = wp.ones(self._model_maxdims, dtype=wp.float32, device=self._device)
        self._sigma = storage.sigma = wp.zeros(nw, dtype=wp.vec2f, device=self._device)
        self._allocate_values(0)
        self._solver = CompactCholeskySolver(self, **(solver_kwargs or {}))

    def _allocate_values(self, capacity):
        self._capacity = capacity
        self.values = self._data.values = wp.zeros((self._num_worlds, capacity), dtype=wp.float32, device=self._device)
        self.factors = self._data.factors = wp.zeros_like(self.values)
        self.columns = self._data.columns = wp.zeros(self.values.shape, dtype=wp.int32, device=self._device)
        self.entry_rows = self._data.entry_rows = wp.zeros_like(self.columns)
        self._operator.mat = self.values

    def _launch(self, kernel, shape, *inputs, **kwargs):
        wp.launch(kernel, shape, [self._data, *inputs], device=self._device, **kwargs)

    def build(self, model, data, jacobians, reset_to_zero=True):
        nw, n = self._num_worlds, self.max_dimension
        J = jacobians.data
        body_support, assemble = make_jacobian_kernels(J.J_cts_data.ndim)
        jacobian = (model.info.bodies_offset, J.J_cts_offsets, J.J_cts_data)
        self._launch(body_support, self.support.shape, *jacobian)
        self._launch(build_adjacency, self.graph.shape)
        self._launch(rcm_order, nw, block_dim=32)
        self._launch(permute_graph, self.graph.shape)
        self._launch(symbolic_cholesky, nw, block_dim=32)
        required = int(self.counts.numpy().max(initial=0))
        if required > self._capacity:
            self._allocate_values(required)
        self._launch(write_columns, (nw, n))
        self._launch(
            assemble,
            (nw, self._capacity),
            *jacobian,
            model.bodies.inv_m_i,
            data.bodies.inv_I_i,
            model.info.num_joint_dynamic_cts,
            model.info.joint_dynamic_cts_offset,
            data.joints.inv_m_j,
            model.info.num_joint_effort_cts,
            model.info.joint_effort_cts_offset,
            model.info.joint_bounded_cts_group_offset,
            model.info.num_joint_friction_cts,
            data.joints.inv_m_a,
        )

    def apply_preconditioner(self, P):
        self._P = self._data.P = P
        self._launch(scale_values, self.values.shape)

    def diagonal(self, output):
        self._launch(copy_diagonal, (self._num_worlds, self.max_dimension), output)

    def regularize(self, eta):
        self._launch(regularize_diagonal, (self._num_worlds, self.max_dimension), eta)

    def update_proximal(self, status, sigma):
        self._sigma = self._data.sigma = sigma
        self._launch(add_proximal, (self._num_worlds, self.max_dimension), status)
        self.compute()

    def multiply(self, x, y, physical=False, alpha=1.0, beta=0.0):
        self._launch(multiply_symmetric, self._num_worlds, physical, alpha, beta, x, y)


LinearSolverNameToType["LLTCompact"] = CompactCholeskySolver
