from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List
import time

import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpla
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class _SparseLUFactor:
    def __init__(self, matrix: sp.spmatrix):
        t0 = time.perf_counter()
        self.factor = spla.splu(matrix.tocsc())
        self.factor_time = time.perf_counter() - t0
        self.solve_time = 0.0
        self.solve_adjoint_time = 0.0

    def solve(self, rhs: np.ndarray, record: bool = True) -> np.ndarray:
        t0 = time.perf_counter()
        result = self.factor.solve(rhs)
        if record:
            self.solve_time += time.perf_counter() - t0
        return result

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        result = self.factor.solve(rhs, trans="H")
        self.solve_adjoint_time += time.perf_counter() - t0
        return result

    def factor_nnz(self) -> int:
        return int(self.factor.nnz)

    def profile_summary(self) -> dict:
        return {
            "nodes": 1,
            "leaves": 1,
            "leaf_factor_time": self.factor_time,
            "child_factor_wall_time": 0.0,
            "schur_build_time": 0.0,
            "schur_child_solve_time": 0.0,
            "dense_factor_time": 0.0,
            "forward_solve_time": self.solve_time,
            "adjoint_solve_time": self.solve_adjoint_time,
        }


class _DenseLUFactor:
    def __init__(self, matrix: np.ndarray, backend: str, gpu_min_dofs: int):
        self.shape = matrix.shape
        self.memory_mb = float(matrix.nbytes / (1024.0**2))
        self.backend = "gpu" if backend == "gpu" and matrix.shape[0] >= gpu_min_dofs else "cpu"
        t0 = time.perf_counter()
        if self.backend == "gpu":
            matrix_gpu = cp.asarray(matrix)
            self.factor = cpla.lu_factor(matrix_gpu)
            cp.cuda.Stream.null.synchronize()
        else:
            self.factor = la.lu_factor(matrix)
        self.factor_time = time.perf_counter() - t0
        self.solve_time = 0.0
        self.solve_adjoint_time = 0.0

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if self.backend == "gpu":
            rhs_gpu = cp.asarray(rhs)
            result = cp.asnumpy(cpla.lu_solve(self.factor, rhs_gpu))
            cp.cuda.Stream.null.synchronize()
        else:
            result = la.lu_solve(self.factor, rhs)
        self.solve_time += time.perf_counter() - t0
        return result

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if self.backend == "gpu":
            rhs_gpu = cp.asarray(rhs)
            result = cp.asnumpy(cpla.lu_solve(self.factor, rhs_gpu, trans=2))
            cp.cuda.Stream.null.synchronize()
        else:
            result = la.lu_solve(self.factor, rhs, trans=2)
        self.solve_adjoint_time += time.perf_counter() - t0
        return result


@dataclass
class _SchurChild:
    indices: np.ndarray
    solver: object
    a_is: sp.csc_matrix
    a_si: sp.csr_matrix

    def solve(self, rhs: np.ndarray, record: bool = True) -> np.ndarray:
        return self.solver.solve(rhs, record=record)

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        return self.solver.solve_adjoint(rhs)


class RecursiveGridSchurDirectSolver:
    """
    Exact recursive grid Schur direct solver.

    Every non-leaf node chooses the longest active grid axis and keeps one grid
    plane as separator. The two remaining slabs are eliminated recursively. This
    is an exact direct method for the current 7-point stencil; it changes only
    the elimination order, not the discrete Helmholtz model.
    """

    def __init__(
        self,
        matrix: sp.spmatrix,
        dims: tuple[int, int, int],
        leaf_dofs: int = 4096,
        min_axis_size: int = 3,
        schur_block_size: int = 64,
        dense_backend: str = "cpu",
        gpu_min_separator_dofs: int = 512,
        parallel_depth: int = 0,
        profile_print_depth: int = 0,
        depth: int = 0,
    ):
        self.matrix = matrix.tocsc()
        self.dtype = np.result_type(self.matrix.dtype, np.complex128)
        self.n = self.matrix.shape[0]
        self.dims = tuple(int(v) for v in dims)
        self.leaf_dofs = int(leaf_dofs)
        self.min_axis_size = int(min_axis_size)
        self.schur_block_size = int(schur_block_size)
        self.dense_backend = str(dense_backend).lower()
        self.gpu_min_separator_dofs = int(gpu_min_separator_dofs)
        self.parallel_depth = int(parallel_depth)
        self.profile_print_depth = int(profile_print_depth)
        self.depth = int(depth)
        self.split_axis = self._choose_split_axis()
        self.is_leaf = self.n <= self.leaf_dofs or self.split_axis < 0

        self.leaf_factor = None
        self.children: List[_SchurChild] = []
        self.separator_indices = np.array([], dtype=np.int64)
        self.separator_shape = (0, 0)
        self.separator_memory_mb = 0.0
        self.separator_factor = None
        self.profile = {
            "child_factor_wall_time": 0.0,
            "schur_build_time": 0.0,
            "schur_child_solve_time": 0.0,
            "total_factor_time": 0.0,
            "forward_solve_time": 0.0,
            "adjoint_solve_time": 0.0,
        }

        t0 = time.perf_counter()
        if self.is_leaf:
            self.leaf_factor = _SparseLUFactor(self.matrix)
        else:
            self._factor_recursive_node()
        self.profile["total_factor_time"] = time.perf_counter() - t0

    @classmethod
    def from_grid(
        cls,
        matrix: sp.spmatrix,
        nx: int,
        ny: int,
        nz: int,
        leaf_dofs: int = 4096,
        min_axis_size: int = 3,
        schur_block_size: int = 64,
        dense_backend: str = "cpu",
        gpu_min_separator_dofs: int = 512,
        parallel_depth: int = 0,
        profile_print_depth: int = 0,
    ) -> "RecursiveGridSchurDirectSolver":
        return cls(
            matrix=matrix,
            dims=(nx, ny, nz),
            leaf_dofs=leaf_dofs,
            min_axis_size=min_axis_size,
            schur_block_size=schur_block_size,
            dense_backend=dense_backend,
            gpu_min_separator_dofs=gpu_min_separator_dofs,
            parallel_depth=parallel_depth,
            profile_print_depth=profile_print_depth,
        )

    def _choose_split_axis(self) -> int:
        candidates = [(size, axis) for axis, size in enumerate(self.dims) if size >= self.min_axis_size]
        if not candidates:
            return -1
        return max(candidates)[1]

    def _factor_recursive_node(self):
        axis = self.split_axis
        split_at = self.dims[axis] // 2
        grid = np.arange(self.n, dtype=np.int64).reshape(self.dims, order="F")

        sep_slices = [slice(None), slice(None), slice(None)]
        sep_slices[axis] = split_at
        self.separator_indices = grid[tuple(sep_slices)].ravel(order="F")
        if self.depth <= self.profile_print_depth:
            print(
                f"[SchurProfile] depth={self.depth} dims={self.dims} "
                f"split_axis={axis} separator_dofs={self.separator_indices.size}",
                flush=True,
            )

        child_build_inputs = []
        child_ranges = [(0, split_at), (split_at + 1, self.dims[axis])]
        for start, stop in child_ranges:
            child_slices = [slice(None), slice(None), slice(None)]
            child_slices[axis] = slice(start, stop)
            child_indices = grid[tuple(child_slices)].ravel(order="F")
            if child_indices.size > 0:
                child_dims = list(self.dims)
                child_dims[axis] = stop - start
                child_matrix = self.matrix[child_indices, :][:, child_indices].tocsc()
                a_is = self.matrix[child_indices, :][:, self.separator_indices].tocsc()
                a_si = self.matrix[self.separator_indices, :][:, child_indices].tocsr()
                child_build_inputs.append(
                    (child_indices, tuple(child_dims), child_matrix, a_is, a_si)
                )

        t0 = time.perf_counter()
        child_solvers = self._factor_child_solvers(child_build_inputs)
        self.profile["child_factor_wall_time"] = time.perf_counter() - t0
        if self.depth <= self.profile_print_depth:
            print(
                f"[SchurProfile] depth={self.depth} child_factor_wall="
                f"{self.profile['child_factor_wall_time']:.3f}s",
                flush=True,
            )
        for child_data, child_solver in zip(child_build_inputs, child_solvers):
            child_indices, _, _, a_is, a_si = child_data
            self.children.append(
                _SchurChild(
                    indices=child_indices,
                    solver=child_solver,
                    a_is=a_is,
                    a_si=a_si,
                )
            )

        t0 = time.perf_counter()
        separator_matrix = self._build_separator_schur()
        self.profile["schur_build_time"] = time.perf_counter() - t0
        if self.depth <= self.profile_print_depth:
            print(
                f"[SchurProfile] depth={self.depth} schur_build="
                f"{self.profile['schur_build_time']:.3f}s "
                f"child_solve_in_build={self.profile['schur_child_solve_time']:.3f}s",
                flush=True,
            )
        self.separator_shape = separator_matrix.shape
        self.separator_memory_mb = float(separator_matrix.nbytes / (1024.0**2))
        self.separator_factor = _DenseLUFactor(
            separator_matrix,
            backend=self.dense_backend,
            gpu_min_dofs=self.gpu_min_separator_dofs,
        )
        if self.depth <= self.profile_print_depth:
            print(
                f"[SchurProfile] depth={self.depth} dense_lu="
                f"{self.separator_factor.factor_time:.3f}s "
                f"backend={self.separator_factor.backend}",
                flush=True,
            )

    def _factor_child_solvers(self, child_build_inputs: list[tuple]) -> list:
        if self.depth < self.parallel_depth and len(child_build_inputs) > 1:
            with ThreadPoolExecutor(max_workers=len(child_build_inputs)) as executor:
                futures = [
                    executor.submit(
                        RecursiveGridSchurDirectSolver,
                        matrix=child_matrix,
                        dims=child_dims,
                        leaf_dofs=self.leaf_dofs,
                        min_axis_size=self.min_axis_size,
                        schur_block_size=self.schur_block_size,
                        dense_backend=self.dense_backend,
                        gpu_min_separator_dofs=self.gpu_min_separator_dofs,
                        parallel_depth=self.parallel_depth,
                        profile_print_depth=self.profile_print_depth,
                        depth=self.depth + 1,
                    )
                    for _, child_dims, child_matrix, _, _ in child_build_inputs
                ]
                return [future.result() for future in futures]

        return [
            RecursiveGridSchurDirectSolver(
                matrix=child_matrix,
                dims=child_dims,
                leaf_dofs=self.leaf_dofs,
                min_axis_size=self.min_axis_size,
                schur_block_size=self.schur_block_size,
                dense_backend=self.dense_backend,
                gpu_min_separator_dofs=self.gpu_min_separator_dofs,
                parallel_depth=self.parallel_depth,
                profile_print_depth=self.profile_print_depth,
                depth=self.depth + 1,
            )
            for _, child_dims, child_matrix, _, _ in child_build_inputs
        ]

    def _build_separator_schur(self) -> np.ndarray:
        s_count = self.separator_indices.size
        schur = self.matrix[self.separator_indices, :][:, self.separator_indices].toarray()
        schur = np.asarray(schur, dtype=self.dtype)

        for child in self.children:
            for start in range(0, s_count, self.schur_block_size):
                stop = min(start + self.schur_block_size, s_count)
                rhs_block = child.a_is[:, start:stop].toarray()
                t0 = time.perf_counter()
                solved = child.solve(rhs_block, record=False)
                self.profile["schur_child_solve_time"] += time.perf_counter() - t0
                schur[:, start:stop] -= child.a_si @ solved

        return schur

    def solve(self, rhs: np.ndarray, record: bool = True) -> np.ndarray:
        t0 = time.perf_counter()
        if self.is_leaf:
            result = self.leaf_factor.solve(rhs, record=record)
            if record:
                self.profile["forward_solve_time"] += time.perf_counter() - t0
            return result

        rhs_matrix, squeeze = self._as_rhs_matrix(rhs)
        result = np.zeros_like(rhs_matrix, dtype=self.dtype)
        separator_rhs = rhs_matrix[self.separator_indices, :].copy()
        interior_rhs_list = []

        for child in self.children:
            interior_rhs = rhs_matrix[child.indices, :]
            interior_rhs_list.append(interior_rhs)
            separator_rhs -= child.a_si @ child.solve(interior_rhs, record=record)

        separator_solution = self.separator_factor.solve(separator_rhs)
        result[self.separator_indices, :] = separator_solution

        for child, interior_rhs in zip(self.children, interior_rhs_list):
            corrected_rhs = interior_rhs - child.a_is @ separator_solution
            result[child.indices, :] = child.solve(corrected_rhs, record=record)

        result = result[:, 0] if squeeze else result
        if record:
            self.profile["forward_solve_time"] += time.perf_counter() - t0
        return result

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if self.is_leaf:
            result = self.leaf_factor.solve_adjoint(rhs)
            self.profile["adjoint_solve_time"] += time.perf_counter() - t0
            return result

        rhs_matrix, squeeze = self._as_rhs_matrix(rhs)
        result = np.zeros_like(rhs_matrix, dtype=self.dtype)
        separator_rhs = rhs_matrix[self.separator_indices, :].copy()
        interior_rhs_list = []

        for child in self.children:
            interior_rhs = rhs_matrix[child.indices, :]
            interior_rhs_list.append(interior_rhs)
            separator_rhs -= child.a_is.conjugate().transpose() @ child.solve_adjoint(interior_rhs)

        separator_solution = self.separator_factor.solve_adjoint(separator_rhs)
        result[self.separator_indices, :] = separator_solution

        for child, interior_rhs in zip(self.children, interior_rhs_list):
            corrected_rhs = interior_rhs - child.a_si.conjugate().transpose() @ separator_solution
            result[child.indices, :] = child.solve_adjoint(corrected_rhs)

        result = result[:, 0] if squeeze else result
        self.profile["adjoint_solve_time"] += time.perf_counter() - t0
        return result

    def _as_rhs_matrix(self, rhs: np.ndarray) -> tuple[np.ndarray, bool]:
        rhs_array = np.asarray(rhs, dtype=self.dtype)
        if rhs_array.ndim == 1:
            return rhs_array.reshape((-1, 1)), True
        return rhs_array, False

    def factor_nnz(self) -> int:
        if self.is_leaf:
            return self.leaf_factor.factor_nnz()
        return int(sum(child.solver.factor_nnz() for child in self.children) + self.separator_shape[0] * self.separator_shape[1])

    def max_separator_dofs(self) -> int:
        if self.is_leaf:
            return 0
        child_max = max((child.solver.max_separator_dofs() for child in self.children), default=0)
        return max(int(self.separator_indices.size), child_max)

    def stats(self) -> dict:
        if self.is_leaf:
            return {
                "total_dofs": int(self.n),
                "leaf": True,
                "dims": self.dims,
                "leaf_factor_nnz": self.leaf_factor.factor_nnz(),
            }

        return {
            "total_dofs": int(self.n),
            "leaf": False,
            "dims": self.dims,
            "parallel_depth": int(self.parallel_depth),
            "split_axis": int(self.split_axis),
            "separator_dofs": int(self.separator_indices.size),
            "max_separator_dofs": self.max_separator_dofs(),
            "schur_shape": tuple(int(x) for x in self.separator_shape),
            "schur_memory_mb": self.separator_memory_mb,
            "dense_backend": self.separator_factor.backend,
            "recursive_factor_nnz_estimate": self.factor_nnz(),
        }

    def profile_summary(self) -> dict:
        if self.is_leaf:
            summary = self.leaf_factor.profile_summary()
            summary["total_factor_time"] = self.profile["total_factor_time"]
            return summary

        summary = {
            "nodes": 1,
            "leaves": 0,
            "leaf_factor_time": 0.0,
            "child_factor_wall_time": self.profile["child_factor_wall_time"],
            "schur_build_time": self.profile["schur_build_time"],
            "schur_child_solve_time": self.profile["schur_child_solve_time"],
            "dense_factor_time": self.separator_factor.factor_time,
            "forward_solve_time": self.profile["forward_solve_time"],
            "adjoint_solve_time": self.profile["adjoint_solve_time"],
            "total_factor_time": self.profile["total_factor_time"],
        }
        for child in self.children:
            child_summary = child.solver.profile_summary()
            for key, value in child_summary.items():
                if key not in {"forward_solve_time", "adjoint_solve_time", "total_factor_time"}:
                    summary[key] = summary.get(key, 0.0) + value
        return summary
