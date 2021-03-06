"Unit tests for nullspace test"

# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import pytest
from dolfin_utils.test import *

def build_elastic_nullspace(V, x):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh().geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(dim)]

    # Build translational null space basis
    for i in range(gdim):
        V.sub(i).dofmap().set(nullspace_basis[i], 1.0);

    # Build rotational null space basis
    if gdim == 2:
        V.sub(0).set_x(nullspace_basis[2], -1.0, 1);
        V.sub(1).set_x(nullspace_basis[2], 1.0, 0);
    elif gdim == 3:
        V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
        V.sub(1).set_x(nullspace_basis[3],  1.0, 0);

        V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
        V.sub(2).set_x(nullspace_basis[4], -1.0, 0);

        V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
        V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    return VectorSpaceBasis(nullspace_basis)


def build_broken_elastic_nullspace(V, x):
    """Function to build incorrect null space for 2D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(4)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[2], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[2], 1.0, 0);

    # Add vector that is not in nullspace
    V.sub(1).set_x(nullspace_basis[3], 1.0, 1);

    for x in nullspace_basis:
        x.apply("insert")
    return VectorSpaceBasis(nullspace_basis)


@pytest.mark.skip
def test_nullspace_orthogonal():
    """Test that null spaces orthogonalisation"""
    meshes = [UnitSquareMesh(MPI.comm_world, 12, 12), UnitCubeMesh(MPI.comm_world, 4, 4, 4)]
    for mesh in meshes:
        for p in range(1, 4):
            V = VectorFunctionSpace(mesh, 'CG', p)
            zero = Constant([0.0]*mesh.geometry.dim)
            L = dot(TestFunction(V), zero)*dx
            x = assemble(L)

            # Build nullspace
            null_space = build_elastic_nullspace(V, x)

            assert not null_space.is_orthogonal()
            assert not null_space.is_orthonormal()

            # Orthogonalise nullspace
            null_space.orthonormalize()

            # Checl that null space basis is orthonormal
            assert null_space.is_orthogonal()
            assert null_space.is_orthonormal()

@pytest.mark.skip
def test_nullspace_check():
    # Mesh
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12)

    # Elasticity form
    V = VectorFunctionSpace(mesh, 'CG', 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(sym(grad(u)), grad(v))*dx

    # Assemble matrix and create compatible vector
    A = assemble(a)
    x = PETScVector()
    A.init_vector(x, 1)

    # Create null space basis and test
    null_space = build_elastic_nullspace(V, x)
    assert in_nullspace(A, null_space)
    assert in_nullspace(A, null_space, "right")
    assert in_nullspace(A, null_space, "left")

    # Create incorect null space basis and test
    null_space = build_broken_elastic_nullspace(V, x)
    assert not in_nullspace(A, null_space)
    assert not in_nullspace(A, null_space, "right")
    assert not in_nullspace(A, null_space, "left")

    # Reset backend
    parameters["linear_algebra_backend"] = prev_backend
