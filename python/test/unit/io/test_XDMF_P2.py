# Copyright (C) 2018 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import os
from dolfin import *
from dolfin_utils.test import tempdir
import dolfin

p2_mesh_factories = [
    (dolfin.cpp.generation.UnitDiscMesh.create, (MPI.comm_world, 3, dolfin.cpp.mesh.GhostMode.none)),
    (dolfin.cpp.generation.UnitSphereMesh.create, (MPI.comm_world, 1, 2, dolfin.cpp.mesh.GhostMode.none)),
]

@pytest.mark.parametrize("mesh_factory", p2_mesh_factories)
def test_read_write_p2_mesh(tempdir, mesh_factory):
    mesh_func, args = mesh_factory
    mesh = mesh_func(*args)

    filename = os.path.join(tempdir, "tri6_mesh.xdmf")
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        xdmf.write(mesh, XDMFFile.Encoding.HDF5)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        mesh2 = xdmf.read_mesh(mesh.mpi_comm(), cpp.mesh.GhostMode.none)

    assert mesh2.num_entities_global(mesh.topology.dim) == mesh.num_entities_global(mesh.topology.dim)
    assert mesh2.num_entities_global(0) == mesh.num_entities_global(0)

def test_read_write_p2_function(tempdir):
    mesh = dolfin.cpp.generation.UnitDiscMesh.create(MPI.comm_world, 3,
                                                     dolfin.cpp.mesh.GhostMode.none)
    cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    Q = FunctionSpace(mesh, "Lagrange", 2)

    F = Function(Q)
    F.interpolate(Expression("x[0]", degree=1))

    filename = os.path.join(tempdir, "tri6_function.xdmf")
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        xdmf.write(F, XDMFFile.Encoding.HDF5)
