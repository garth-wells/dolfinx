// Copyright (C) 2018 Chris Richardson, Nathan Sime
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitSphereMesh.h"
#include <cmath>
#include <dolfin/common/types.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/refinement/refine.h>
#include <dolfin/io/XDMFFile.h>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh UnitSphereMesh::create(MPI_Comm comm, std::size_t n,
                                const mesh::GhostMode ghost_mode)
{
  return build_icosahedron_surface_mesh(comm, n, ghost_mode);
}
//-----------------------------------------------------------------------------
mesh::Mesh UnitSphereMesh::build_icosahedron_surface_mesh(
        MPI_Comm comm,
        const dolfin::mesh::GhostMode ghost_mode)
{
  assert(n >= 0);

  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    EigenRowArrayXXd geom(0, 3);
    EigenRowArrayXXi64 topo(0, 3);
    return mesh::MeshPartitioning::build_distributed_mesh(
            comm, mesh::CellType::Type::triangle, geom, topo, {},
            ghost_mode);
  }

  // Golden ratio
  const double gr = (1.0 + std::pow(5.0, 0.5)) / 2.0;

  EigenRowArrayXXd points(12, 3);

  points.block(0, 0, 12, 3) <<
          -1.0, gr, 0.0,
          -gr, 0.0, 1.0,
          0.0, 1.0, gr,
          1.0, gr, 0.0,
          0.0, 1.0, -gr,
          -gr, 0.0, -1.0,
          gr, 0.0, 1.0,
          0.0, -1.0, gr,
          -1.0, -gr, 0.0,
          0.0, -1.0, -gr,
          gr, 0.0, -1.0,
          1.0, -gr, 0.0;

  EigenRowArrayXXi64 cells(20, 3);

  cells.block(0, 0, 20, 3) <<
          0, 1, 2,
          0, 2, 3,
          0, 3, 4,
          0, 4, 5,
          0, 5, 1,
          3, 2, 6,
          2, 1, 7,
          1, 5, 8,
          5, 4, 9,
          4, 3, 10,
          11, 6, 7,
          11, 7, 8,
          11, 8, 9,
          11, 9, 10,
          11, 10, 6,
          7, 6, 2,
          8, 7, 1,
          9, 8, 5,
          10, 9, 4,
          6, 10, 3;

  mesh::Mesh initial_surface(MPI_COMM_SELF, mesh::CellType::Type::triangle,
                             points, cells, {}, mesh::GhostMode::none);

  if (n > 0)
  {
    // Locally refine the intial mesh on process 0. The user may then refine further
    // for parallel efficiency after initial mesh distribution.
    for (std::size_t ref_level = 0; ref_level < n; ++ref_level)
      initial_surface = refinement::refine(initial_surface, false);

    // Project points onto surface of the sphere
    auto &refined_points = initial_surface.geometry().points();
    for (std::size_t r = 0; r < refined_points.rows(); ++r) {
      double norm = 0.0;
      for (std::size_t d = 0; d < 3; ++d)
        norm += std::pow(refined_points(r, d), 2);
      norm = std::pow(norm, 0.5);
      refined_points.row(r) /= norm;
    }
  }

  // Convert topology to int64_t
  const auto& refined_connectivity = initial_surface.topology().connectivity(2, 0).connections();
  const std::vector<std::int64_t> refined_connectivity_int64(
          refined_connectivity.cbegin(), refined_connectivity.cend());
  Eigen::Map<const EigenRowArrayXXi64> refined_cells(refined_connectivity_int64.data(),
                                                     initial_surface.num_cells(), 3);

  // Distribute the initial mesh
  return mesh::MeshPartitioning::build_distributed_mesh(
          comm, mesh::CellType::Type::triangle,
          refined_points,
          refined_cells,
          {}, ghost_mode);
}
//-----------------------------------------------------------------------------
