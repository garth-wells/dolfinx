// Copyright (C) 2018 Chris Richardson, Nathan Sime
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitSphereMesh.h"
#include <cmath>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/refinement/refine.h>
#include <dolfin/io/XDMFFile.h>
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh UnitSphereMesh::create(MPI_Comm comm, std::size_t n,
                                  std::size_t geo_p_dim,
                                  const mesh::GhostMode ghost_mode)
{
  if (n < 0)
    throw std::domain_error("UnitSphereMesh must have n >= 0");

  if ((geo_p_dim < 1) or (geo_p_dim > 2))
    throw std::domain_error("UnitSphereMesh supports geo_p_dim = 1 or 2");

  return build_icosahedron_surface_mesh(comm, n, geo_p_dim, ghost_mode);
}
//-----------------------------------------------------------------------------
mesh::Mesh UnitSphereMesh::build_icosahedron_surface_mesh(
        MPI_Comm comm,
        std::size_t n,
        std::size_t geo_p_dim,
        const dolfin::mesh::GhostMode ghost_mode)
{
  assert(n >= 0);
  assert(geo_p_dim == 1 or geo_p_dim == 2);

  const std::size_t points_per_cell = (geo_p_dim + 1)*(geo_p_dim + 2)/2;

  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    EigenRowArrayXXd geom(0, 3);
    EigenRowArrayXXi64 topo(0, points_per_cell);
    return mesh::MeshPartitioning::build_distributed_mesh(
            comm, mesh::CellType::Type::triangle, geom, topo, {},
            ghost_mode);
  }

  // Golden ratio
  const double gr = (1.0 + std::pow(5.0, 0.5)) / 2.0;

  // Initial hard coded icosahedron surface
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

  // If no refinement, just build the mesh
  if (n == 0 and geo_p_dim == 1)
  {
    return mesh::MeshPartitioning::build_distributed_mesh(
            comm, mesh::CellType::Type::triangle,
            points,
            cells,
            {}, ghost_mode);
  }

  // Construct local mesh for refinement on process 0. The user may then
  // refine further for parallel efficiency after initial mesh distribution.
  mesh::Mesh initial_surface(MPI_COMM_SELF, mesh::CellType::Type::triangle,
                             points, cells, {}, mesh::GhostMode::none);

  // Geometric refinement
  for (std::size_t ref_level = 0; ref_level < n; ++ref_level)
    initial_surface = refinement::refine(initial_surface, false);

  // Element p-order refinement
  if (geo_p_dim > 1)
    initial_surface = geo_p_refine_mesh(initial_surface);

  // Interpolate points on surface of the sphere.
  auto& refined_points = initial_surface.geometry().points();
  for (std::int64_t r = 0; r < refined_points.rows(); ++r) {
    double norm = 0.0;
    for (std::size_t d = 0; d < 3; ++d)
      norm += std::pow(refined_points(r, d), 2);
    norm = std::pow(norm, 0.5);
    refined_points.row(r) /= norm;
  }

  // Convert topology to int64_t
  const auto& refined_connectivity =
          initial_surface.coordinate_dofs().entity_points(2).connections();
  const std::vector<std::int64_t> refined_connectivity_int64(
          refined_connectivity.cbegin(), refined_connectivity.cend());
  Eigen::Map<const EigenRowArrayXXi64>
          refined_cells(refined_connectivity_int64.data(),
                        initial_surface.num_cells(), points_per_cell);

  // Distribute the initial mesh
  return mesh::MeshPartitioning::build_distributed_mesh(
          comm, mesh::CellType::Type::triangle,
          refined_points,
          refined_cells,
          {}, ghost_mode);
}
//-----------------------------------------------------------------------------
mesh::Mesh UnitSphereMesh::geo_p_refine_mesh(const mesh::Mesh& mesh)
{
  // P1 -> P2 only for now
  mesh.init(1); // init edges
  mesh.init(mesh.topology().dim(), 1);

  const std::size_t num_edges = (std::size_t) mesh.num_entities(1);
  const std::size_t num_cells = mesh.num_cells();
  const std::size_t num_verts = mesh.num_vertices();

  assert(num_edges > 0);
  std::vector<geometry::Point> edge_midpoints(num_edges);

  // Old points and cells
  const EigenRowArrayXXd& old_points = mesh.geometry().points();
  const auto& old_connectivity_int32 =
          mesh.topology().connectivity(2, 0).connections();
  const std::vector<std::int64_t> old_connectivity_int64(
          old_connectivity_int32.cbegin(), old_connectivity_int32.cend());
  Eigen::Map<const EigenRowArrayXXi64>
          old_connectivity(old_connectivity_int64.data(),
                        num_cells, 3);

  // New points and cells
  EigenRowArrayXXd new_points(num_verts + num_edges, 3);
  new_points.block(0, 0, num_verts, 3)
          = old_points.block(0, 0, num_verts, 3);
  EigenRowArrayXXi64 new_connectivity(num_cells, 6);
  new_connectivity.block(0, 0, num_cells, 3)
          = old_connectivity.block(0, 0, num_cells, 3);

  // Get edge midpoints
  for (const auto& edge : mesh::MeshRange<mesh::Edge>(mesh))
  {
    const auto& midpoint = edge.midpoint();
    new_points.row(num_verts + edge.index()) << midpoint[0], midpoint[1], midpoint[2];
  }

  // Populate new tri_6 cells with edge points
  std::vector<double> new_cell_verts(3);
  for (const auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    for (const auto& edge : mesh::EntityRange<mesh::Edge>(cell))
      new_cell_verts[cell.index(edge)] = num_verts + edge.index();

    // Need the connectivity in VTK order
    new_connectivity.block(cell.index(), 3, 1, 3)
            << new_cell_verts[1], new_cell_verts[2], new_cell_verts[0];
  }

  return mesh::Mesh(mesh.mpi_comm(), mesh.type().cell_type(),
                    new_points, new_connectivity,
                    {}, mesh.get_ghost_mode());
}
//-----------------------------------------------------------------------------
