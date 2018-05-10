// Copyright (C) 2018 Chris Richardson, Nathan Sime
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{
namespace generation
{

/// A mesh consisting of a spherical domain with quadratic geometry.
/// This class is useful for testing.

class UnitSphereMesh
{
public:
  /// Create mesh of unit sphere for testing quadratic geometry.
  ///
  /// @note The geometric points of the mesh interpolate the
  ///   surface of the unit sphere. For optimal representation
  ///   of the unit sphere surface, u, the user must solve
  ///   the nonlinear projection problem: Given the initial
  ///   surface, u_0, find the correction, d, in V^3_p such
  ///   that:
  ///             F(d) := \int ( (u + d)^2 - 1) v dx = 0
  ///   for all v in V^3_p.
  ///
  /// @param n
  ///   number of refinement levels
  /// @param geo_p_dim
  ///   Order of geometry representation
  static mesh::Mesh create(MPI_Comm comm, std::size_t n,
                           std::size_t geo_p_dim,
                           const mesh::GhostMode ghost_mode);

private:
  // Construct a unit icosahedron. The icosahedron is then
  // refined locally on process 0 n times. This mesh is then
  // distributed to parallel processes. The icosahedron
  // approximates the unit sphere.
  //
  // @param n
  //   number of refinement levels
  static mesh::Mesh build_icosahedron_surface_mesh(
          MPI_Comm comm, std::size_t n, std::size_t geo_p_dim,
          const mesh::GhostMode ghost_mode);

  static mesh::Mesh geo_p_refine_mesh(const mesh::Mesh& mesh);
};
} // namespace generation
} // namespace dolfin
