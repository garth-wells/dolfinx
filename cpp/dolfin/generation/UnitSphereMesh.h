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
  /// Create mesh of unit sphere for testing quadratic geometry
  /// @param n
  ///   number of refinement levels
  static mesh::Mesh create(MPI_Comm comm, std::size_t n,
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
          MPI_Comm comm, const mesh::GhostMode ghost_mode);
};
} // namespace generation
} // namespace dolfin
