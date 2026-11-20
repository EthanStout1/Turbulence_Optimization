# from dolfinx.io import gmshio
from mpi4py import MPI
import meshio
import numpy as np

# mesh, cell_markers, facet_markers = gmshio.read_from_msh(
#    "cyl.msh", MPI.COMM_WORLD, gdim=2
# )


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data.astype(np.int32)]},
    )
    return out_mesh


# if proc == 0:
# Read in mesh
msh = meshio.read("cyl.msh")

# Create and save one file for the mesh, and one file for the facets
triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
line_mesh = create_mesh(msh, "line", prune_z=True)
meshio.write("Cyl.xdmf", triangle_mesh)
meshio.write("Cyl_facets.xdmf", line_mesh)
MPI.COMM_WORLD.barrier()
