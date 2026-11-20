from petsc4py import PETSc
import numpy as np
from ufl import (
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    nabla_grad,
    inner,
    split,
    Identity,
    derivative,
    sqrt,
)
from basix.ufl import element, mixed_element
from dolfinx import la, default_scalar_type, log
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    dirichletbc,
    DirichletBC,
    functionspace,
    locate_dofs_topological,
)

# from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem as NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, VTXWriter
from mpi4py import MPI
from pathlib import Path


################# Set Key Solver Parameters
tol = 1e-7  # Tolerance
K = 1  # Polynomial Degree
dt = 0.1  # Time Step
Time = 30.0  # Simulation time
count = 1  # Time loop counter


################# Set Key Solver Parameters
dvisc = 0.004  # Dynamic Viscosity
dens = 1060.0  # Density
kvisc = dvisc / dens  # Kinematic Viscosity
vel_i = 0.0005  # Initial Velocity (set equal to velocity over the domain)
vel_in = vel_i  # Inlet Velocity
press_out = 0.0  # Outlet Pressure

################# Read in Mesh
with XDMFFile(MPI.COMM_WORLD, "Channel_mesh_fine.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    cd = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

with XDMFFile(MPI.COMM_WORLD, "Channel_mesh_fine_facet.xdmf", "r") as xdmf:
    fd = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

tdim = mesh.topology.dim
fdim = tdim - 1


################# Define Finite Elements
U_el = element(
    "CG", mesh.basix_cell(), K + 1, shape=(mesh.geometry.dim,)
)  # Velocity Element
P_el = element("CG", mesh.basix_cell(), K)  # Pressure Element


################# Define Function Spaces (For initial Conditions)
W_el = mixed_element([U_el, P_el])  # Define Mixed element
W = functionspace(mesh, W_el)  # Define Mixed Function Space


################# Define Trial and Test Functions
du = TrialFunction(W)  # For Jacobian
up = Function(W)
u, p = split(up)

vq = TestFunction(W)
v, q = split(vq)


################# Define Function Spaces for Runge-Kutta Terms
V_sub, _ = W.sub(0).collapse()
u_1 = Function(V_sub)
u_2 = Function(V_sub)
u_3 = Function(V_sub)

################ Set up Initial Conditions using initial value for each RK Term
for i, vel in enumerate([vel_i, 0.0]):
    u_1.sub(i).interpolate(
        lambda x, vel_in=vel: np.full(x.shape[1], vel_in, dtype=np.float64)
    )
    u_2.sub(i).interpolate(
        lambda x, vel_in=vel: np.full(x.shape[1], vel_in, dtype=np.float64)
    )
    u_3.sub(i).interpolate(
        lambda x, vel_in=vel: np.full(x.shape[1], vel_in, dtype=np.float64)
    )


#### Check if Velocity was assigned properly
folder = Path("../scratch")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(mesh.comm, folder / "Vel_check.bp", [u_3], engine="BP4")
vtx_u.write(0.0)


################ Define Boundary Conditions
def inflow(x):
    return np.stack((np.full(x.shape[1], vel_in), np.zeros(x.shape[1])))


def noslip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))


ID_Wall_1 = 3
ID_Wall_2 = 4
ID_In = 1
ID_Out = 2

# Locate DOFs
wall_1_dofs = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_Wall_1))
wall_2_dofs = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_Wall_2))
in_dofs = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_In))
out_dofs_p = locate_dofs_topological((W.sub(1)), fdim, fd.find(ID_Out))
# in_dofs_p = locate_dofs_topological((W.sub(1)), fdim, fd.find(ID_In))
out_dofs_v = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_Out))

# Create BCs
u_wall_1 = Function(V_sub)
u_wall_2 = Function(V_sub)
u_wall_1.interpolate(noslip)
u_wall_2.interpolate(noslip)
bc_wall_1 = dirichletbc(u_wall_1, wall_1_dofs, W.sub(0))
bc_wall_2 = dirichletbc(u_wall_2, wall_2_dofs, W.sub(0))

u_in = Function(V_sub)
u_in.interpolate(inflow)
bc_inlet = dirichletbc(u_in, in_dofs, W.sub(0))
# bc_inlet_p = dirichletbc(PETSc.ScalarType(1000.0001), in_dofs_p, W.sub(1))

bc_out_v = Function(V_sub)
bc_out_v.interpolate(inflow)
bc_outlet_v = dirichletbc(bc_out_v, out_dofs_v, W.sub(0))

bc_outlet = dirichletbc(PETSc.ScalarType(press_out), out_dofs_p, W.sub(1))

# Collect BCs
BCs = [bc_wall_1, bc_wall_2, bc_inlet, bc_outlet, bc_outlet_v]


################ Weak Formulation Constants
iden = Identity(2)  # Identity Matrix
n_u = Constant(mesh, default_scalar_type(kvisc))
rho = Constant(mesh, default_scalar_type(dens))
cons = Constant(mesh, default_scalar_type(6.0 / 11.0 * dt))  # BDF3 hf coefficient


################ Weak/Variational Form of IC NS
# Assume elements are at least C0 continuous on element boundaries
# Assume Pressure Dirichlet BCs are assigned on the boundary of the global domain

L_1 = (
    dot(u, v) * dx
    - 18.0 / 11.0 * dot(u_3, v) * dx
    + 9.0 / 11.0 * dot(u_2, v) * dx
    - 2.0 / 11.0 * dot(u_1, v) * dx
)  # u_dot term

L_2 = -cons * (1 / rho) * div(u) * q * dx  # Continuity eqn (scaled for symmetry)

L_3 = (
    cons * dot(dot(u, nabla_grad(u)), v) * dx
    + cons * 1 / 2 * dot(div(u) * u, v) * dx
    - cons * (1 / rho) * p * div(v) * dx
)  # 2 Convective Skew-Symmetric terms and Pressure Term

L_4 = (
    cons
    * n_u
    * inner(nabla_grad(u) + nabla_grad(u).T - 2 / 3 * div(u) * iden, nabla_grad(v))
    * dx
)  # Diffusive Term
# L_4=cons*n_u*inner(nabla_grad(u),nabla_grad(v))*dx\ # Diffusive term with divergence-free constraint directly applied


# L=L_1+L_2+L_3+L_4 # Unsteady sim
L = L_2 + L_3 + L_4  # Steady-state sim


################ Set Solver Definition
Jac = derivative(L, up, du)
problem = NonlinearProblem(L, up, bcs=BCs, J=Jac)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

solver.convergence_criterion = "incremental"
solver.rtol = tol
solver.atol = tol
solver.report = True
# solver.max_it = 200

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()

# Direct Solver
opts[f"{option_prefix}ksp_type"] = "preonly"  # Specify no iterative method
opts[f"{option_prefix}pc_type"] = "lu"  # or qr/chovlesky
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

# Krylov Subspace Method
# opts[f"{option_prefix}ksp_type"] = "cgls"
# opts[f"{option_prefix}ksp_gmres_modifiedgramschmidt"] = "true"
# opts[f"{option_prefix}pc_type"] = "none"
# opts[f"{option_prefix}ksp_initial_guess_nonzero"] = "false"
# opts[f"{option_prefix}hypre_type"] = "boomeramg
# opts["pc_hypre_boomeramg_max_iter"] =
# opts["pc_hypre_boomeramg_cycle_type"] = "v"

ksp.setFromOptions()
log.set_log_level(log.LogLevel.INFO)


################ Time Loop
t_i = 2 * dt
while t_i < Time:
    t_i += dt
    print(t_i)

    n, converged = solver.solve(up)
    assert converged
    print(f"Number of iterations: {n:d}")

    u_, p_ = (up.split()[0].collapse(), up.split()[1].collapse())

    ########## Update stored values
    u_1.x.array[:] = u_2.x.array[:]
    u_2.x.array[:] = u_3.x.array[:]
    u_3.x.array[:] = u_.x.array[:]

    ########### Print out values
    if count % 5 == 0 or count < 4:
        folder = Path("./")
        folder.mkdir(exist_ok=True, parents=True)
        vtx_u = VTXWriter(
            mesh.comm,
            folder / ("Velocity_" + "Time_" + str(t_i) + "_.bp"),
            [u_3],
            engine="BP4",
        )
        vtx_p = VTXWriter(
            mesh.comm,
            folder / ("Pressure_" + "Time_" + str(t_i) + "_.bp"),
            [p_],
            engine="BP4",
        )
        vtx_p.write(t_i)
        vtx_u.write(t_i)

    count += 1

quit()
