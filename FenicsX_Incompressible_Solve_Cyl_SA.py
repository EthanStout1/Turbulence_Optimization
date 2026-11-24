from petsc4py import PETSc
import numpy as np
from ufl import (
    TestFunction,
    TrialFunction,
    div,
    dot,
    grad,
    nabla_grad,
    inner,
    split,
    Identity,
    derivative,
    sqrt,
    exp,
    min_value,
    max_value,
    FacetNormal,
    Measure,
    SpatialCoordinate,
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


################# Set Key Solver Parameters
Re = 100  # Reynolds Number
c_ref = 1.0  # Reference length scale
rho_ref = 1.0  # Reference Density
vel_ref = 1.0  # Reference Velocity
nu_ref = vel_ref * c_ref / Re  # Reference kinematic viscosity
vel_i = vel_ref  # Initial Velocity (set equal to velocity over the domain)
vel_in = vel_ref  # Inlet Velocity
p_out = 0.0  # Outlet Pressure (arbitrary choice)
nu_t_wall = 0.0  # Wall eddy viscosity
nu_t_ff = 3 * nu_ref  # Wall eddy viscosity in the farfield

################# Read in Mesh
with XDMFFile(MPI.COMM_WORLD, "make_cyl_mesh/Cyl_fine.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    cd = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

with XDMFFile(MPI.COMM_WORLD, "make_cyl_mesh/Cyl_fine_facets.xdmf", "r") as xdmf:
    fd = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

tdim = mesh.topology.dim
fdim = tdim - 1


ds = Measure("ds", domain=mesh, subdomain_data=fd, metadata={"quadrature_degree": 6})
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})

x = SpatialCoordinate(mesh)

################# Define Finite Element Spaces
U_el = element(
    "Lagrange", mesh.basix_cell(), K + 1, shape=(mesh.geometry.dim,)
)  # Velocity Element
P_el = element("Lagrange", mesh.basix_cell(), K)  # Pressure Element
Nu_tilda_el = element("Lagrange", mesh.basix_cell(), K)  # Working Viscosity Element


################# Define Function Spaces
W_el = mixed_element([U_el, P_el, Nu_tilda_el])  # Define Mixed element
W = functionspace(mesh, W_el)  # Define Mixed Function Space


################# Define Trial and Test Functions
du = TrialFunction(W)  # For Jacobian
up = Function(W)
u, p, nt = split(up)

vq = TestFunction(W)
v, q, z = split(vq)


################# Define Function Spaces
V_sub, _ = W.sub(0).collapse()


################ Define Primitive Boundary Conditions
def inflow(x):
    return np.stack((np.full(x.shape[1], vel_in), np.zeros(x.shape[1])))


def noslip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))


ID_Top_Bot = 7
ID_In = 8
ID_Out = 9
ID_Wall = 10

# Locate DOFs
top_bot_dofs_v = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_Top_Bot))
top_bot_dofs_nu = locate_dofs_topological((W.sub(2)), fdim, fd.find(ID_Top_Bot))

in_dofs_v = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_In))
in_dofs_nu = locate_dofs_topological((W.sub(2)), fdim, fd.find(ID_In))

out_dofs_p = locate_dofs_topological((W.sub(1)), fdim, fd.find(ID_Out))
out_dofs_nu = locate_dofs_topological((W.sub(2)), fdim, fd.find(ID_Out))

wall_dofs_v = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_Wall))
wall_dofs_nu = locate_dofs_topological((W.sub(2)), fdim, fd.find(ID_Wall))


# Create BCs
u_top_bot = Function(V_sub)
u_top_bot.interpolate(inflow)
bc_top_bot = dirichletbc(u_top_bot, top_bot_dofs_v, W.sub(0))
bc_top_bot_nu = dirichletbc(PETSc.ScalarType(nu_t_ff), top_bot_dofs_nu, W.sub(2))

u_in = Function(V_sub)
u_in.interpolate(inflow)
bc_inlet = dirichletbc(u_in, in_dofs_v, W.sub(0))
bc_inlet_nu = dirichletbc(PETSc.ScalarType(nu_t_ff), in_dofs_nu, W.sub(2))


bc_outlet = dirichletbc(PETSc.ScalarType(p_out), out_dofs_p, W.sub(1))
bc_outlet_nu = dirichletbc(PETSc.ScalarType(nu_t_ff), out_dofs_nu, W.sub(2))

u_wall = Function(V_sub)
u_wall.interpolate(noslip)
bc_wall = dirichletbc(u_wall, wall_dofs_v, W.sub(0))
bc_wall_nu = dirichletbc(PETSc.ScalarType(nu_t_wall), wall_dofs_nu, W.sub(2))


# Collect BCs
# BCs = [bc_wall_1, bc_wall_2, bc_inlet, bc_outlet, bc_outlet_v]
# BCs_main = [
#    bc_top_bot,
#    bc_inlet,
#    bc_wall,
#    bc_top_bot_nu,
#    bc_inlet_nu,
#    bc_outlet_nu,
#    bc_wall_nu,
# ]

BCs_main = [
    bc_top_bot,
    bc_inlet,
    bc_wall,
    bc_top_bot_nu,
    bc_inlet_nu,
    bc_wall_nu,
]


################ Weak Formulation Constants
iden = Identity(2)  # Identity Matrix
nu = Constant(mesh, default_scalar_type(nu_ref))
rho = Constant(mesh, default_scalar_type(rho_ref))
n = FacetNormal(mesh)

################ SA Variables


# SA parameter constants
cb1 = 0.1355
sigma = 2.0 / 3.0
cb2 = 0.622
kappa = 0.41
cw2 = 0.3
cw3 = 2.0
cv1 = 7.1
ct3 = 1.2
ct4 = 0.5
cw1 = (cb1 / (kappa * kappa)) + (1 + cb2) / sigma

# Turbulent eddy viscosity
X = nt / nu_ref
fv1 = (X * X * X) / (X * X * X + cv1 * cv1 * cv1)
nu_turb = nt * fv1

# Wall distance equation (circle centered at (0,0) )
circ_r = 0.5  # Radius of the circle
d = sqrt(x[0] * x[0] + x[1] * x[1]) - circ_r

# Vorticity magnitude
Wij = 0.5 * (grad(u) - grad(u).T)
Omega = abs(u[1].dx(0) - u[0].dx(1))

# Other relevant terms
ft2 = ct3 * exp(-ct4 * X * X)
fv2 = 1 - (X / (1 + X * fv1))
S_tilda = max_value(Omega + (nt / (kappa * kappa * d * d)) * fv2, 0.3 * Omega)
S_safe = max_value(S_tilda, 1e-12)  # Prevent div by 0
r = min_value((nt / (S_safe * kappa * kappa * d * d)), 10)
g = r + cw2 * ((r**6) - r)
fw = g * ((1 + cw3**6) / (g**6 + cw3**6)) ** (1 / 6)


################ SA Term

L_SA_1 = dot(u, grad(nt)) * z * dx  # Advection
L_SA_2 = -cb1 * (1 - ft2) * S_tilda * nt * z * dx  # Production
L_SA_3 = (
    (cw1 * fw - (cb1 / (kappa * kappa)) * ft2) * (nt / d) * (nt / d) * z * dx
)  # Destruction
L_SA_4 = (1 / sigma) * inner((nu + nt) * grad(nt), grad(z)) * dx
L_SA_5 = -(1 / sigma) * (cb2 * inner(grad(nt), grad(nt))) * z * dx

L_SA = L_SA_1 + L_SA_2 + L_SA_3 + L_SA_4 + L_SA_5


################ Weak/Variational Form of IC NS

L_1 = div(u) * q * dx  # Continuity eqn (scaled for symmetry)

L_2 = (
    dot(dot(u, nabla_grad(u)), v) * dx
    + 1.0 / 2.0 * dot(div(u) * u, v) * dx
    - (1.0 / rho) * p * div(v) * dx
)  # 2 Convective Skew-Symmetric terms and Pressure Term

L_3 = (
    (nu + nu_turb)
    * inner(nabla_grad(u) + nabla_grad(u).T - 2.0 / 3.0 * div(u) * iden, nabla_grad(v))
    * dx
)  # Diffusive Term


# L = L_1 + L_2 + L_3 + L_neummm  # Steady-state sim
L = L_1 + L_2 + L_3 + L_SA  # Steady-state sim


################ Set Solver Definition
Jac = derivative(L, up, du)
problem = NonlinearProblem(L, up, bcs=BCs_main, J=Jac)
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
n, converged = solver.solve(up)
assert converged
print(f"Number of iterations: {n:d}")

u_, p_, nt_ = (
    up.split()[0].collapse(),
    up.split()[1].collapse(),
    up.split()[2].collapse(),
)


########### Print out values
folder = Path("./tmp")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(
    mesh.comm,
    folder / ("Velocity" + "_.bp"),
    [u_],
    engine="BP4",
)
vtx_p = VTXWriter(
    mesh.comm,
    folder / ("Pressure" + "_.bp"),
    [p_],
    engine="BP4",
)

vtx_nu = VTXWriter(
    mesh.comm,
    folder / ("Working_Viscosity" + "_.bp"),
    [nt_],
    engine="BP4",
)

vtx_u.write(0.0)
vtx_p.write(0.0)
vtx_nu.write(0.0)


quit()
