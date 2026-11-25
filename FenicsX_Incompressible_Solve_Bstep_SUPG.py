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
    outer,
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
    CellDiameter,
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

from dolfinx.fem.petsc import LinearProblem

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, VTXWriter
from mpi4py import MPI
from pathlib import Path

################# MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()


################# Set Key Solver Parameters
tol = 1e-7  # Tolerance
K = 1  # Polynomial Degree


################# Set Key Solver Parameters
Re = 15000  # Reynolds Number
c_ref = 1.0  # Reference length scale
rho_ref = 1.0  # Reference Density
vel_ref = 1.0  # Reference Velocity
nu_ref = vel_ref * c_ref / Re  # Reference kinematic viscosity
vel_i = vel_ref  # Initial Velocity (set equal to velocity over the domain)
vel_in = vel_ref  # Inlet Velocity
# p_out = 0.0  # Outlet Pressure (arbitrary choice)
nu_t_wall = 0.0  # Wall eddy viscosity
nu_t_ff = 3 * nu_ref  # Wall eddy viscosity in the farfield

################# Pseudo Time-Stepping Parameters
dt0 = 1e-3  # Initial time step
dt_min = 1e-7  # Minimum dt to tolerate before killing the sim
dt_max = 1e20  # Max time step
dt_min_exit = 1e4  # Minimum dt to hit before you can exit time loop
dt_increase_fac = 2.0  # How much to increase time step if going well
dt_decrease_fac = 5.0  # How much to decrease time step if failed to converge
max_ts = 150  # Maximum number of time steps
tol_exit = 1e-3  # Exit pseudo-time stepping if relative decrease is less than this

################# Newton method parameters
max_newt = 20  # Maximum number of Newton iterations per time step
ls_order = 3  # Line search order
mu_1 = 1e-4  # Backtracking line search fac


################# Read in Mesh
with XDMFFile(MPI.COMM_WORLD, "make_bstep_mesh/backwards_step.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    cd = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

with XDMFFile(
    MPI.COMM_WORLD, "make_bstep_mesh/backwards_step_facets.xdmf", "r"
) as xdmf:
    fd = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

tdim = mesh.topology.dim
fdim = tdim - 1


ds = Measure("ds", domain=mesh, subdomain_data=fd, metadata={"quadrature_degree": 6})
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})

x = SpatialCoordinate(mesh)

# Boundary Tags
ID_In = 8
ID_Out = 9
ID_Wall = 10


################# Define Finite Element Space for Distance computation
D_el = element("Lagrange", mesh.basix_cell(), K)  # Distance Element

################# Define Distance Function Spaces
D = functionspace(mesh, D_el)

################# Define Distance Trial and Test Functions
d = TrialFunction(D)
d_test = TestFunction(D)


################# Define Distance Boundary Conditions
wall_dofs_d = locate_dofs_topological(D, fdim, fd.find(ID_Wall))
bc_wall_d = dirichletbc(PETSc.ScalarType(0.0), wall_dofs_d, D)
BCs_d = [bc_wall_d]

################# Define distance FEM Problem
a = inner(grad(d), grad(d_test)) * dx
L = d_test * dx


################# Solve distance FEM Problem
problem_poiss = LinearProblem(
    a,
    L,
    bcs=[bc_wall_d],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson_distance",
)

phi_ = problem_poiss.solve()
grad_phi_sq_ = inner(grad(phi_), grad(phi_))
# d_ = sqrt(grad_phi_sq_ + 2 * phi_) - sqrt(grad_phi_sq_)
d_ = max_value(sqrt(grad_phi_sq_ + 2 * phi_) - sqrt(grad_phi_sq_), 1e-12)

################# Define Primary Finite Element Spaces
U_el = element(
    "Lagrange", mesh.basix_cell(), K + 1, shape=(mesh.geometry.dim,)
)  # Velocity Element
P_el = element("Lagrange", mesh.basix_cell(), K)  # Pressure Element
Nu_tilda_el = element("Lagrange", mesh.basix_cell(), K)  # Working Viscosity Element


################# Define Mixed Function Spaces
W_el = mixed_element([U_el, P_el, Nu_tilda_el])  # Define Mixed element
W = functionspace(mesh, W_el)  # Define Mixed Function Space


################# Define Trial and Test Functions
du = TrialFunction(W)  # For Jacobian
up = Function(W)
u, p, nt = split(up)

vq = TestFunction(W)
v, q, z = split(vq)


################# Define Function Spaces
V_sub, V_map = W.sub(0).collapse()
D_sub, D_map = W.sub(2).collapse()

u_1 = Function(V_sub)
nt_1 = Function(D_sub)


################# Define Initial conditions
u_1.interpolate(lambda x: np.stack((np.full(x.shape[1], vel_in), np.zeros(x.shape[1]))))
nt_1.interpolate(lambda x: np.full(x.shape[1], 3 * nu_ref))

up.sub(0).interpolate(u_1)
up.sub(2).interpolate(nt_1)
up.x.scatter_forward()


################ Define Primitive Boundary Conditions
def inflow(x):
    return np.stack((np.full(x.shape[1], vel_in), np.zeros(x.shape[1])))


# def inflow(x):
#    eta = (x[1] - 0) / (1)
#    ux = 6.0 * vel_in * eta * (1.0 - eta)
#    return np.stack((ux, np.zeros_like(ux)))


def noslip(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))


nu_t_in = Constant(mesh, default_scalar_type(3 * nu_ref))  # Wrap as a constant

# Locate DOFs
in_dofs_v = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_In))
in_dofs_nu = locate_dofs_topological((W.sub(2)), fdim, fd.find(ID_In))

out_dofs_p = locate_dofs_topological((W.sub(1)), fdim, fd.find(ID_Out))
out_dofs_nu = locate_dofs_topological((W.sub(2)), fdim, fd.find(ID_Out))

wall_dofs_v = locate_dofs_topological((W.sub(0), V_sub), fdim, fd.find(ID_Wall))
wall_dofs_nu = locate_dofs_topological((W.sub(2)), fdim, fd.find(ID_Wall))


# Create BCs
u_in = Function(V_sub)
u_in.interpolate(inflow)
bc_inlet = dirichletbc(u_in, in_dofs_v, W.sub(0))
bc_inlet_nu = dirichletbc(nu_t_in, in_dofs_nu, W.sub(2))


# bc_outlet = dirichletbc(PETSc.ScalarType(p_out), out_dofs_p, W.sub(1))
# bc_outlet_nu = dirichletbc(PETSc.ScalarType(nu_t_ff), out_dofs_nu, W.sub(2))

u_wall = Function(V_sub)
u_wall.interpolate(noslip)
bc_wall = dirichletbc(u_wall, wall_dofs_v, W.sub(0))
bc_wall_nu = dirichletbc(PETSc.ScalarType(nu_t_wall), wall_dofs_nu, W.sub(2))


# Collect BCs
BCs_main = [
    bc_inlet,
    bc_wall,
    bc_inlet_nu,
    bc_wall_nu,
]


################ Weak Formulation Constants
iden = Identity(2)  # Identity Matrix
nu = Constant(mesh, default_scalar_type(nu_ref))
rho = Constant(mesh, default_scalar_type(rho_ref))
dt = Constant(mesh, default_scalar_type(dt0))
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
X = nt / nu
fv1 = (X * X * X) / (X * X * X + cv1 * cv1 * cv1)
nu_turb = nt * fv1

# Vorticity magnitude
Wij = 0.5 * (grad(u) - grad(u).T)
Omega = abs(u[1].dx(0) - u[0].dx(1))

# Other relevant terms
ft2 = ct3 * exp(-ct4 * X * X)
fv2 = 1 - (X / (1 + X * fv1))
S_tilda = max_value(Omega + (nt / (kappa * kappa * d_ * d_)) * fv2, 0.3 * Omega)
S_safe = max_value(S_tilda, 1e-12)  # Prevent div by 0
r = min_value((nt / (S_safe * kappa * kappa * d_ * d_)), 10)
g = r + cw2 * ((r**6) - r)
fw = g * ((1 + cw3**6) / (g**6 + cw3**6)) ** (1 / 6)


################ SA Term

L_SA_0 = ((nt - nt_1) / dt) * z * dx
L_SA_1 = dot(u, grad(nt)) * z * dx  # Advection
L_SA_2 = -cb1 * (1 - ft2) * S_tilda * nt * z * dx  # Production
L_SA_3 = (
    (cw1 * fw - (cb1 / (kappa * kappa)) * ft2) * (nt / d_) * (nt / d_) * z * dx
)  # Destruction
L_SA_4 = (1 / sigma) * inner((nu + nt) * grad(nt), grad(z)) * dx
L_SA_5 = -(1 / sigma) * (cb2 * inner(grad(nt), grad(nt))) * z * dx

L_SA = L_SA_1 + L_SA_2 + L_SA_3 + L_SA_4 + L_SA_5
L_SA_unst = L_SA + L_SA_0


################ Weak/Variational Form of IC NS

L_0 = inner(u - u_1, v) / dt * dx
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


################ SUPG Stabilization

# Stabilization parameters
h = CellDiameter(mesh)
mag_u = sqrt(inner(u, u) + 1e-12)
tau_mom_unst = (
    (2 / dt) * (2 / dt)
    + (2 * mag_u / h) * (2 * mag_u / h)
    + (4 * (nu + nu_turb) / (h * h)) * (4 * (nu + nu_turb) / (h * h))
) ** (-1 / 2)
tau_mom_st = (
    (2 * mag_u / h) * (2 * mag_u / h)
    + (4 * (nu + nu_turb) / (h * h)) * (4 * (nu + nu_turb) / (h * h))
) ** (-1 / 2)

tau_sa_unst = (
    (2 / dt) * (2 / dt)
    + (2 * mag_u / h) * (2 * mag_u / h)
    + (4 * ((nu + nt) / sigma) / (h * h)) * (4 * ((nu + nt) / sigma) / (h * h))
) ** (-1 / 2)
tau_sa_st = (
    (2 * mag_u / h) * (2 * mag_u / h)
    + (4 * ((nu + nt) / sigma) / (h * h)) * (4 * ((nu + nt) / sigma) / (h * h))
) ** (-1 / 2)

# Momentum equation strong form residual and stabilization
R_mom_strong_st = div(
    outer(u, u)
    + (1.0 / rho) * p * iden
    - (nu + nu_turb) * (grad(u) + grad(u).T - (2.0 / 3.0) * div(u) * iden)
)

R_mom_strong_unst = ((u - u_1) / dt) + R_mom_strong_st

L_mom_stab_st = inner(R_mom_strong_st, dot(tau_mom_st * grad(v), u)) * dx
L_mom_stab_unst = inner(R_mom_strong_unst, dot(tau_mom_unst * grad(v), u)) * dx

# nu_tilda equation strong form residual and stabilization
R_nt_strong_st = (
    dot(u, grad(nt))
    - cb1 * (1 - ft2) * S_tilda * nt
    + (cw1 * fw - (cb1 / (kappa * kappa)) * ft2) * (nt / d_) * (nt / d_)
    - (1 / sigma) * (div((nu + nt) * (grad(nt))) + (cb2 * inner(grad(nt), grad(nt))))
)

R_nt_strong_unst = R_nt_strong_st + ((nt - nt_1) / dt)

L_nt_stab_st = inner(R_nt_strong_st, dot(tau_sa_st * grad(z), u)) * dx
L_nt_stab_unst = inner(R_nt_strong_unst, dot(tau_sa_unst * grad(z), u)) * dx

L = L_1 + L_2 + L_3 + L_mom_stab_st + L_SA + L_nt_stab_st  # Steady-state sim
L_unst = L_1 + L_2 + L_3 + L_0 + L_mom_stab_unst + L_SA_unst + L_nt_stab_unst


################ Set Solver Definition
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "secant",
    "snes_linesearch_type": "bt",
    # "snes_linesearch_alpha": mu_1,
    "snes_linesearch_order": ls_order,
    "snes_atol": tol,
    "snes_rtol": tol,
    "snes_monitor": None,
    "snes_max_it": max_newt,
    "ksp_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

problem_RANS_steady = NonlinearProblem(
    L,
    up,
    bcs=BCs_main,
    petsc_options=petsc_options,
    petsc_options_prefix="RANS_Solve_steady",
)

problem_RANS_unsteady = NonlinearProblem(
    L_unst,
    up,
    bcs=BCs_main,
    petsc_options=petsc_options,
    petsc_options_prefix="RANS_Solve_unsteady",
)

# Loop over time until solution stops changing
ts_diff = Function(W)
if rank == 0:
    print("Starting pseudo time-stepping")
up_1 = up.x.array.copy()
for i in range(max_ts):
    if rank == 0:
        print("Starting time step:", i)
    # Solve unsteady problem
    problem_RANS_unsteady.solve()
    converged = problem_RANS_unsteady.solver.getConvergedReason()

    # Check if converged
    if converged > 0:
        # Check the relative change between iterations
        ts_diff.x.array[:] = up.x.array - up_1
        delta = la.norm(ts_diff.x) / la.norm(up.x)
        if rank == 0:
            print("The relative change is:", delta)

        # Proceed to steady solve if below tolerance
        if (delta < tol_exit) and (dt.value > dt_min_exit):
            break
        # Otherwise, increase dt and move to the next time step
        else:
            dt.value = min(dt.value * dt_increase_fac, dt_max)
            if rank == 0:
                print("Newton solver converged, increasing time step to dt=", dt.value)

            up_1 = up.x.array.copy()

            u_1.x.array[:] = up.x.array[V_map]
            nt_1.x.array[:] = up.x.array[D_map]

            u_1.x.scatter_forward()
            nt_1.x.scatter_forward()

    # Failed to converge, decrease time step
    else:
        dt.value /= dt_decrease_fac
        if rank == 0:
            print("Newton solver failed to converge, decreasing time step to", dt.value)

        assert dt.value > dt_min, f"dt dropped below the minimum value, exiting"

        up.x.array[:] = up_1
        up.x.scatter_forward()


# Solve a steady problem
print("Attempting steady Newton problem")
problem_RANS_steady.solve()
converged = problem_RANS_steady.solver.getConvergedReason()
num_iter = problem_RANS_steady.solver.getIterationNumber()
assert converged > 0, f"Solver did not converge, got {converged}."
print(
    f"Solver converged after {num_iter} iterations with converged reason {converged}."
)


################ Time Loop
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

dist_func = Function(D)
dist_func.interpolate(Expression(d_, D.element.interpolation_points))
dist_func.x.scatter_forward()
vtx_d = VTXWriter(
    mesh.comm,
    folder / ("Distance" + "_.bp"),
    [dist_func],
    engine="BP4",
)

vtx_u.write(0.0)
vtx_p.write(0.0)
vtx_nu.write(0.0)
vtx_d.write(0.0)


quit()
