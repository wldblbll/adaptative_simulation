"""adaptfem: adaptive allocation of computational effort in incremental nonlinear FEM.

Research code. Plane-strain Q4 elements, J2 plasticity with isotropic (linear + Voce)
hardening, radial return with consistent tangent, incremental Newton-Raphson, and full
cost instrumentation. The orchestrator (Phase 2) and the learned policies (Phase 3) are
layered on top of the same solver, so every comparison uses the same mesh, discretisation
and integrator.
"""
from .material import J2Material
from .mesh import rect_mesh, notched_plate_mesh
from .element import Q4Elements
from .solver import FEModel, Cost, State
from . import cases

__all__ = ["J2Material", "rect_mesh", "notched_plate_mesh", "Q4Elements", "FEModel",
           "Cost", "State", "cases"]
