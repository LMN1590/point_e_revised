import numpy as np
import taichi as ti

from typing import TYPE_CHECKING,Union

from .primitive_base import PrimitiveBase
from ...utils.const import I_DTYPE, F_DTYPE, NORM_EPS
from ..materials import Material
from ...configs.config_dataclass import BoxConfig

if TYPE_CHECKING:
    from ..mpm_solver import MPMSolver

@ti.data_oriented
class Capsule(PrimitiveBase):
    @classmethod
    def default_cfg(cls):
        cfg = PrimitiveBase.default_cfg()
        cfg.height = 0.1
        cfg.radius = 0.03
        
        return cfg

    def __init__(self, solver:'MPMSolver', cfg):
        super().__init__(solver, cfg)
        assert len(self.size) == self.solver.dim and len(self.size) == self.solver.dim

        # Instantiate shape in MPM
        if not self.is_rigid:
            self.material = getattr(Material, self.cfg.material)
            raise NotImplementedError # TODO

    @ti.kernel
    def seed(self):
        raise NotImplementedError # TODO

    def _inv_inertia(self):
        # Ref: https://www.gamedev.net/tutorials/programming/math-and-physics/capsule-inertia-tensor-r3856/
        raise NotImplementedError # TODO

    @ti.func
    def _sdf(self, s, grid_pos):
        out = grid_pos
        out[1] += self.height / 2
        out[1] -= min(max(out[1], 0.0), self.height)
        return out.norm(NORM_EPS) - self.radius

    @ti.func
    def _normal(self, s, grid_pos):
        out = grid_pos
        out[1] += self.height / 2
        out[1] -= min(max(out[1], 0.0), self.height)
        return out.normalized(NORM_EPS)
