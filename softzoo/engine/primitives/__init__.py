from functools import partial
from typing import TYPE_CHECKING,Union

from .primitive_base import PrimitiveBase
from .box import Box
from .sphere import Sphere
from .capsule import Capsule
from .mesh import Mesh
from .ellipsoid import Ellipsoid

from ...configs.config_dataclass import(
    PrimitiveBaseConfig
)
from ...utils.general_utils import Enum

if TYPE_CHECKING:
    from ..mpm_solver import MPMSolver

class Primitive(Enum):
    Box = 0
    Sphere = 1
    Capsule = 2
    Mesh = 3
    Ellipsoid = 4
    
def add(name, solver:'MPMSolver', cfg:PrimitiveBaseConfig):
    if name == 'Box':
        solver.primitives.append(Box(solver, cfg))
    elif name == 'Sphere':
        solver.primitives.append(Sphere(solver, cfg))
    elif name == 'Capsule':
        solver.primitives.append(Capsule(solver, cfg))
    elif name == 'Mesh':
        solver.primitives.append(Mesh(solver, cfg))
    elif name == 'Ellipsoid':
        solver.primitives.append(Ellipsoid(solver, cfg))
        
for member in list(Primitive.__members__):
    member_add = partial(add, name=member)
    member = getattr(Primitive, member)
    setattr(member, 'add', member_add)