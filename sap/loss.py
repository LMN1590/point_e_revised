import torch
import torch.nn.functional as F

from pytorch3d.loss.chamfer import _apply_batch_reduction,_chamfer_distance_single_direction,_handle_pointcloud_input,_validate_chamfer_reduction_inputs
from pytorch3d.ops.knn import knn_gather, knn_points

from typing import Union,Optional

def chamfer_distance_surface(
    x_0:torch.Tensor,
    v_mesh:torch.Tensor,
    x_0_weight:Optional[torch.Tensor],
    v_mesh_weight:Optional[torch.Tensor],
    x_0_lengths=None,
    v_mesh_lengths=None,
    x_0_normals=None,
    v_mesh_normals=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm:int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    if point_reduction == "max" and (x_0_normals is not None or v_mesh_normals is not None):
        raise ValueError('Normals must be None if point_reduction is "max"')

    x_0,x_0_lengths,x_0_normals = _handle_pointcloud_input(x_0, x_0_lengths, x_0_normals)
    v_mesh, v_mesh_lengths,v_mesh_normals = _handle_pointcloud_input(v_mesh, v_mesh_lengths, v_mesh_normals)
    
    N, P1, D = x_0.shape
    if x_0_weight is None:
        weights_x0 = None
    elif x_0_weight.dim() == 1 and x_0_weight.shape[0] == N:
        weights_x0 = x_0_weight.view(N,1).expand(-1,P1)
    elif x_0_weight.dim() == 2 and x_0_weight.shape == (N,P1):
        weights_x0 = x_0_weight

    cham_x_0, cham_norm_x_0 = _chamfer_distance_single_direction_modded_weight(
        x_0,
        v_mesh,
        x_0_lengths,
        v_mesh_lengths,
        x_0_normals,
        v_mesh_normals,
        weights_x0,
        point_reduction,
        norm,
        abs_cosine,
    )
    
    if single_directional:
        loss = cham_x_0
        loss_normals = cham_norm_x_0
    else:
        N, P2, D = v_mesh.shape
        if v_mesh_weight is None:
            weights_v_mesh = None
        elif v_mesh_weight.dim() == 1 and v_mesh_weight.shape[0] == N:
            weights_v_mesh = v_mesh_weight.view(N,1).expand(-1,P2)
        elif v_mesh_weight.dim() == 2 and v_mesh_weight.shape == (N,P2):
            weights_v_mesh = v_mesh_weight
        
        cham_v_mesh, cham_norm_v_mesh = _chamfer_distance_single_direction_modded_weight(
            v_mesh,
            x_0,
            v_mesh_lengths,
            x_0_lengths,
            v_mesh_normals,
            x_0_normals,
            weights_v_mesh,
            point_reduction,
            norm,
            abs_cosine,
        )
        if point_reduction == "max":
            loss = torch.maximum(cham_x_0, cham_v_mesh)
            loss_normals = None
        elif point_reduction is not None:
            loss = cham_x_0 + cham_v_mesh
            if cham_norm_x_0 is not None:
                loss_normals = cham_norm_x_0 + cham_norm_v_mesh
            else:
                loss_normals = None
        else:
            loss = (cham_x_0, cham_v_mesh)
            if cham_norm_x_0 is not None:
                loss_normals = (cham_norm_x_0, cham_norm_v_mesh)
            else:
                loss_normals = None
    weights = None
    return _apply_batch_reduction(loss, loss_normals, weights, batch_reduction)

    
    
def _chamfer_distance_single_direction_modded_weight(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals,
    y_normals,
    weights, # (N,P1)
    point_reduction: Union[str, None],
    norm: int,
    abs_cosine: bool,
):
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if weights is not None:
        cham_x *= weights
        
    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction == "max":
        assert not return_normals
        cham_x = cham_x.max(1).values  # (N,)
    elif point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals
    