import torch

def finger_penetration_loss(fingers_pos:torch.Tensor, dist_threshold:float = 0.15):
    """
    Compute the intersection loss between multiple fingers based on their positions.
    Parameters:
        finger_pos: torch.Tensor: A tensor of shape (num_fingers,num_points,3) representing the positions of points on each finger.
        dist_threshold: float: The distance threshold below which the loss is applied
    """
    
    num_fingers = fingers_pos.shape[0]
    loss = torch.zeros(1, device=fingers_pos.device)
    for i in range(num_fingers):
        for j in range(i+1, num_fingers):
            dist = torch.cdist(fingers_pos[i], fingers_pos[j])  # (num_points,num_points)
            penetration_loss  = torch.sum(torch.clamp(dist_threshold-dist,min=0.0))
            loss += penetration_loss/(dist.shape[0]*dist.shape[1])  # average over point pairs
    return loss / (num_fingers*(num_fingers-1)/2)  # average over finger pairs

    
    