import torch

def disp_to_dist_and_depth(disp_map: torch.Tensor, rays: torch.Tensor, fov: float, baseline: float, calc_z_depth=True):
    """
    Converts disparity map to a Euclidean distance map and a z-depth map

    This function was built upon the code of from_dist of
    https://github.com/jseuffert90/map_processing/blob/main/map_converter.py (commit 6089125).
    Please confer the map_converter.py for more documentation.

    Parameters
    ----------
    disp_map : torch.Tensor
        the disparity map of shape (BS, H, W)
    rays : torch.Tensor
        Light ray directions of shape (H, W, 1) for each pixel (conf. map_converter.py from map_processing)
    fov : float
        field of view in radians
    baseline : float
        length of baseline (same unit as distance and depth maps)
    calc_z_depth : bool
        if False, does not calculate the z depth and returns None as a surrogate for the z depth map

    Returns
    -------
    list
        a list containing the Euc. distance maps and the z-depth maps
    """

    height, width = disp_map.shape[-2:]
    assert disp_map.dim() == 3
    assert rays.dim() == 3
    
    rays = rays[None] # shape (1, H, W, 3)
    cos_angle = -rays[:, :, :, 0]
    cos_angle[cos_angle > 1] = 1
    cos_angle[cos_angle < -1] = -1
    beta_l = torch.arccos(cos_angle)
    beta_r = beta_l - disp_map
       
    dist_map = torch.sin(beta_r) * baseline / torch.sin(disp_map) # shape (BS, H, W)
    if calc_z_depth:
        ordered_point_cloud = rays * dist_map[:, :, :, None] # shape (BS, H, W, 3)
        z_depth_map = ordered_point_cloud[:, :, :, 2] # shape (BS, H, W)
    else:
        z_depth_map = None

    return dist_map, z_depth_map
