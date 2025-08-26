import torch

import os
import matplotlib.pyplot as plt
from tqdm import trange
from scipy import ndimage

def visualize_psr_grid(psr_grid, pose=None, out_dir=None, out_video_name='video.mp4'):
    if pose is not None:
        device = psr_grid.device
        # get world coordinate of grid points [-1, 1]
        res = psr_grid.shape[-1]
        x = torch.linspace(-1, 1, steps=res)
        co_x, co_y, co_z = torch.meshgrid(x, x, x)
        co_grid = torch.stack(
                [co_x.reshape(-1), co_y.reshape(-1), co_z.reshape(-1)], 
                dim=1).to(device).unsqueeze(0)

        # visualize the projected occ_soft value
        res = 128
        psr_grid = psr_grid.reshape(-1)
        out_mask = psr_grid>0
        in_mask = psr_grid<0
        pix = pose.transform_points_screen(co_grid, ((res, res),))[..., :2].round().long().squeeze()
        vis_mask = (pix[..., 0]>=0) & (pix[..., 0]<=res-1) & \
                    (pix[..., 1]>=0) & (pix[..., 1]<=res-1)
        pix_out = pix[vis_mask & out_mask]
        pix_in = pix[vis_mask & in_mask]
        
        img = torch.ones([res,res]).to(device)
        psr_grid = torch.sigmoid(- psr_grid * 5)
        img[pix_out[:, 1], pix_out[:, 0]] = psr_grid[vis_mask & out_mask]
        img[pix_in[:, 1], pix_in[:, 0]] = psr_grid[vis_mask & in_mask]
        # save_image(img, 'tmp.png', normalize=True)
        return img
    elif out_dir is not None:
        dir_psr_vis = out_dir
        os.makedirs(dir_psr_vis, exist_ok=True)
        psr_grid = psr_grid.squeeze().detach().cpu().numpy()
        axis = ['x', 'y', 'z']
        s = psr_grid.shape[0]
        for idx in trange(s):
            my_dpi = 100
            plt.figure(figsize=(1000/my_dpi, 300/my_dpi), dpi=my_dpi)
            plt.subplot(1, 3, 1)
            plt.imshow(ndimage.rotate(psr_grid[idx], 180, mode='nearest'), cmap='nipy_spectral')
            plt.clim(-1, 1)
            plt.colorbar()
            plt.title('x')
            plt.grid("off")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(ndimage.rotate(psr_grid[:, idx], 180, mode='nearest'), cmap='nipy_spectral')
            plt.clim(-1, 1)
            plt.colorbar()
            plt.title('y')
            plt.grid("off")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(ndimage.rotate(psr_grid[:,:,idx], 90, mode='nearest'), cmap='nipy_spectral')
            plt.clim(-1, 1)
            plt.colorbar()
            plt.title('z')
            plt.grid("off")
            plt.axis("off")


            plt.savefig(os.path.join(dir_psr_vis, '{}'.format(idx)), pad_inches = 0, dpi=100)
            plt.close()
        os.system("rm {}/{}".format(dir_psr_vis, out_video_name))
        os.system("ffmpeg -framerate 25 -start_number 0 -i {}/%d.png -pix_fmt yuv420p -crf 17 {}/{}".format(dir_psr_vis, dir_psr_vis, out_video_name))