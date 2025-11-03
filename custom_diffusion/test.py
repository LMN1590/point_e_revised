from .config.diffusion_dataclass import DIFFUSION_CONFIGS
from .diffusion.configs import diffusion_from_config

if __name__ == "__main__":
    finger_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['custom_finger_diffusion'])