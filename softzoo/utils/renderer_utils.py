from skvideo.io import FFmpegWriter
from typing import Optional

def get_video_writer(filename: str, rate: Optional[int] = 30):
    rate = str(rate)
    return FFmpegWriter(filename,
                        inputdict={'-r': rate},
                        outputdict={# '-vcodec': 'h265', #'mpeg4',
                                    '-pix_fmt': 'yuv420p',
                                    '-r': rate})