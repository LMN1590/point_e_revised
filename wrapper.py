import os

ROOT_DIR = 'config/cone_check'
config_files = [
    os.path.join(ROOT_DIR,dir)
    for dir in os.listdir(ROOT_DIR)
    if dir.split('.')[-1] == 'yaml'
]

for file in config_files:
    os.system(f'python run.py --config_path {file}')
    # print(f'python run.py --config_path {file}')