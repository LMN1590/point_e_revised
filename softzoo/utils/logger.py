import os
from torch.utils.tensorboard import SummaryWriter
import pprint
import time
from collections import deque
import numpy as np

class Logger:
    def __init__(self, logdir, write_mode='w', with_tensorboard=False, suffix=""):
        self._text_writer = open(os.path.join(logdir, f'results{suffix}.txt'),
                                 write_mode)
        if with_tensorboard:
            self._tf_writer = SummaryWriter(logdir)
        self._logdir = logdir
        self._scalars = {'default': {}}
        self._timer = {'default': {}}
        self._accum_timer = {'default': {}}

        self._deque_maxlen = 100

    def write(self, iter, group='default', use_scientific_notation=True):
        text = [f'iter#{iter}']
        for name, value in self._scalars[group].items():
            if hasattr(self, '_tf_writer'):
                self._tf_writer.add_scalar(f'{group}/{name}', value, iter)
            if use_scientific_notation:
                text.append(f'{name}: {value:.4e}')
            else:
                text.append(f'{name}: {value}')

        for name, value_list in self._accum_timer[group].items():
            value = np.mean(value_list)
            if hasattr(self, '_tf_writer'):
                self._tf_writer.add_scalar(f'{group}/{name}', value, iter)
            if use_scientific_notation:
                text.append(f't/{name}: {value:.4e}')
            else:
                text.append(f't/{name}: {value}')

        self.print('  '.join(text))

    def reset(self, group='default'):
        for name in self._accum_timer[group].keys():
            self._accum_timer[group][name] = deque(maxlen=self._deque_maxlen)

    def print(self, data):
        if not isinstance(data, str):
            data = pprint.pformat(data)
        self._text_writer.write(data + '\n')
        self._text_writer.flush()
        print(data)

    def scalar(self, name, value, group=None):
        group = 'default' if group is None else group
        self._scalars[group][name] = float(value)

    def tic(self, name, group=None):
        group = 'default' if group is None else group
        self._timer[group][name] = [time.time()]

    def toc(self, name, group=None):
        group = 'default' if group is None else group
        assert len(
            self._timer[group][name]) == 1, f'Should call tic({name}) first'
        self._timer[group][name].append(time.time())

        if not name in self._accum_timer[group].keys():
            self._accum_timer[group][name] = deque(maxlen=self._deque_maxlen)
        tic, toc = self._timer[group][name]
        self._accum_timer[group][name].append(toc - tic)

    def create_group(self, group):
        self._scalars[group] = {}
        self._timer[group] = {}
        self._accum_timer[group] = {}

    def close(self):
        self._text_writer.close()
        if hasattr(self, '_tf_writer'):
            self._tf_writer.close()