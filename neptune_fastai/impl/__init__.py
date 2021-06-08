#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

__all__ = [
    'NeptuneCallback'
]

import time
import hashlib
from typing import List

from fastai.learner import Learner
from fastai.callback.hook import total_params
from fastai.basics import Callback, store_attr
from fastai.callback.tracker import SaveModelCallback
from fastai.torch_core import trainable_params, default_device

try:
    # neptune-client=0.9.0 package structure
    import neptune.new as neptune
    from neptune.new.internal.utils import verify_type
    from neptune.new.types import File
except ImportError:
    # neptune-client=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type
    from neptune.types import File

from neptune_fastai import __version__

INTEGRATION_VERSION_KEY = 'source_code/integrations/neptune-fastai'


class NeptuneCallback(Callback):
    def __init__(self,
                 run: neptune.Run,
                 base_namespace: str = '',
                 save_best_model: bool = True,
                 save_model_freq: int = 0,
                 **kwargs):
        super().__init__(**kwargs)

        verify_type('run', run, neptune.Run)
        verify_type('base_namespace', base_namespace, str)
        verify_type('save_best_model', save_best_model, bool)
        verify_type('save_model_freq', save_model_freq, int)

        self.neptune_run = run

        run[INTEGRATION_VERSION_KEY] = __version__

        store_attr('base_namespace,save_best_model,save_model_freq')

    @property
    def _batch_size(self) -> int:
        return self.dls.bs

    @property
    def _optimizer_name(self) -> str:
        return self.opt_func.__name__

    @property
    def _device(self):
        return default_device() or getattr(self.dls, 'device', default_device())

    @property
    def _vocab(self) -> List[str]:
        return self.dls.vocab

    @property
    def _total_model_parameters(self) -> int:
        params, _ = total_params(self.learn)
        return params

    @property
    def _trainable_model_parameters(self) -> int:
        return sum([p.numel() for p in trainable_params(self.learn)])

    @property
    def _optimizer_criterion(self) -> str:
        return repr(self.loss_func.func)

    @property
    def _optimizer_hyperparams(self):
        if len(self.learn.opt.hypers) == 1:
            return dict(self.learn.opt.hypers[0])

        return {
            f'group_layer_{layer}/{hyper}': value
            for layer, opts in enumerate(self.learn.opt.hypers)
            for hyper, value in opts.items()
        }

    @property
    def name(self) -> str:
        return 'Neptune'

    def after_create(self):
        if self.save_best_model:
            if not hasattr(self, 'save_model'):
                self.learn.add_cb(SaveModelCallback())

            self.learn.save = _log_model(
                self.learn.save,
                self.neptune_run,
                self.learn
            )

    def before_fit(self):
        self.neptune_run['config'] = {
            'n_epoch': self.n_epoch,
            'device': self._device,
            'model': {
                'batch_size': self._batch_size,
                'vocab': {
                    'details': self._vocab,
                    'total': len(self._vocab)
                },
                'params': {
                    'total': self._total_model_parameters,
                    'trainable_params': self._trainable_model_parameters,
                    'non_trainable_params': self._total_model_parameters - self._trainable_model_parameters
                },
            },
            'optimizer': {
                'name': self._optimizer_name,
                'criterion': self._optimizer_criterion,
                'initial_hyperparameters': self._optimizer_hyperparams,
            }
        }

        _log_model_architecture(self.neptune_run, self.learn)
        _log_dataset_metadata(self.neptune_run, self.learn)

    def before_batch(self):
        if self.learn.train_iter == 1:
            self.neptune_run['config/input_shape'] = {
                'x': str(list(self.x[0].shape)),
                'y': 1 if len(self.y[0].shape) == 0 else str(list(self.y[0].shape))
            }

    def after_batch(self):
        if self.learn.training:
            self.neptune_run['logs/training/batch/loss'].log(value=self.learn.loss.clone())

            if hasattr(self, 'smooth_loss'):
                self.neptune_run['logs/training/batch/smooth_loss'].log(value=self.learn.smooth_loss.clone())

    def after_epoch(self):
        for metric_name, metric_value in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if metric_name not in {'epoch', 'time'}:
                self.neptune_run[f'logs/training/epoch/{metric_name}'].log(metric_value)
        self.neptune_run['logs/training/epoch/duration'].log(value=time.time() - self.learn.recorder.start_epoch)

        for param, value in self._optimizer_hyperparams.items():
            self.neptune_run[f'logs/training/epoch/optimizer_hyperparameters/{param}'].log(value)

        if self.n_epoch > 1 and self.save_model_freq > 0 and self.save_best_model:
            if self.epoch % self.save_model_freq == 0:
                self.learn.save(f'{self.learn.save_model.fname}')


def _log_model_architecture(run: neptune.Run, learn: Learner):
    if hasattr(learn, 'arch'):
        run['config/model/architecture_name'] = getattr(learn.arch, '__name__', '')

    model = File.from_content(repr(learn.model))

    run['config/model/architecture'].upload(model)
    run['io_files/artifacts/model_architecture'].upload(model)


def _log_dataset_metadata(run: neptune.Run, learn: Learner):
    sha = hashlib.sha1(str(learn.dls.path).encode())

    run['io_files/resources/dataset'] = {
        'path': learn.dls.path,
        'size': learn.dls.n,
        'sha': sha.hexdigest(),
    }


def _log_model(save, run: neptune.Run, learn: Learner):
    def _save_model_logger(*args, **kwargs):
        path = save(*args, **kwargs)

        if path is not None and path.exists():
            run['io_files/artifacts/model/best'].upload(str(path), wait=True)
            run[f'io_files/artifacts/model/epoch_{learn.epoch}'].upload(str(path), wait=True)
    return _save_model_logger
