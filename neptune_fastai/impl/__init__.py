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
import warnings
from typing import List

from fastai.learner import Learner
from fastai.callback.hook import total_params
from fastai.callback.tracker import SaveModelCallback
from fastai.basics import Callback, store_attr, join_path_file
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
    order = SaveModelCallback.order + 1

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
        self.best_model_epoch = 0
        self.fit_index = _retrieve_fit_index(run, f'{base_namespace}/metrics/')

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
    def _frozen_level(self):
        return self.opt.frozen_idx if hasattr(self, 'opt') and hasattr(self.opt, 'frozen_idx') else 0

    @property
    def name(self) -> str:
        return 'neptune'

    def before_fit(self):
        self.neptune_run[f'{self.base_namespace}/config'] = {
            'device': self._device,
            'batch_size': self._batch_size,
            'model': {
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
            'criterion': self._optimizer_criterion,
            'optimizer': {
                'name': self._optimizer_name,
            }
        }

        _log_model_architecture(self.neptune_run, self.base_namespace, self.learn)
        _log_dataset_metadata(self.neptune_run, self.base_namespace, self.learn)

        self.neptune_run[f'{self.base_namespace}/config/optimizer/initial_hyperparameters'] = self._optimizer_hyperparams

        prefix = f'{self.base_namespace}/metrics/fit_{self.fit_index}'

        self.neptune_run[f'{prefix}/n_epoch'] = self.n_epoch

        if self._frozen_level > 0:
            self.neptune_run[f'{prefix}/frozen_level'] = self._frozen_level

        every_epoch = self.save_model_freq > 0
        if hasattr(self, 'save_model') and every_epoch and not self.save_model.every_epoch:
            warnings.warn(
                'NeptuneCallback: SaveModelCallback is required to have every_epoch set to True when using '
                'save_model_freq. Model checkpoints will not be uploaded.'
            )
            self.save_model_freq = 0

        elif every_epoch or self.save_best_model:
            self.cbs.add(SaveModelCallback(every_epoch=every_epoch))

    def before_batch(self):
        target = 'training'
        if not self.learn.training:
            target = 'validation'

        if self.learn.iter == 0:
            self.neptune_run[f'{self.base_namespace}/metrics/fit_{self.fit_index}/{target}/n_iter'] = self.n_iter

        if self.learn.train_iter == 1:
            self.neptune_run[f'{self.base_namespace}/config/input_shape'] = {
                'x': str(list(self.x[0].shape)),
                'y': 1 if len(self.y[0].shape) == 0 else str(list(self.y[0].shape))
            }

    def after_batch(self):
        target = 'training'
        if not self.learn.training:
            target = 'validation'

        prefix = f'{self.base_namespace}/metrics/fit_{self.fit_index}/{target}/batch'

        self.neptune_run[f'{prefix}/loss'].log(
            value=self.learn.loss.clone()
        )

        if hasattr(self, 'smooth_loss'):
            self.neptune_run[f'{prefix}/smooth_loss'].log(
                value=self.learn.smooth_loss.clone()
            )

    def after_train(self):
        prefix = f'{self.base_namespace}/metrics/fit_{self.fit_index}/training/epoch'

        for metric_name, metric_value in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if metric_name not in {'epoch', 'time'}:
                self.neptune_run[f'{prefix}/{metric_name}'].log(metric_value)

        self.neptune_run[f'{prefix}/duration'].log(
            value=time.time() - self.learn.recorder.start_epoch
        )

        _log_optimizer_hyperparams(self.neptune_run,
                                   f'{prefix}/optimizer_hyperparameters',
                                   self._optimizer_hyperparams,
                                   self.n_epoch)

    def after_validate(self):
        prefix = f'{self.base_namespace}/metrics/fit_{self.fit_index}/validation/epoch'

        for metric_name, metric_value in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if metric_name not in {'epoch', 'time', 'train_loss'}:
                self.neptune_run[f'{prefix}/{metric_name}'].log(metric_value)

        self.neptune_run[f'{prefix}/duration'].log(
            value=time.time() - self.learn.recorder.start_epoch
        )

    def after_epoch(self):
        if hasattr(self, 'save_model') and hasattr(self.save_model, 'every_epoch') and self.save_model.every_epoch:
            if self.save_model_freq > 0:
                if self.epoch % self.save_model_freq == 0:
                    path = join_path_file(f'{self.learn.save_model.fname}_{self.learn.save_model.epoch}',
                                          self.learn.path / self.learn.model_dir,
                                          ext='.pth')
                    prefix = f'{self.base_namespace}/io_files/artifacts/model_checkpoints/fit_{self.fit_index}/epoch_{self.learn.epoch}'
                    self.neptune_run[prefix].upload(str(path))

            if self.save_best_model:
                super(type(self.save_model), self.save_model).after_epoch()

                if hasattr(self.save_model, 'new_best') and self.save_model.new_best:
                    self.best_model_epoch = self.epoch

    def after_fit(self):
        if self.save_best_model:
            if hasattr(self, 'save_model') and hasattr(self.save_model, 'every_epoch') and self.save_model.every_epoch:
                path = join_path_file(f'{self.learn.save_model.fname}_{self.best_model_epoch}',
                                      self.learn.path / self.learn.model_dir,
                                      ext='.pth')
            else:
                path = join_path_file(f'{self.learn.save_model.fname}', self.learn.path / self.learn.model_dir, ext='.pth')

            prefix = f'{self.base_namespace}/io_files/artifacts/model_checkpoints/fit_{self.fit_index}/best'
            self.neptune_run[prefix].upload(str(path))

        self.fit_index += 1


def _log_model_architecture(run: neptune.Run, base_namespace: str, learn: Learner):
    if hasattr(learn, 'arch'):
        run[f'{base_namespace}/config/model/architecture_name'] = getattr(learn.arch, '__name__', '')

    model = File.from_content(repr(learn.model))

    run[f'{base_namespace}/config/model/architecture'].upload(model)
    run[f'{base_namespace}/io_files/artifacts/model_architecture'].upload(model)


def _log_dataset_metadata(run: neptune.Run, base_namespace: str, learn: Learner):
    sha = hashlib.sha1(str(learn.dls.path).encode())

    run[f'{base_namespace}/io_files/resources/dataset'] = {
        'path': learn.dls.path,
        'size': learn.dls.n,
        'sha': sha.hexdigest(),
    }


def _log_or_assign_metric(run: neptune.Run, number_of_epochs: int, metric: str, value):
    if number_of_epochs > 1:
        run[metric].log(value)
    else:
        run[metric] = value


def _retrieve_fit_index(run: neptune.Run, path: str):
    return len(run.get_attribute(path) or [])


def _log_optimizer_hyperparams(run: neptune.Run, prefix: str, optimizer_hyperparams: dict, n_epoch: int):
    for param, value in optimizer_hyperparams.items():
        _log_or_assign_metric(
            run,
            n_epoch,
            f'{prefix}/{param}',
            value
        )
