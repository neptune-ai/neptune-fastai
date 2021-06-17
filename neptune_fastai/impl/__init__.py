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
    'NeptuneCallback',
    'retrieve_fit_index'
]

import os
import time
import hashlib
import warnings
from typing import List, Optional, Dict, Callable, Union

from fastai.learner import Learner
from fastai.callback.hook import total_params
from fastai.basics import store_attr, join_path_file
from fastai.torch_core import trainable_params, default_device
from fastai.callback.tracker import TrackerCallback, SaveModelCallback

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


class NeptuneCallback(TrackerCallback):
    order = SaveModelCallback.order + 1

    def __init__(self,
                 run: neptune.Run,
                 base_namespace: str = '',
                 fname: str = 'neptune_model',
                 monitor: Union[str, Callable] = 'valid_loss',
                 comp: Optional[Callable] = None,
                 min_delta: float = 0.0,
                 reset_on_fit: bool = True,
                 cleanup_after_fit: bool = True,
                 with_opt: bool = False,
                 save_best_model: bool = True,
                 save_model_freq: int = 0):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)

        verify_type('run', run, neptune.Run)
        verify_type('base_namespace', base_namespace, str)
        verify_type('fname', fname, str)
        verify_type('min_delta', min_delta, float)
        verify_type('reset_on_fit', reset_on_fit, bool)
        verify_type('cleanup_after_fit', cleanup_after_fit, bool)
        verify_type('with_opt', with_opt, bool)
        verify_type('save_best_model', save_best_model, bool)
        verify_type('save_model_freq', save_model_freq, int)

        self.neptune_run = run
        self._saved_files = set()
        self._warned_save_model = False
        self.save_model_freq = save_model_freq
        self.fit_index = retrieve_fit_index(run, f'{base_namespace}/metrics/')

        run[INTEGRATION_VERSION_KEY] = __version__

        store_attr('base_namespace,fname,cleanup_after_fit,with_opt,save_best_model,save_model_freq')

    @property
    def name(self) -> str:
        return 'neptune'

    @property
    def _filename(self) -> str:
        return f'{self.fname}_fit_{self.fit_index}'

    @property
    def _batch_size(self) -> int:
        return self.dls.bs

    @property
    def _optimizer_name(self) -> Optional[str]:
        return self.opt_func.__name__

    @property
    def _device(self) -> str:
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
    def _optimizer_hyperparams(self) -> Optional[dict]:
        if hasattr(self, 'opt') and hasattr(self.opt, 'hypers'):
            if len(self.opt.hypers) == 1:
                return dict(self.learn.opt.hypers[0])

            return {
                f'group_layer_{layer}/{hyper}': value
                for layer, opts in enumerate(self.learn.opt.hypers)
                for hyper, value in opts.items()
            }

    @property
    def _frozen_level(self) -> int:
        return self.opt.frozen_idx if hasattr(self, 'opt') and hasattr(self.opt, 'frozen_idx') else 0

    @property
    def _input_shape(self) -> Dict:
        if hasattr(self, 'x') and hasattr(self, 'y'):
            return {
                'x': str(list(self.x[0].shape)),
                'y': 1 if len(self.y[0].shape) == 0 else str(list(self.y[0].shape))
            }

    @property
    def _target(self) -> str:
        return 'training' if self.learn.training else 'validation'

    def _log_model_configuration(self):
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
                'initial_hyperparameters': self._optimizer_hyperparams
            }
        }

    def _save(self, filename: str):
        self.learn.save(filename, with_opt=self.with_opt)

        path = str(join_path_file(filename, self.learn.path / self.learn.model_dir, ext='.pth'))
        self._saved_files.add(path)

        return path

    def _clean_saved_files(self):
        for saved_file in self._saved_files:
            try:
                os.remove(saved_file)
            except FileNotFoundError:
                pass

    def _check_save_model(self):
        if not self._warned_save_model and hasattr(self, 'save_model') and\
                (self.save_best_model or self.save_model_freq > 0):
            warnings.warn(f'NeptuneCallback: Extra model weight files will be stored temporarily in your machine '
                          f'addition to the ones from your SaveModelCallback. To avoid running out of storage remove '
                          f'SaveModelCallback.')
            self._warned_save_model = True

    def before_fit(self):
        super().before_fit()

        self._check_save_model()
        self._log_model_configuration()

        _log_model_architecture(self.neptune_run, self.base_namespace, self.learn)
        _log_dataset_metadata(self.neptune_run, self.base_namespace, self.learn)

        prefix = f'{self.base_namespace}/metrics/fit_{self.fit_index}'

        self.neptune_run[f'{prefix}/n_epoch'] = self.n_epoch

        if self._frozen_level > 0:
            self.neptune_run[f'{prefix}/frozen_level'] = self._frozen_level

    def before_batch(self):
        if self.learn.iter == 0:
            self.neptune_run[
                f'{self.base_namespace}/metrics/fit_{self.fit_index}/{self._target}/batch_counter'
            ] = self.n_iter

        if self.learn.train_iter == 1:
            self.neptune_run[f'{self.base_namespace}/config/input_shape'] = self._input_shape

    def after_batch(self):
        prefix = f'{self.base_namespace}/metrics/fit_{self.fit_index}/{self._target}/batch'

        self.neptune_run[f'{prefix}/loss'].log(value=self.learn.loss.clone())

        if hasattr(self, 'smooth_loss'):
            self.neptune_run[f'{prefix}/smooth_loss'].log(value=self.learn.smooth_loss.clone())

    def after_train(self):
        prefix = f'{self.base_namespace}/metrics/fit_{self.fit_index}/training/loader'

        for metric_name, metric_value in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if metric_name not in {'epoch', 'time'}:
                self.neptune_run[f'{prefix}/{metric_name}'].log(metric_value)

        self.neptune_run[f'{prefix}/duration'].log(
            value=time.time() - self.learn.recorder.start_epoch
        )

        _log_optimizer_hyperparams(self.neptune_run,
                                   f'{self.base_namespace}/metrics/fit_{self.fit_index}/optimizer_hyperparameters',
                                   self._optimizer_hyperparams,
                                   self.n_epoch)

    def after_validate(self):
        prefix = f'{self.base_namespace}/metrics/fit_{self.fit_index}/validation/loader'

        for metric_name, metric_value in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if metric_name not in {'epoch', 'time', 'train_loss'}:
                self.neptune_run[f'{prefix}/{metric_name}'].log(metric_value)

        self.neptune_run[f'{prefix}/duration'].log(
            value=time.time() - self.learn.recorder.start_epoch
        )

    def after_epoch(self):
        if self.save_model_freq > 0 and self.epoch % self.save_model_freq == 0:
            path = self._save(f'{self._filename}_epoch_{self.epoch}')
            prefix = f'{self.base_namespace}/io_files/artifacts/model_checkpoints/fit_{self.fit_index}/' \
                     f'epoch_{self.epoch}'
            self.neptune_run[prefix].upload(path)

        if self.save_best_model:
            # Enforce tracker to check for new best model
            super().after_epoch()

            if self.new_best:
                self._save(self._filename)

    def after_fit(self):
        if self.save_best_model:
            path = join_path_file(self._filename,
                                  self.learn.path / self.learn.model_dir,
                                  ext='.pth')
            prefix = f'{self.base_namespace}/io_files/artifacts/model_checkpoints/fit_{self.fit_index}/best'

            self.neptune_run[prefix].upload(str(path))

        self.neptune_run.sync(wait=True)

        if self.cleanup_after_fit:
            self._clean_saved_files()

        self.fit_index += 1


def _log_model_architecture(run: neptune.Run, base_namespace: str, learn: Learner):
    if hasattr(learn, 'arch'):
        run[f'{base_namespace}/config/model/architecture_name'] = getattr(learn.arch, '__name__', '')

    model_architecture = File.from_content(repr(learn.model))

    run[f'{base_namespace}/config/model/architecture'].upload(model_architecture)
    run[f'{base_namespace}/io_files/artifacts/model_architecture'].upload(model_architecture)


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


def retrieve_fit_index(run: neptune.Run, path: str) -> int:
    return len(run.get_attribute(path) or [])


def _log_optimizer_hyperparams(run: neptune.Run, prefix: str, optimizer_hyperparams: dict, n_epoch: int):
    for param, value in optimizer_hyperparams.items():
        _log_or_assign_metric(
            run,
            n_epoch,
            f'{prefix}/{param}',
            value
        )
