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

__all__ = ["NeptuneCallback", "retrieve_fit_index"]

import hashlib
import time
import warnings
from typing import Dict, List, Optional

from fastai.basics import Callback, join_path_file, store_attr
from fastai.callback.hook import total_params
from fastai.callback.tracker import SaveModelCallback
from fastai.learner import Learner
from fastai.torch_core import default_device, trainable_params
from neptune_fastai import __version__

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
    from neptune.new.internal.utils import verify_type
    from neptune.new.internal.utils.compatibility import expect_not_an_experiment
    from neptune.new.types import File
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type
    from neptune.internal.utils.compatibility import expect_not_an_experiment
    from neptune.types import File


INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-fastai"


class NeptuneCallback(Callback):
    # pylint: disable=trailing-whitespace
    """Neptune callback for logging metadata during fastai training loop

    See guide with example in the `Neptune-fastai docs`_.

    This callback logs paramaters, metrics, losses, model configuration,
    optimizer configuration and info about the dataset:
    path, number of samples and hash value.

    Metrics and losses are logged separately for every ``learner.fit()`` call.
    For examples when you call ``learner.fit(n)`` the first time it will create
    a folder named fit_0 under the folder metrics that contains optimizer hyperparameters,
    batch and loader-level metrics.
    metrics
        |--> fit_0
            |--> batch
            |--> loader
            |--> optimizer hyperparameters
        |--> ...
        |--> fit_n

    Note:
        You can use public ``api_token="ANONYMOUS"`` and set ``project="common/fastai-integration"``
        for testing without registration.

    Args:
        run (:obj: `neptune.new.run.Run`): Neptune run object
            A run in Neptune is a representation of all metadata that you log to Neptune.
            Learn more in `run docs`_.
        base_namespace (:obj: `str`, optional): Root namespace. All metdata will be logged inside.
        Defaults is empty string. In this case metadata is logged without common "base_namespace"
        upload_saved_models (:obj: `str`, optional):  `'all'` or `'last'`.
        When using `'all'`, it uploads all model checkpoints created by `SaveModelCallback()`.
        When using `'last'`, it uploads the last model checkpoint created by `SaveModelCallback()`.
        Defaults to `'all'`.

    Examples:
        For more examples visit `example scripts`_.

        Full script that does model training and logging of the metadata.
            import fastai
            from neptune.new.integrations.fastai import NeptuneCallback
            from fastai.vision.all import *
            import neptune.new as neptune
            from neptune.new.types import File

            run = neptune.init(
                project="common/fastai-integration", api_token="ANONYMOUS", tags="more options"
            )

            path = untar_data(URLs.MNIST_TINY)
            dls = ImageDataLoaders.from_csv(path)

            # Single & Multi phase logging

            # 1. Log a single training phase
            learn = cnn_learner(dls, resnet18, metrics=accuracy)
            learn.fit_one_cycle(1, cbs=[NeptuneCallback(run=run, base_namespace="experiment_1")])
            learn.fit_one_cycle(2)

            # 2. Log all training phases of the learner
            learn = cnn_learner(dls, resnet18, cbs=[NeptuneCallback(run=run, base_namespace="experiment_2")])
            learn.fit_one_cycle(1)
            learn.fit_one_cycle(2)

            # Log model weights

            # Add SaveModelCallback()

            # 1. Log Every N epochs
            n = 2
            learn = cnn_learner(
                dls, resnet18, metrics=accuracy,
                cbs=[SaveModelCallback(every_epoch=n),
                    NeptuneCallback(run=run, base_namespace='experiment_3', upload_saved_models='all')])

            learn.fit_one_cycle(5)

            # 2. Best Model
            learn = cnn_learner(
                dls, resnet18, metrics=accuracy,
                cbs=[SaveModelCallback(), NeptuneCallback(run=run, base_namespace='experiment_4')])

            learn.fit_one_cycle(5)

            # Log images
            batch = dls.one_batch()
            for i, (x, y) in enumerate(dls.decode_batch(batch)):
                # Neptune supports torch tensors
                # fastai uses their own tensor type name TensorImage
                # so you have to convert it back to torch.Tensor
                run["images/one_batch"].log(
                    File.as_image(x.as_subclass(torch.Tensor).permute(2, 1, 0) / 255.0),
                    name=f"{i}",
                    description=f"Label: {y}",
                )

            # Stop Run
            run.stop()

    .. _Neptune-fastai docs:
        https://docs.neptune.ai/integrations-and-supported-tools/model-training/fastai
       _run docs:
        https://docs.neptune.ai/api-reference/run
       _example scripts:
        https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/fastai/scripts

    """
    order = SaveModelCallback.order + 1

    def __init__(
        self,
        run: neptune.Run,
        base_namespace: str = "",
        upload_saved_models: Optional[str] = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)

        expect_not_an_experiment(run)
        verify_type("run", run, neptune.Run)
        verify_type("base_namespace", base_namespace, str)
        verify_type("upload_saved_models", upload_saved_models, (str, type(None)))

        assert upload_saved_models is None or upload_saved_models in ("all", "last")

        self.neptune_run = run
        self.fit_index = retrieve_fit_index(run, f"{base_namespace}/metrics/")

        run[INTEGRATION_VERSION_KEY] = __version__

        store_attr("base_namespace,upload_saved_models")

    @property
    def name(self) -> str:
        return "neptune"

    @property
    def _batch_size(self) -> int:
        return self.dls.bs

    @property
    def _optimizer_name(self) -> Optional[str]:
        return self.opt_func.__name__

    @property
    def _device(self) -> str:
        return default_device() or getattr(self.dls, "device", default_device())

    @property
    def _vocab(self) -> List[str]:
        return self.dls.vocab

    @property
    def _total_model_parameters(self) -> int:
        params, _ = total_params(self.learn)
        return params

    @property
    def _trainable_model_parameters(self) -> int:
        return sum(p.numel() for p in trainable_params(self.learn))

    @property
    def _optimizer_criterion(self) -> str:

        return repr(
            self.learn.loss_func.func
            if hasattr(self.learn.loss_func, "func")
            else self.learn.loss_func
        )

    @property
    def _optimizer_hyperparams(self) -> Optional[dict]:
        if hasattr(self, "opt") and hasattr(self.opt, "hypers"):
            if len(self.opt.hypers) == 1:
                return dict(self.learn.opt.hypers[0])

            return {
                f"group_layer_{layer}/{hyper}": value
                for layer, opts in enumerate(self.learn.opt.hypers)
                for hyper, value in opts.items()
            }

    @property
    def _frozen_level(self) -> int:
        return (
            self.opt.frozen_idx
            if hasattr(self, "opt") and hasattr(self.opt, "frozen_idx")
            else 0
        )

    @property
    def _input_shape(self) -> Dict:
        if hasattr(self, "x") and hasattr(self, "y"):
            return {
                "x": str(list(self.x[0].shape)),
                "y": 1 if len(self.y[0].shape) == 0 else str(list(self.y[0].shape)),
            }

    @property
    def _target(self) -> str:
        return "training" if self.learn.training else "validation"

    def _log_model_configuration(self):
        config = {
            "device": self._device,
            "batch_size": self._batch_size,
            "model": {
                "params": {
                    "total": self._total_model_parameters,
                    "trainable_params": self._trainable_model_parameters,
                    "non_trainable_params": self._total_model_parameters
                    - self._trainable_model_parameters,
                },
            },
            "criterion": self._optimizer_criterion,
            "optimizer": {
                "name": self._optimizer_name,
                "initial_hyperparameters": self._optimizer_hyperparams,
            },
        }

        if hasattr(self.learn.dls, "vocab"):
            config["model"]["vocab"] = {
                "details": self._vocab,
                "total": len(self._vocab),
            }

        self.neptune_run[f"{self.base_namespace}/config"] = config

    def after_create(self):
        if not hasattr(self, "save_model") and self.upload_saved_models:
            warnings.warn(
                "NeptuneCallback: SaveModelCallback is necessary for uploading model checkpoints."
            )

    def before_fit(self):
        self._log_model_configuration()

        _log_model_architecture(self.neptune_run, self.base_namespace, self.learn)
        _log_dataset_metadata(self.neptune_run, self.base_namespace, self.learn)

        prefix = f"{self.base_namespace}/metrics/fit_{self.fit_index}"

        self.neptune_run[f"{prefix}/n_epoch"] = self.n_epoch

        if self._frozen_level > 0:
            self.neptune_run[f"{prefix}/frozen_level"] = self._frozen_level

    def before_batch(self):
        if self.learn.iter == 0:
            self.neptune_run[
                f"{self.base_namespace}/metrics/fit_{self.fit_index}/{self._target}/batch_counter"
            ] = self.n_iter

        if self.learn.train_iter == 1:
            self.neptune_run[
                f"{self.base_namespace}/config/input_shape"
            ] = self._input_shape

    def after_batch(self):
        prefix = (
            f"{self.base_namespace}/metrics/fit_{self.fit_index}/{self._target}/batch"
        )

        self.neptune_run[f"{prefix}/loss"].log(value=self.learn.loss.clone())

        if hasattr(self, "smooth_loss"):
            self.neptune_run[f"{prefix}/smooth_loss"].log(
                value=self.learn.smooth_loss.clone()
            )

    def after_train(self):
        prefix = f"{self.base_namespace}/metrics/fit_{self.fit_index}/training/loader"

        for metric_name, metric_value in zip(
            self.learn.recorder.metric_names, self.learn.recorder.log
        ):
            if metric_name not in {"epoch", "time"}:
                self.neptune_run[f"{prefix}/{metric_name}"].log(metric_value)

        self.neptune_run[f"{prefix}/duration"].log(
            value=time.time() - self.learn.recorder.start_epoch
        )

        _log_optimizer_hyperparams(
            self.neptune_run,
            f"{self.base_namespace}/metrics/fit_{self.fit_index}/optimizer_hyperparameters",
            self._optimizer_hyperparams,
            self.n_epoch,
        )

    def after_validate(self):
        prefix = f"{self.base_namespace}/metrics/fit_{self.fit_index}/validation/loader"

        for metric_name, metric_value in zip(
            self.learn.recorder.metric_names, self.learn.recorder.log
        ):
            if metric_name not in {"epoch", "time", "train_loss"}:
                self.neptune_run[f"{prefix}/{metric_name}"].log(metric_value)

        self.neptune_run[f"{prefix}/duration"].log(
            value=time.time() - self.learn.recorder.start_epoch
        )

    def after_epoch(self):
        if (
            self.upload_saved_models == "all"
            and hasattr(self, "save_model")
            and hasattr(self.save_model, "every_epoch")
            and self.save_model.every_epoch
            and self.epoch % self.save_model.every_epoch == 0
        ):
            filename = f"{self.learn.save_model.fname}_{self.learn.save_model.epoch}"
            path = join_path_file(
                filename, self.learn.path / self.learn.model_dir, ext=".pth"
            )
            prefix = (
                f"{self.base_namespace}/io_files/artifacts/model_checkpoints/fit_{self.fit_index}/"
                f"epoch_{self.learn.save_model.epoch}"
            )
            self.neptune_run[prefix].upload(str(path))

    def after_fit(self):
        if (
            self.upload_saved_models
            and hasattr(self, "save_model")
            and hasattr(self.save_model, "every_epoch")
            and not self.save_model.every_epoch
        ):
            filename = self.learn.save_model.fname
            path = join_path_file(
                filename, self.learn.path / self.learn.model_dir, ext=".pth"
            )
            prefix = f"{self.base_namespace}/io_files/artifacts/model_checkpoints/fit_{self.fit_index}/{filename}"

            self.neptune_run[prefix].upload(str(path))

        self.fit_index += 1


def _log_model_architecture(run: neptune.Run, base_namespace: str, learn: Learner):
    if hasattr(learn, "arch"):
        run[f"{base_namespace}/config/model/architecture_name"] = getattr(
            learn.arch, "__name__", ""
        )

    model_architecture = File.from_content(repr(learn.model))

    run[f"{base_namespace}/config/model/architecture"].upload(model_architecture)
    run[f"{base_namespace}/io_files/artifacts/model_architecture"].upload(
        model_architecture
    )


def _log_dataset_metadata(run: neptune.Run, base_namespace: str, learn: Learner):
    sha = hashlib.sha1(str(learn.dls.path).encode())

    run[f"{base_namespace}/io_files/resources/dataset"] = {
        "path": learn.dls.path,
        "size": learn.dls.n,
        "sha": sha.hexdigest(),
    }


def _log_or_assign_metric(run: neptune.Run, number_of_epochs: int, metric: str, value):
    if number_of_epochs > 1:
        run[metric].log(value)
    else:
        run[metric] = value


def retrieve_fit_index(run: neptune.Run, path: str) -> int:
    return len(run.get_attribute(path) or [])


def _log_optimizer_hyperparams(
    run: neptune.Run, prefix: str, optimizer_hyperparams: dict, n_epoch: int
):
    for param, value in optimizer_hyperparams.items():
        _log_or_assign_metric(run, n_epoch, f"{prefix}/{param}", value)
