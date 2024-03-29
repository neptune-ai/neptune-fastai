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
from functools import partial
from itertools import islice
from pathlib import Path

import torch
from fastai.basics import (
    URLs,
    accuracy,
    error_rate,
    untar_data,
)
from fastai.callback.all import SaveModelCallback
from fastai.optimizer import Adam
from fastai.tabular.all import (
    Categorify,
    FillMissing,
    Normalize,
    TabularDataLoaders,
    tabular_learner,
)
from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    cnn_learner,
    get_image_files,
    squeezenet1_0,
)

try:
    from neptune import init_run
    from neptune.integrations.fastai import NeptuneCallback
except ImportError:
    from neptune.new import init_run
    from neptune.new.integrations.fastai import NeptuneCallback

import neptune_fastai


def is_cat(x):
    return x[0].isupper()


class TestE2E:
    def test_vision_classification_with_handler(self):
        # given (Subject)
        run = init_run(name="Integration fastai (vision classification)")["test"]
        root_obj = run.get_root_object()

        path = untar_data(URLs.PETS) / "images"

        dls = ImageDataLoaders.from_name_func(
            path,
            list(islice(get_image_files(path), 128)),
            valid_pct=0.2,
            seed=42,
            label_func=is_cat,
            item_tfms=Resize(224),
            device=torch.device("cpu"),
        )

        opt_func = partial(Adam, lr=3e-3, wd=0.01)

        learn = cnn_learner(
            dls,
            squeezenet1_0,
            metrics=error_rate,
            cbs=[NeptuneCallback(run, "experiment")],
            pretrained=False,
            opt_func=opt_func,
        )

        learn.fit(1)
        root_obj.sync()

        # then
        # correct integration version is logged
        logged_version = root_obj["source_code/integrations/neptune-fastai"].fetch()
        assert logged_version == neptune_fastai.__version__

        # and
        exp_config = run["experiment/config"].fetch()
        assert exp_config["batch_size"] == 64
        assert exp_config["criterion"] == "CrossEntropyLoss()"
        assert exp_config["optimizer"]["name"] == "N/A"
        assert exp_config["input_shape"] == {"x": "[3, 224, 224]", "y": 1}

        # and
        dateset = run["experiment/io_files/resources/dataset"].fetch()
        assert dateset["path"].endswith(str(Path(".fastai/data/oxford-iiit-pet/images")))
        assert dateset["size"] == 103

        # and
        metrics = run["experiment/metrics"].fetch()
        assert len(metrics.keys()) == 1
        assert metrics["fit_0"]["n_epoch"] == 1

    def test_tabular_model(self):
        # given (Subject)
        run = init_run(name="Integration fastai (tabular model)")

        path = untar_data(URLs.ADULT_SAMPLE)

        dls = TabularDataLoaders.from_csv(
            path / "adult.csv",
            path=path,
            y_names="salary",
            cat_names=[
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
            ],
            cont_names=["age", "fnlwgt", "education-num"],
            procs=[Categorify, FillMissing, Normalize],
            device=torch.device("cpu"),
        )

        learn = tabular_learner(dls, metrics=accuracy)

        # when
        learn.fit(
            2,
            cbs=[
                NeptuneCallback(run=run, base_namespace="experiment"),
                SaveModelCallback(monitor="accuracy"),
            ],
        )
        run.sync()

        # then
        # correct integration version is logged
        logged_version = run["source_code/integrations/neptune-fastai"].fetch()
        assert logged_version == neptune_fastai.__version__

        exp_config = run["experiment/config"].fetch()
        assert exp_config["batch_size"] == 64
        assert exp_config["criterion"] == "CrossEntropyLoss()"
        assert exp_config["input_shape"] == {"x": "[64, 7]", "y": "[1]"}

        # and
        dateset = run["experiment/io_files/resources/dataset"].fetch()
        assert dateset["path"].endswith(str(Path(".fastai/data/adult_sample")))
        assert dateset["size"] == 26049

        # and
        metrics = run["experiment/metrics"].fetch()
        assert len(metrics.keys()) == 1
        assert metrics["fit_0"]["n_epoch"] == 2
