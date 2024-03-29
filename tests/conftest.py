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
import neptune
import pytest
from fastai.basics import (
    URLs,
    untar_data,
)
from fastai.tabular.all import (
    Categorify,
    FillMissing,
    Normalize,
    TabularDataLoaders,
)


@pytest.fixture()
def run():
    exp = neptune.init_run()
    yield exp
    exp.stop()


@pytest.fixture()
def dataset():
    path = untar_data(URLs.ADULT_SAMPLE)

    yield TabularDataLoaders.from_csv(
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
    )
