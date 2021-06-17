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
from itertools import islice

from fastai.basics import untar_data, URLs, error_rate
from fastai.vision.all import ImageDataLoaders, get_image_files, Resize, cnn_learner, resnet34

from neptune import new as neptune
from neptune_fastai.impl import NeptuneCallback


def is_cat(x):
    return x[0].isupper()


def main():
    neptune_run = neptune.init()

    path = untar_data(URLs.PETS) / 'images'

    dls = ImageDataLoaders.from_name_func(path,
                                          list(islice(get_image_files(path), 256)),
                                          valid_pct=0.2,
                                          seed=42,
                                          label_func=is_cat,
                                          item_tfms=Resize(224)
                                          )

    learn = cnn_learner(dls,
                        resnet34,
                        metrics=error_rate,
                        cbs=[NeptuneCallback(neptune_run, 'experiment')],
                        pretrained=False)

    learn.fit_one_cycle(1)
    learn.fine_tune(2)
    learn.fit(2)


if __name__ == '__main__':
    main()
