from neptune import new as neptune
from neptune_fastai.impl import NeptuneCallback

from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.callback.mixup import *
from fastcore.script import *

from itertools import islice


def is_cat(x):
    return x[0].isupper()


def main():
    neptune_run = neptune.init()

    path = untar_data(URLs.PETS) / 'images'

    dls = ImageDataLoaders.from_name_func(path,
                                          list(islice(get_image_files(path), 128)),
                                          valid_pct=0.2,
                                          seed=42,
                                          label_func=is_cat,
                                          item_tfms=Resize(224)
                                          )

    learn = cnn_learner(dls,
                        resnet34,
                        metrics=error_rate,
                        cbs=[
                            NeptuneCallback(neptune_run, save_best_model=False),
                            SaveModelCallback()
                        ])
    learn.fit_one_cycle(3)


if __name__ == '__main__':
    main()
