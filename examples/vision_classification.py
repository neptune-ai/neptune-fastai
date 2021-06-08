from neptune import new as neptune
from neptune_fastai.impl import NeptuneCallback

from itertools import islice

from fastai.basics import untar_data, URLs, error_rate
from fastai.vision.all import ImageDataLoaders, get_image_files, Resize, cnn_learner, resnet34
from fastai.callback.all import SaveModelCallback


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
    learn.fit_one_cycle(2)


if __name__ == '__main__':
    main()
