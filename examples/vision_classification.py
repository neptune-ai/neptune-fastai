from neptune import new as neptune
from neptune_fastai.impl import NeptuneCallback

from itertools import islice

from fastai.basics import untar_data, URLs, error_rate
from fastai.vision.all import ImageDataLoaders, get_image_files, Resize, cnn_learner, resnet34
from fastai.callback.all import SaveModelCallback


def is_cat(x):
    return x[0].isupper()


def main():
    neptune_run = neptune.init(run='FAS-200')

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
                        cbs=[
                            #NeptuneCallback(neptune_run, 'experiment', save_best_model=False),
                            # SaveModelCallback()
                        ],
                        pretrained=False)

    learn.fit_one_cycle(1, cbs=[NeptuneCallback(neptune_run, 'experiment', save_best_model=False)])
    learn.fine_tune(2)
    learn.fit(2, cbs=[NeptuneCallback(neptune_run, 'experiment', save_best_model=False)])


if __name__ == '__main__':
    main()
