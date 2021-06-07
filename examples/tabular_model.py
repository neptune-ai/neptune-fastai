from neptune import new as neptune
from neptune_fastai.impl import NeptuneCallback

from fastai.basics import *
from fastai.tabular.all import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.callback.mixup import *
from fastcore.script import *


def main():
    neptune_run = neptune.init()

    path = untar_data(URLs.ADULT_SAMPLE)

    dls = TabularDataLoaders.from_csv(path / 'adult.csv',
                                      path=path,
                                      y_names="salary",
                                      cat_names=[
                                          'workclass',
                                          'education',
                                          'marital-status',
                                          'occupation',
                                          'relationship',
                                          'race'
                                      ],
                                      cont_names=['age', 'fnlwgt', 'education-num'],
                                      procs=[Categorify, FillMissing, Normalize])

    learn = tabular_learner(dls,
                            metrics=accuracy,
                            cbs=[
                                NeptuneCallback(run=neptune_run),
                                SaveModelCallback()
                            ])
    learn.fit_one_cycle(10)


if __name__ == '__main__':
    main()