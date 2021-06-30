# Neptune - fast.ai integration

See [the official docs](https://docs.neptune.ai/integrations-and-supported-tools/model-training/fastai).

```python
from fastai.basics import URLs, untar_data, accuracy
from fastai.tabular.all import tabular_learner, TabularDataLoaders, Categorify, FillMissing, Normalize
from fastai.callback.all import SaveModelCallback

from neptune import new as neptune
from neptune_fastai.impl import NeptuneCallback


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
                        metrics=accuracy)
learn.fit_one_cycle(10,
                    cbs=[
                            NeptuneCallback(run=neptune_run,
                                            base_namespace='experiment'),
                            SaveModelCallback(monitor='accuracy')
                        ])
```