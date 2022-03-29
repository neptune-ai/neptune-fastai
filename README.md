# Neptune + Fastai Integration

Experiment tracking, model registry, data versioning, and live model monitoring for fastai trained models.

## What will you get with this integration? 

* Log, display, organize, and compare ML experiments in a single place
* Version, store, manage, and query trained models, and model building metadata
* Record and monitor model training, evaluation, or production runs live

## What will be logged to Neptune?

* Hyper-parameters
* Losses & metrics
* Training code (Python scripts or Jupyter notebooks) and git information
* Dataset version
* Model Configuration, architecture, and weights
* [other metadata](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)

![image](https://user-images.githubusercontent.com/97611089/160639808-bd381089-66c8-4ed5-a895-0c018b378e0a.png)
*Example dashboard with train-valid metrics and selected parameters*


## Resources

* [Documentation](https://docs.neptune.ai/integrations-and-supported-tools/model-training/fastai)
* [Code example on GitHub](https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/fastai/scripts)
* [Example dashboard in the Neptune app](https://app.neptune.ai/o/common/org/fastai-integration/e/FAS-61/dashboard/fastai-dashboard-1f456716-f509-4432-b8b3-a7f5242703b6)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/fastai/notebooks/Neptune_fastai.ipynb)

## Minimal example

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

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting-started/getting-help#frequently-asked-questions)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
