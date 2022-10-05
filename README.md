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

## Example

```python
# On the command line:
pip install fastai neptune-client[fastai]
```
```python
# In Python:
import neptune.new as neptune

# Start a run
run = neptune.init(project="common/fastai-integration",
                   api_token="ANONYMOUS",
                   source_files=["*.py"])

# Log a single training phase
learn = learner(...)
learn.fit(..., cbs = NeptuneCallback(run=run))

# Log all training phases of the learner
learn = cnn_learner(..., cbs=NeptuneCallback(run=run))
learn.fit(...)
learn.fit(...)

# Stop the run
run.stop()
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting-started/getting-help#frequently-asked-questions)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
