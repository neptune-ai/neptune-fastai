## [UNRELEASED] neptune-fastai 0.10.2

## Fixes
- Skip vocab for models that don't use it. ([#28](https://github.com/neptune-ai/neptune-fastai/pull/28))
- Attribute error when loss is not BaseLoss. ([#29](https://github.com/neptune-ai/neptune-fastai/pull/29))

## neptune-fastai 0.10.1

### Features
- Mechanism to prevent using legacy Experiments in new-API integrations ([#17](https://github.com/neptune-ai/neptune-fastai/pull/17))

## neptune-fastai 0.10.0

### Breaking changes
- Behavior of uploading models and fastai minimal version requirement set to 2.4 ([#16](https://github.com/neptune-ai/neptune-fastai/pull/16))

## neptune-fastai 0.9.6

### Fixes
- Warning instead of an error when calling callback from method without SaveModelCallback ([#15](https://github.com/neptune-ai/neptune-fastai/pull/15))
