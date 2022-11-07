## neptune-fastai 0.10.4

### Changes
- moved neptune_fastai package to src dir ([#37](https://github.com/neptune-ai/neptune-fastai/pull/37))
- Poetry as a package builder ([#44](https://github.com/neptune-ai/neptune-fastai/pull/44))

### Fixes
- Fixed NeptuneCallback import error - now possible to directly import with `from neptune_fastai import NeptuneCallback`
  ([#39](https://github.com/neptune-ai/neptune-fastai/pull/39))

## neptune-fastai 0.10.3

### Changes
- Changed integrations utils to be imported from non-internal package ([#33](https://github.com/neptune-ai/neptune-fastai/pull/33))

## neptune-fastai 0.10.2

### Fixes
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
