## neptune-fastai 1.1.1

### Fixes
- Don't error if `optim.__name__` is not present. (https://github.com/neptune-ai/neptune-fastai/pull/54)

## neptune-fastai 1.1.0

### Changes
- Updated integration for compatibility with `neptune 1.X`
- Removed `neptune` and `neptune-client` from base requirements - installation is checked at runtime
- Functions outside the callback accept `Handler` as well

## neptune-fastai 1.0.0

  ### Changes
  - `NeptuneCallback` now accepts a namespace `Handler` as an alternative to `Run` for the `run` argument. This means that
    you can call it like `NeptuneCallback(run=run["some/namespace/"])` to log everything to the `some/namespace/`
    location of the run.

  ### Breaking changes
  - Instead of the `log()` method, the integration now uses `append()` which is available since version 0.16.14
    of neptune-client.

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
