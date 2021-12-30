#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pytest import fail

from fastai.basics import accuracy
from fastai.tabular.all import tabular_learner
from fastai.callback.tracker import SaveModelCallback

from neptune_fastai.impl import NeptuneCallback

try:
    # neptune-client=0.9.0 package structure
    from neptune.new.attributes.atoms.float import Float
    from neptune.new.attributes.series.float_series import FloatSeries
except ImportError:
    # neptune-client=1.0.0 package structure
    from neptune.attributes.atoms.float import Float
    from neptune.attributes.series.float_series import FloatSeries


class TestBase:
    def test_basename(self, run, dataset):
        neptune_callback = NeptuneCallback(run=run,
                                           base_namespace='experiment',
                                           upload_saved_models=None)

        learn = tabular_learner(dataset, metrics=accuracy, cbs=[neptune_callback])
        learn.fit_one_cycle(1)

        run.sync()

        structure = run.get_structure()

        assert 'experiment' in structure
        assert 'config' in structure['experiment']
        assert 'io_files' in structure['experiment']
        assert 'metrics' in structure['experiment']

        assert 'fit_0' in structure['experiment']['metrics']
        assert len(structure['experiment']['metrics']) == 1

    def test_basename_fit_callback(self, run, dataset):
        neptune_callback = NeptuneCallback(run=run,
                                           base_namespace='experiment',
                                           upload_saved_models=None)

        learn = tabular_learner(dataset, layers=[10, 10], metrics=accuracy)
        learn.fit_one_cycle(1, cbs=[neptune_callback])

        run.sync()

        structure = run.get_structure()

        assert 'experiment' in structure
        assert 'config' in structure['experiment']
        assert 'io_files' in structure['experiment']
        assert 'metrics' in structure['experiment']

        assert 'fit_0' in structure['experiment']['metrics']
        assert len(structure['experiment']['metrics']) == 1

    def test_multiple_fits(self, run, dataset):
        neptune_callback = NeptuneCallback(run=run,
                                           base_namespace='experiment',
                                           upload_saved_models=None)

        learn = tabular_learner(dataset, metrics=accuracy, layers=[10, 10], cbs=[neptune_callback])
        learn.fit_one_cycle(1)
        learn.fit_one_cycle(1)

        run.sync()

        structure = run.get_structure()

        assert 'experiment' in structure
        assert 'config' in structure['experiment']
        assert 'io_files' in structure['experiment']
        assert 'metrics' in structure['experiment']

        assert 'fit_0' in structure['experiment']['metrics']
        assert 'fit_1' in structure['experiment']['metrics']
        assert len(structure['experiment']['metrics']) == 2

    def test_frozen_fits(self, run, dataset):
        neptune_callback = NeptuneCallback(run=run,
                                           upload_saved_models=None)

        learn = tabular_learner(dataset, metrics=accuracy, layers=[10, 10], cbs=[neptune_callback])
        learn.fit_one_cycle(1)
        learn.opt.freeze_to(1)
        learn.fit_one_cycle(1)
        learn.unfreeze()

        run.sync()

        structure = run.get_structure()

        assert 'config' in structure
        assert 'io_files' in structure
        assert 'metrics' in structure

        assert 'fit_0' in structure['metrics']
        assert 'fit_1' in structure['metrics']
        assert len(structure['metrics']) == 2

        assert 'frozen_level' not in structure['metrics']['fit_0']
        assert 'frozen_level' in structure['metrics']['fit_1']

    def test_optimizer_hyperparams(self, run, dataset):
        neptune_callback = NeptuneCallback(run=run, upload_saved_models=None)

        learn = tabular_learner(dataset, metrics=accuracy, layers=[10, 10], cbs=[neptune_callback])
        learn.fit_one_cycle(1)
        learn.fit_one_cycle(2)

        run.sync()

        structure = run.get_structure()

        assert 'config' in structure
        assert 'io_files' in structure
        assert 'metrics' in structure

        assert 'fit_0' in structure['metrics']
        assert 'fit_1' in structure['metrics']
        assert len(structure['metrics']) == 2

        assert isinstance(structure['metrics']['fit_0']['optimizer_hyperparameters']['eps'], Float)
        assert isinstance(structure['metrics']['fit_1']['optimizer_hyperparameters']['eps'], FloatSeries)

    def test_saving_from_constructor(self, run, dataset):
        learn = tabular_learner(dataset, metrics=accuracy, layers=[10, 10],
                                cbs=[SaveModelCallback(), NeptuneCallback(run=run)])
        learn.fit_one_cycle(1)

        learn = tabular_learner(dataset, metrics=accuracy, layers=[10, 10],
                                cbs=[SaveModelCallback(every_epoch=2), NeptuneCallback(run=run)])
        learn.fit_one_cycle(2)

        run.sync()

        structure = run.get_structure()

        assert 'config' in structure
        assert 'io_files' in structure
        assert 'metrics' in structure

        assert 'fit_0' in structure['metrics']
        assert 'fit_1' in structure['metrics']
        assert len(structure['metrics']) == 2

        assert 'artifacts' in structure['io_files']
        assert 'model_checkpoints' in structure['io_files']['artifacts']
        assert 'fit_0' in structure['io_files']['artifacts']['model_checkpoints']
        assert 'fit_1' in structure['io_files']['artifacts']['model_checkpoints']
        assert len(structure['io_files']['artifacts']['model_checkpoints']) == 2

        assert 'model' in structure['io_files']['artifacts']['model_checkpoints']['fit_0']
        assert len(structure['io_files']['artifacts']['model_checkpoints']['fit_0']) == 1

        assert 'epoch_0' in structure['io_files']['artifacts']['model_checkpoints']['fit_1']
        assert len(structure['io_files']['artifacts']['model_checkpoints']['fit_1']) == 1

    def test_saving_from_method(self, run, dataset):
        learn = tabular_learner(dataset, metrics=accuracy, layers=[10, 10])
        learn.fit_one_cycle(1, cbs=[SaveModelCallback(), NeptuneCallback(run=run)])

        learn = tabular_learner(dataset, metrics=accuracy, layers=[10, 10])
        learn.fit_one_cycle(2, cbs=[SaveModelCallback(every_epoch=2), NeptuneCallback(run=run)])

        run.sync()

        structure = run.get_structure()

        assert 'config' in structure
        assert 'io_files' in structure
        assert 'metrics' in structure

        assert 'fit_0' in structure['metrics']
        assert 'fit_1' in structure['metrics']
        assert len(structure['metrics']) == 2

        assert 'artifacts' in structure['io_files']
        assert 'model_checkpoints' in structure['io_files']['artifacts']
        assert 'fit_0' in structure['io_files']['artifacts']['model_checkpoints']
        assert 'fit_1' in structure['io_files']['artifacts']['model_checkpoints']
        assert len(structure['io_files']['artifacts']['model_checkpoints']) == 2

        assert 'model' in structure['io_files']['artifacts']['model_checkpoints']['fit_0']
        assert len(structure['io_files']['artifacts']['model_checkpoints']['fit_0']) == 1

        assert 'epoch_0' in structure['io_files']['artifacts']['model_checkpoints']['fit_1']
        assert len(structure['io_files']['artifacts']['model_checkpoints']['fit_1']) == 1

    def test_without_save_model_constr(self, run, dataset):
        try:
            learn = tabular_learner(dataset,
                                    metrics=accuracy,
                                    layers=[10, 10],
                                    cbs=[NeptuneCallback(run=run), SaveModelCallback()])
            learn.fit_one_cycle(1)
        except AttributeError as exception:
            fail(exception)

    def test_without_save_model_method(self, run, dataset):
        try:
            learn = tabular_learner(dataset, metrics=accuracy, layers=[10, 10])
            learn.fit_one_cycle(1, cbs=[NeptuneCallback(run=run), SaveModelCallback()])
        except AttributeError as exception:
            fail(exception)
