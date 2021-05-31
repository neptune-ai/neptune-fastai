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

__all__ = [
    # TODO: add importable public names here, `neptune-client` uses `import *`
    # https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
]

# TODO: use `warnings.warn` for user caused problems: https://stackoverflow.com/a/14762106/1565454
import warnings

try:
    # neptune-client=0.9.0 package structure
    import neptune.new as neptune
    from neptune.new.internal.utils import verify_type
except ImportError:
    # neptune-client=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type

from neptune_integration_template import __version__  # TODO: change module name

INTEGRATION_VERSION_KEY = 'source_code/integrations/integration-template'  # TODO: change path

# TODO: Implementation of neptune-integration here
