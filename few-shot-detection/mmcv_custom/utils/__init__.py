# Copyright (c) Open-MMLab. All rights reserved.
from .registry import Registry, build_from_cfg
from .misc import (check_prerequisites, concat_list, deprecated_api_warning,
                   has_method, import_modules_from_strings, is_list_of,
                   is_method_overridden, is_seq_of, is_str, is_tuple_of,
                   iter_cast, list_cast, requires_executable, requires_package,
                   slice_list, to_1tuple, to_2tuple, to_3tuple, to_4tuple,
                   to_ntuple, tuple_cast)
from .parrots_wrapper import (_BatchNorm,_InstanceNorm, TORCH_VERSION)
from .device_type import (IS_IPU_AVAILABLE, IS_MLU_AVAILABLE,
                          IS_MPS_AVAILABLE, IS_NPU_AVAILABLE)
from .version_utils import digit_version, get_git_hash

__all__ = [
    'check_prerequisites', 'concat_list', 'deprecated_api_warning',
    'has_method', 'import_modules_from_strings', 'is_list_of',
    'is_method_overridden', 'is_seq_of', 'is_str', 'is_tuple_of',
    'iter_cast', 'list_cast', 'requires_executable', 'requires_package',
    'slice_list', 'to_1tuple', 'to_2tuple', 'to_3tuple', 'to_4tuple',
    'to_ntuple', 'tuple_cast', 'Registry', 'build_from_cfg', '_BatchNorm', 
    '_InstanceNorm', 'IS_MLU_AVAILABLE', 'IS_IPU_AVAILABLE','IS_MPS_AVAILABLE', 
    'IS_NPU_AVAILABLE', 'TORCH_VERSION', 'digit_version', 'get_git_hash'
]
