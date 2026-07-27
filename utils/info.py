import numpy as np

def get_var_info(spm):
    for attr in dir(spm):
        if not attr.startswith('__'):
            value = getattr(spm, attr)
            if isinstance(value, np.ndarray):
                print(f"{attr}: {type(value)}, dtype={value.dtype}, shape={value.shape}")
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                print(f"{attr}: {type(value)}, contains numpy arrays with dtype={value[0].dtype} and shape={value[0].shape}")
            else:
                print(f"{attr}: {type(value)}")