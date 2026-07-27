import numpy as np
import nibabel as nib


def masked2d_to_unmasked3d(data, mask):
    '''
    mask2d data: e.g. output of "betas, ResMS, info = spm.get_betas(maskFile)" function
    mask: nib.load(maskFile)
    1) get total amount of voxels
    2) turn mask into boolean
    3) turn boolean 3D mask into boolean 11D array
    4) get indices, where True
    5) create new variable with amount of voxels in 1D array
    6) index new variable where boolean 1D mask is True, set those values = data
    '''
    n_voxels_total = np.prod(mask.shape)
    mask_bool = mask.get_fdata() > 0
    mask_bool_1D = mask_bool.flatten()
    mask_bool_1D_idx = np.where(mask_bool_1D)[0]
    data_3D = np.zeros(n_voxels_total)
    data_3D[mask_bool_1D_idx] = data
    data_3D = data_3D.reshape(mask.shape)
    data_img = nib.Nifti1Image(data_3D, affine = mask.affine)

    return data_3D, data_img

def ceck_zeros_and_nans(data):

    zero_indices = np.argwhere(data == 0)
    if zero_indices.size > 0:
        print(f"! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! Zero found {zero_indices.shape} indices:")
    # else:
    #     print("    No zeros found")


    data_nan = np.zeros(data.shape)
    data_nan[np.isnan(data)] = 1
    nan_found = np.sum(data_nan)
    if nan_found > 0:
        print(f'! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! found {nan_found} nan values in the variable')
    # else: print('    No nan values found')