"""Tools for model-based motion correction

Some more text here.
"""
import os.path as op
import numpy as np
import ipywidgets as wdg
import IPython.display as display
from IPython.display import Image
import matplotlib.pyplot as plt

import nibabel as nib
import dipy.core.gradients as dpg
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)

from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

syn_metric_dict = {'CC': CCMetric,
                   'EM': EMMetric,
                   'SSD': SSDMetric}


def syn_registration(moving, static,
                     moving_grid2world=None,
                     static_grid2world=None,
                     step_length=0.25,
                     metric='CC',
                     dim=3,
                     level_iters=[10, 10, 5],
                     sigma_diff=2.0,
                     prealign=None):
    """Register a source image (moving) to a target image (static).

    Parameters
    ----------
    moving : ndarray
        The source image data to be registered
    moving_grid2world : array, shape (4,4)
        The affine matrix associated with the moving (source) data.
    static : ndarray
        The target image data for registration
    static_grid2world : array, shape (4,4)
        The affine matrix associated with the static (target) data
    metric : string, optional
        The metric to be optimized. One of `CC`, `EM`, `SSD`,
        Default: CCMetric.
    dim: int (either 2 or 3), optional
       The dimensions of the image domain. Default: 3
    level_iters : list of int, optional
        the number of iterations at each level of the Gaussian Pyramid (the
        length of the list defines the number of pyramid levels to be
        used).

    Returns
    -------
    warped_moving : ndarray
        The data in `moving`, warped towards the `static` data.
    forward : ndarray (..., 3)
        The vector field describing the forward warping from the source to the
        target.
    backward : ndarray (..., 3)
        The vector field describing the backward warping from the target to the
        source.
    """
    use_metric = syn_metric_dict[metric](dim, sigma_diff=sigma_diff)

    sdr = SymmetricDiffeomorphicRegistration(use_metric, level_iters,
                                             step_length=step_length)
    mapping = sdr.optimize(static, moving,
                           static_grid2world=static_grid2world,
                           moving_grid2world=moving_grid2world,
                           prealign=prealign)

    warped_moving = mapping.transform(moving)
    return warped_moving, mapping


def resample(moving, static, moving_grid2world, static_grid2world):
    """Resample an image from one space to another."""
    identity = np.eye(4)
    affine_map = AffineMap(identity,
                           static.shape, static_grid2world,
                           moving.shape, moving_grid2world)
    resampled = affine_map.transform(moving)

# Affine registration pipeline:
affine_metric_dict = {'MI': MutualInformationMetric}


def c_of_mass(moving, static, static_grid2world, moving_grid2world,
              reg, starting_affine, params0=None):
    transform = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)
    transformed = transform.transform(moving)
    return transformed, transform.affine


def translation(moving, static, static_grid2world, moving_grid2world,
                reg, starting_affine, params0=None):
    transform = TranslationTransform3D()
    translation = reg.optimize(static, moving, transform, params0,
                               static_grid2world, moving_grid2world,
                               starting_affine=starting_affine)

    return translation.transform(moving), translation.affine


def rigid(moving, static, static_grid2world, moving_grid2world,
          reg, starting_affine, params0=None):
    transform = RigidTransform3D()
    rigid = reg.optimize(static, moving, transform, params0,
                         static_grid2world, moving_grid2world,
                         starting_affine=starting_affine)
    return rigid.transform(moving), rigid.affine


def affine(moving, static, static_grid2world, moving_grid2world,
           reg, starting_affine, params0=None):
    transform = AffineTransform3D()
    affine = reg.optimize(static, moving, transform, params0,
                          static_grid2world, moving_grid2world,
                          starting_affine=starting_affine)

    return affine.transform(moving), affine.affine


def affine_registration(moving, static,
                        moving_grid2world=None,
                        static_grid2world=None,
                        nbins=32,
                        sampling_prop=None,
                        metric='MI',
                        pipeline=[c_of_mass, translation, rigid, affine],
                        level_iters=[10000, 1000, 100],
                        sigmas=[5.0, 2.5, 0.0],
                        factors=[4, 2, 1],
                        params0=None):

    """
    Find the affine transformation between two 3D images.
    """
    # Define the Affine registration object we'll use with the chosen metric:
    use_metric = affine_metric_dict[metric](nbins, sampling_prop)
    affreg = AffineRegistration(metric=use_metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    # Bootstrap this thing with the identity:
    starting_affine = np.eye(4)
    # Go through the selected transformation:
    for func in pipeline:
        transformed, starting_affine = func(moving, static,
                                            static_grid2world,
                                            moving_grid2world,
                                            affreg, starting_affine,
                                            params0)
    return transformed, starting_affine


def register_series(series, ref, pipeline):
    """ Register a series to a reference image.

    Parameters
    ----------
    series : Nifti1Image object
       The data is 4D with the last dimension separating different 3D volumes
    ref : Nifti1Image or integer or iterable
    """
    if isinstance(ref, nib.Nifti1Image):
        static = ref
        static_data = static.get_data()
        s_g2w = static.get_affine()
        moving = series
        moving_data = moving.get_data()
        m_g2w = moving.get_affine()

    elif isinstance(ref, int) or np.iterable(ref):
        data = series.get_data()
        idxer = np.zeros(data.shape[-1]).astype(bool)
        idxer[ref] = True
        static_data = data[..., idxer]
        if len(static_data.shape) > 3:
            static_data = np.mean(static_data, -1)
        moving_data = data[..., ~idxer]
        m_g2w = s_g2w = series.affine

    affine_list = []
    transformed_list = []
    for ii in range(moving_data.shape[-1]):
        this_moving = moving_data[..., ii]
        transformed, affine = affine_registration(this_moving, static_data,
                                                  moving_grid2world=m_g2w,
                                                  static_grid2world=s_g2w,
                                                  pipeline=pipeline)
        transformed_list.append(transformed)
        affine_list.append(affine)

    return transformed_list, affine_list


def make_widget(data, cmap='bone', dims=4, contours=False):
    """Create an ipython widget for displaying 3D/4D data."""
    def plot_image3d(z=data.shape[-1]//2):
        fig, ax = plt.subplots(1)
        im = ax.imshow(data[:, :, z], cmap=cmap, vmax=np.max(data),
                       vmin=np.min(data))
        if contours:
            cc = measure.find_contours(data[:, :, z], contours)
            for n, c in enumerate(cc):
                ax.plot(c[:, 1], c[:, 0], linewidth=2)
        plt.colorbar(im)
        fig.set_size_inches([10, 10])
        plt.show()

    def plot_image4d(z=data.shape[-2]//2, b=data.shape[-1]//2):
        fig, ax = plt.subplots(1)
        im = ax.imshow(data[:, :, z, b], cmap=cmap, vmax=np.max(data),
                       vmin=np.min(data))
        fig.set_size_inches([10, 10])
        plt.colorbar(im)
        plt.show()

    if dims == 4:
        pb_widget = wdg.interactive(plot_image4d,
                                    z=wdg.IntSlider(min=0,
                                                    max=data.shape[-2]-1,
                                                    value=data.shape[-2]//2),
                                    b=wdg.IntSlider(min=0,
                                                    max=data.shape[-1]-1,
                                                    value=0))
    elif dims == 3:
        if len(data.shape) == 4:
            # RGB images:
            zidx = -2
        else:
            zidx = -1
        zmax = data.shape[zidx] - 1
        zval = data.shape[zidx] // 2

        pb_widget = wdg.interactive(plot_image3d,
                                    z=wdg.IntSlider(min=0,
                                                    max=zmax,
                                                    value=zval))
    display.display(pb_widget)
