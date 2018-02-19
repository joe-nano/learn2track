import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table

from learn2track import neurotools, datasets


def make_dummy_dataset(volume_manager, nb_subjects=3, seed=1234):
    rng = np.random.RandomState(seed)
    nb_bundles = 7
    nb_gradients = 64

    subjects = []
    for subject_id in range(nb_subjects):

        dwi, gradients = make_dummy_dwi(nb_gradients, seed=seed)

        volume = neurotools.resample_dwi(dwi, gradients.bvals, gradients.bvecs).astype(np.float32)
        volume_shape = np.array(dwi.shape)

        tracto_data = neurotools.TractographyData(dwi, gradients)

        for bundle_id in range(nb_bundles):
            streamlines = [rng.randn(rng.randint(5, 100), 3) * 5 + volume_shape[:3]/2. for i in range(rng.randint(5, 30))]
            tracto_data.add(streamlines, "bundle_{}".format(bundle_id))

        subject_id = volume_manager.register(volume)
        tracto_data.subject_id = subject_id
        subjects.append(tracto_data)

    return datasets.TractographyDataset(subjects, name="test", keep_on_cpu=True)


def make_dummy_dwi(nb_gradients, volume_shape=None, seed=1234):
    rng = np.random.RandomState(seed)
    if volume_shape is None:
        volume_shape = np.array((rng.randint(5, 30), rng.randint(5, 30), rng.randint(5, 30), nb_gradients))
    else:
        volume_shape += (nb_gradients,)

    dwi = nib.Nifti1Image(rng.rand(*volume_shape), affine=np.eye(4))
    bvals = [0] + [1000] * (nb_gradients - 1)
    bvecs = rng.randn(nb_gradients, 3)
    bvecs /= np.sqrt(np.sum(bvecs ** 2, axis=1, keepdims=True))
    gradients = gradient_table(bvals, bvecs)
    return dwi, gradients
