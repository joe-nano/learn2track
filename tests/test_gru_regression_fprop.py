import numpy as np

import dipy
import nibabel as nib
from dipy.core.gradients import gradient_table

import theano

from learn2track.utils import Timer
from learn2track import datasets, batch_schedulers, neurotools, factories


def make_dummy_dataset(nb_subjects=3, seed=1234):
    rng = np.random.RandomState(seed)
    nb_bundles = 7
    nb_gradients = 64

    subjects = []
    for subject_id in range(nb_subjects):

        volume_shape = np.array((rng.randint(5, 30), rng.randint(5, 30), rng.randint(5, 30), nb_gradients))

        dwi = nib.Nifti1Image(rng.rand(*volume_shape), affine=np.eye(4))
        bvals = [0] + [1000] * (nb_gradients-1)
        bvecs = rng.randn(nb_gradients, 3)
        bvecs /= np.sqrt(np.sum(bvecs**2, axis=1, keepdims=True))
        gradients = gradient_table(bvals, bvecs)

        volume = neurotools.resample_dwi(dwi, gradients.bvals, gradients.bvecs).astype(np.float32)
        tracto_data = neurotools.TractographyData(dwi, gradients)

        for bundle_id in range(nb_bundles):
            streamlines = [rng.randn(rng.randint(100), 3) * 5 + volume_shape[:3]/2.
                           for i in range(rng.randint(30))]
            tracto_data.add(streamlines, "bundle_{}".format(bundle_id))

        tracto_data.volume = volume
        subjects.append(tracto_data)

    return datasets.TractographyDataset(subjects, "test", keep_on_cpu=True)





def test_gru_regression_fprop():
    hidden_sizes = 50

    with Timer("Creating dataset", newline=True):
        trainset = make_dummy_dataset()
        print("Dataset sizes:", len(trainset))

        batch_scheduler = batch_schedulers.TractographyBatchScheduler(trainset,
                                                                      batch_size=16,
                                                                      noisy_streamlines_sigma=None,
                                                                      seed=1234)
        print ("An epoch will be composed of {} updates.".format(batch_scheduler.nb_updates_per_epoch))
        print (batch_scheduler.input_size, hidden_sizes, batch_scheduler.target_size)

    with Timer("Creating model"):
        hyperparams = {'model': 'gru_regression',
                       'SGD': "1e-2",
                       'hidden_sizes': hidden_sizes,
                       'learn_to_stop': False}
        model = factories.model_factory(hyperparams, batch_scheduler)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))


    # Test fprop with missing streamlines from one subject in a batch
    output = model.get_output(trainset.symb_inputs)
    fct = theano.function([trainset.symb_inputs], output, updates=model.graph_updates)

    batch_inputs, batch_targets, batch_mask = batch_scheduler._next_batch(2)
    out = fct(batch_inputs)

    with Timer("Building optimizer"):
        loss = factories.loss_factory(hyperparams, model, trainset)
        optimizer = factories.optimizer_factory(hyperparams, loss)


    fct_loss = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask],
                                loss.loss,
                                updates=model.graph_updates)

    loss_value = fct_loss(batch_inputs, batch_targets, batch_mask)
    print("Loss:", loss_value)


    fct_optim = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask],
                                list(optimizer.directions.values()),
                                updates=model.graph_updates)

    dirs = fct_optim(batch_inputs, batch_targets, batch_mask)


    from ipdb import set_trace as dbg
    dbg()


if __name__ == "__main__":
    test_gru_regression_fprop()
