import numpy as np
import logging
import os

from h5flow.core import H5FlowStage
from h5flow import resources


class PickupNoiseAdder(H5FlowStage):
    '''
        Applies a signal on each light detector channel (``j``) with designated shape and a random amplitude.::

            waveform_new[i] = random.normal[i] * scale[j] + mean[j]) * template[j][i + random.uniform[j] * len(waveform)] + waveform[i]

        Parameters:
         - ``in_dset_name`` : ``str``, required, input dataset path for waveforms
         - ``out_dset_name`` : ``str``, required, output dataset path for waveforms
         - ``noise_template_file`` : ``str``, optional, path to templates for pickup on each channel
         - ``noise_mean`` : ``list`` of ``list`` of mean for amplitude distribution, required
         - ``noise_std`` : ``list`` of ``list`` of std for amplitude distribution, required

        ``in_dset_name`` is required in the data cache.

        Example config::

            pickup_noise:
                classname: PickupNoiseAdder
                requires:
                    - 'light/wvfm'
                params:
                    out_dset_name: 'light/wvfm/w_noise'
                    in_dset_name: 'light/wvfm'
                    noise_mean:
                      - []
                      - []
                    noise_std:
                      - []
                      - []
                    template_file: 'data/module0_flow/noise_template_10MHz.npy'

        Uses the same dtype as the input waveform dataset except with ``'samples'`` converted to floats.

        Template file should be a numpy .npy file containing an array of shape: ``(nadc, nchan, nsamples)``

    '''
    class_version = '0.0.0'

    def __init__(self, **params):
        super(PickupNoiseAdder, self).__init__(**params)

        self.in_dset_name = params['in_dset_name']
        self.out_dset_name = params['out_dset_name']
        self.noise_mean = params.get('noise_mean')
        self.noise_std = params.get('noise_std')
        self.template_file = params.get('template_file')
        self.template = None
        self.apply_noise = True

        
    def init(self, source_name):
        super(PickupNoiseAdder, self).init(source_name)

        # set up noise model
        in_dset = self.data_manager.get_dset(self.in_dset_name)
        if self.noise_mean is not None:
            self.noise_mean = np.array(self.noise_mean)
        else:
            self.noise_mean = np.zeros(in_dset.dtype['samples'].shape[:-1])
        if self.noise_std is not None:
            self.noise_std = np.array(self.noise_std)
        else:
            self.noise_std = np.zeros(in_dset.dtype['samples'].shape[:-1])

        assert self.noise_mean.shape == in_dset.dtype['samples'].shape[:-1], f'Mis-matched noise mean shape is {self.noise_mean.shape}, expected {in_dset.dtype["samples"].shape[:-1]}'
        assert self.noise_std.shape == in_dset.dtype['samples'].shape[:-1], f'Mis-matched noise std shape is {self.noise_std.shape}, expected {in_dset.dtype["samples"].shape[:-1]}'
        if self.template_file is not None and os.path.exists(self.template_file):
            self.template = np.load(self.template_file)
        else:
            if self.rank == 0:
                logging.warning(f'Template file {self.template_file} is None or does not exist! Skipping noise application')
            self.template = np.empty(self.noise_mean.shape + (0,))
            self.apply_noise = False

        # only apply noise to simulation
        if not resources['RunData'].is_mc:
            self.apply_noise = False

        # save all config info
        self.data_manager.set_attrs(self.out_dset_name,
                                    classname=self.classname,
                                    class_version=self.class_version,
                                    source_dset=source_name,
                                    apply_noise=self.apply_noise,
                                    in_dset_name=self.in_dset_name,
                                    noise_mean=self.noise_mean,
                                    noise_std=self.noise_std,
                                    template_file=self.template_file,
                                    )
        self.data_manager.create_dset(self.out_dset_name + '/template', dtype=self.template.dtype)
        self.data_manager.reserve_data(self.out_dset_name + '/template', slice(0, self.template.size))
        self.data_manager.write_data(self.out_dset_name + '/template', slice(0, self.template.size), self.template.ravel())
        self.data_manager.set_attrs(self.out_dset_name + '/template', shape=self.template.shape)

        # then set up new datasets
        self.out_dtype = in_dset.dtype
        self.data_manager.create_dset(self.out_dset_name, dtype=self.out_dtype)
        self.data_manager.create_ref(source_name, self.out_dset_name)

    def run(self, source_name, source_slice, cache):
        super(PickupNoiseAdder, self).run(source_name, source_slice, cache)

        wvfm_data = cache[self.in_dset_name].data # (don't worry about masked events)

        if self.apply_noise:
            noise_prefactor = np.random.normal(size=wvfm_data['samples'].shape[:-1], loc=self.noise_mean[np.newaxis], scale=self.noise_std[np.newaxis])
            noise_phase = np.random.uniform(size=wvfm_data['samples'].shape[:-1]) * self.template.shape[-1]
            noise_idx = (np.indices(wvfm_data['samples'].shape)[-1] + noise_phase[...,np.newaxis].astype(int)) % self.template.shape[-1]
            noise = np.take_along_axis(noise_prefactor[..., np.newaxis] * self.template[np.newaxis, np.newaxis], noise_idx, axis=-1)

            out_wvfm = wvfm_data.copy()
            out_wvfm['samples'] += noise.astype(out_wvfm.dtype['samples'].base)
        else:
            out_wvfm = wvfm_data.copy()

        # reserve new data
        out_slice = self.data_manager.reserve_data(self.out_dset_name, source_slice)
        self.data_manager.write_data(self.out_dset_name, source_slice, out_wvfm.ravel())

        # save references
        ref = np.c_[source_slice, out_slice]
        self.data_manager.write_ref(source_name, self.out_dset_name, ref)
