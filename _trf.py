import eelbrain as eel
import numpy as np
import scipy
import pathlib


def generate_acoustic_predictors(datadir, stims):
    # if pathlib.Path(datadir / "envelopes.pickle").is_file():
    #     envelopes = eel.load.unpickle(str(datadir / "envelopes.pickle"))
    # else:
    #     envelopes = dict()
    #     for i in stims.keys():
    #         # this is obtained out of band by running the shamma lab cochlear
    #         # model matlab script over the wav file. 
    #         mat = scipy.io.loadmat(str(datadir / f"/stimuli/{i}-audspec-1000.mat"))
    #         f, fs, Sxx = mat['frequencies'].flatten(), mat['samplingrate'].flatten(), mat['specgram']
    #         Sxx = Sxx.astype(np.float64)
    #         freqs = eel.Scalar('frequency', f, 'Hz')
    #         spec = eel.NDVar(Sxx, dims=(eel.UTS(0, 1/fs, len(Sxx)), freqs))
    #         envelopes[i] = spec.mean('frequency')

    #     eel.save.pickle(envelopes, str(datadir / "envelopes.pickle"))

    if pathlib.Path(datadir / "envelope_onsets.pickle").is_file():
        envelope_onsets = eel.load.unpickle(str(datadir / "envelope_onsets.pickle"))
    else:
        envelope_onsets = dict()
        for i in stim.keys():
            wav = stims[i]
            gt = gammatone_bank(wav, 20, 5000, 256, location='left', pad=True)
            gt = eel.resample(gt, 1000) # change to 1000 for new exp
            gt = gt.clip(0, out=gt) # remove resampling artifacts

            # apply powerlaw compression
            gt = (gt + 1).log() # log envelope
            gt_on = edge_detector(gt, c=30)
            gt_on_pred = gt_on.sum('frequency')
            envelope_onsets[i] = gt_on_pred
            envelope_onsets[i].name = f"{i}.wav"

        eel.save.pickle(envelope_onsets, str(datadir / "envelope_onsets.pickle"))

    return envelope_onsets



def get_patch_to_label(self, subject):
    self.experiment.e.set(subject=subject)
    src_target = self.experiment.e.load_src(src='ico-1')

    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=self.subjects_dir)
    label_names = [label.name for label in labels]

    lobe_labels = []
    for lobe_idx, lobe_ in enumerate(self.target_lobes):
        for hemi_idx, hemi in enumerate(['lh', 'rh']):
            temp_label = []
            for ROI_idx, ROI in enumerate(self.lobe_mapping[lobe_]):
                if ROI_idx == 0:
                    temp_label = labels[label_names.index(f"{ROI}-{hemi}")]
                else:
                    temp_label += labels[label_names.index(f"{ROI}-{hemi}")]
            temp_label.name = f"{lobe_}-{hemi}"
            lobe_labels.append(temp_label)

    return self.patch_to_ROI(src_target, lobe_labels)


def fit_trfs(self, condition, stimmapping, stims, datadir, fs=25, 
            boosting_kwargs=None):
    """
    fit trfs to all eigenmodes for all provided conditions

    the basic idea is that we use nlgc as a source estimation procedure which
    will give us an estimate of the current time course at the location of each
    eigenmode. so we can therefore fit a trf to each eigenmode. we later can
    compare eigenmode trfs between regions and between linked eigenmode sets of
    eigenmodes.

    condition: list, containing condition information stimmapping: dict, mapping
    from condition to the stimulus name stims: dict, stimulus name and
    corresponding stimulus waveform factor: float, only look at links greater
    than this J value
    """

    if boosting_kwargs is None:
        boosting_kwargs = {'tstart':-0.050, 'tstop':0.250, 'scale_data':True, 
                            'error':'l1', 'basis':0.01, 
                            'basis_window':'hamming', 'partitions':4, 
                            'selective_stopping':False}

    envon = generate_acoustic_predictors(datadir, stims)

    trfs = []

    # fit a trf to each condition
    for subject, visit, session, trial in condition:
        # get the right stimulus for this particular run
        key = stimmapping[f"{subject}{visit}{session}{trial}"]
        stim = stims[key]

        # get the eigenmodes for this region
        model = None
        for _, key, _model in self.models:
            if key == f"{subject}{visit}{session}{trial}":
                model = _model
                break
        if model is None:
            continue

        A = model._model_f[0]._parameters[0]
        neigs = A.shape[1]//84

        # fit a trf to each eigenmode
        for eigno in range(neigs):
            # get the current eigenmode time series
            eis = model._model_f[0]._parameters[4][:,:neigs*84].T[eigno::neigs].T

            # TODO there is probably a way to do this in a single call, by
            # having eelbrain boost all eigenmodes simultaneously, but I don't
            # know what it is. eel.Case treats all eigenmodes as sharing a
            # filter, which is definitely not right.
            for i in range(84):
                ei = eel.NDVar(eis[:, i], dims=(eel.UTS(0, 1/fs, eis.shape[0])))

                boost_result = eel.boosting(y=ei, x=[envons], **boosting_kwargs)

                trfs.append([eigno, i, boost_result])
    
    return trfs


                

