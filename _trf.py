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


def _format_region(region, hemi):
    region = region.strip()

    if hemi == True:
        idx = region.find("->")
        srclobe, srchemi = region[:idx].split('-')
        dstlobe, dsthemi = region[idx+2:].split('-')
    else:
        idx = region.find("->")
        srclobe = region[:idx]
        dstlobe = region[idx+2:]
        srchemi, dsthemi = None, None

    return srclobe, dstlobe, srchemi, dsthemi


def _format_regionlist(regionlist, hemi):
    srclobes, dstlobes, srchemis, dsthemis = [], [], [], []

    for region in regionlist:
        srclobe, dstlobe, srchemi, dsthemi = _format_region(region, hemi)
        srclobes.append(srclobe)
        dstlobes.append(dstlobe)
        srchemis.append(srchemi)
        dsthemis.append(dsthemi)

    return srclobes, dstlobes, srchemis, dsthemis


def subset_trfs_by_region(self, trfs, regionlist, factor=0.95, hemi=False):
    """
    trfs: list, output by fit_trfs, consisting of 
          [eigenmode number, eigenmode index, boost result, nlgc model]
    regionlist: list of str, if hemi is false, of the form 'Frontal->Parietal' else includes hemi
            info in string like 'Frontal-lh->Parietal-rh'
    factor: float, only look at significant eigenmodes above this factor. If 0.0, then
            look at all eigenmodes within region.
    hemi: bool, look at individual hemis or both.
    """

    srclobes, dstlobes, srchemis, dsthemis = _format_regionlist(regionlist, hemi)

    srctrfs = dict()
    dsttrfs = dict()
    for eino, ei, boost_result, model in trfs:
        J = model.get_J_statistics()
        if factor > 0.0:
            J = J > factor
        else:
            # J is used for indexing, so here just look at all eigenmodes 
            J = np.eye(84) 

        dstidxs, srcidxs = np.nonzero(J)

        # according to J statistics, this eigenmode is an active source
        if ei in srcidxs:
            for tup_src in patch_to_labels[ei]:
                this_src_lobe = tup_src[0][:-3]
                if this_src_lobe not in srclobes:
                    continue
                
                if hemi:
                    this_src_hemi = tup_src[0][-2:]
                    if [this_src_lobe, this_src_hemi] not in list(zip(srclobes, srchemis)):
                        continue

                weight = tup_src[1]

                if not eino in srctrfs.keys():
                    srctrfs[eino] = []
                
                srctrfs[eino].append([this_src_lobe, this_src_hemi, ei, boost_result, weight])

        # according to J statistics, this eigenmode is a destination from an
        # active source
        if ei in dstidxs:
            for tup_dst in patch_to_labels[ei]:
                this_dst_lobe = tup_dst[0][:-3]
                if this_dst_lobe not in dstlobes:
                    continue
                
                if hemi:
                    this_dst_hemi = tup_dst[0][-2:]
                    if [this_dst_lobe, this_dst_hemi] not in list(zip(dstlobes, dsthemis)):
                        continue

                weight = tup_dst[1]

                if not eino in dsttrfs.keys():
                    dsttrfs[eino] = []
                
                dsttrfs[eino].append([this_dst_lobe, this_dst_hemi, ei, boost_result, weight])

    return srctrfs, dsttrfs



def fit_trfs(self, condition, stimmapping, stims, fs=25, nlgc_starttime=10,
            nlgc_endtime=50, boostingfs=250, boosting_kwargs=None, 
            verbose=False):
    """
    fit trfs to all eigenmodes for all provided conditions

    the basic idea is that we use nlgc as a source estimation procedure which
    will give us an estimate of the current time course at the location of each
    eigenmode. so we can therefore fit a trf to each eigenmode. we later can
    compare eigenmode trfs between regions and between linked eigenmode sets of
    eigenmodes.

    condition: list, containing condition information 
    stimmapping: dict, mapping from condition to the stimulus name 
    stims: dict or list of dicts, stimulus name and corresponding stimulus waveform 
    fs: int, samplerate of the nlgc model
    nlgc_starttime: int, seconds, start time of nlgc model fit 
                   (usually offset by 5 or 10 sec from start trigger)
    nlgc_starttime: int, seconds, end time of nlgc model fit 
                   (usually precedes the end trigger by 5 or 10 sec)
    boostingfs: samplerate to use for boosting (should be higher than fs) 
    boosting_kwargs: dict, parameters for boosting algorithm
    verbose: bool
    """

    if boosting_kwargs is None:
        boosting_kwargs = {'tstart':-0.050, 'tstop':0.250, 'scale_data':True, 
                            'error':'l1', 'basis':0.01, 
                            'basis_window':'hamming', 'partitions':4, 
                            'selective_stopping':False}


    trfs = []
    noisemodels = {1:[], 2:[], 3:[]}

    # fit a trf to each condition
    for subject, visit, session, trial in condition:
        if verbose:
            print(f"fitting trfs to subject {subject}, {visit} {session} {trial}")
        # get the right stimulus for this particular run
        
        stimlist = []
        key = stimmapping[f"{subject}{visit}{session}{trial}"]

        if isinstance(stims, dict):
            stimlist.append(stims[key])
        else:
            for stim in stims:
                stimlist.append(stim[key])

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
                ei = eel.NDVar(eis[:, i], dims=(eel.UTS(nlgc_starttime, 1/fs, eis.shape[0])))
                ei = ei.sub(time=(nlgc_starttime, nlgc_endtime))
                ei = eel.resample(ei, boostingfs)
                
                boost_result = eel.boosting(y=ei, x=stimlist, **boosting_kwargs)
                trfs.append([eigno, i, boost_result, model])

                for i in range(1, 4):
                    stimlist_noise = []
                    for stim in stimlist:
                        noise = stim.copy()
                        noise.x = np.roll(noise.x, (len(noise.x)//4)*i)
                        stimlist_noise.append(noise)

                    boost_result = eel.boosting(y=ei, x=stimlist, **boosting_kwargs)
                    noisemodels[i].append([eigno, i, boost_result, model])
    return trfs, noisemodels


def get_trf_statmap(self, trfs, noisemodels, tstart=0.0, tstop=0.250, tail=0):
    """
    trfs: list of NDVars, obtained by taking the output of fit_trfs by trfs = [i[2].h[0] for i in trfoutput]
    noisemodels: list of NDVars obtained in the same way
    tstart: float seconds, time to start looking for significant peaks
    tstop: float seconds, time to stop looking for significant peaks
    """
    assert(len(trfs)*3 == len(noisemodels))
    dst = eel.Dataset({"predictor":eel.Factor(["noperm"]*len(trfs) + ["perm"]*len(trfs)\
                                              + ["perm"]*len(trfs) + ['perm']*len(trfs), 
                                              name="permute")})
    trfs_comb = eel.concatenate(trfs + noisemodels, dim="case")

    dst['trial'] = eel.Factor(list(range(len(trfs_)))*4, name="trial")

    dst['trfs'] = trfs_comb
    res = eel.testnd.TTestRelated(
        'trfs', 'predictor', 'noperm', 'perm', ds=dst, match='trial',
        tfce=True, samples=10000, tail=0,
        tstart=tstart,  # Find clusters in the time window from 0 ...
        tstop=tstop,  # ... to 60 ms
    )
    return res


                

