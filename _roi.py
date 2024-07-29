from ._supp_funcs import patch_to_ROI
import mne
import itertools
import numpy as np
import pickle
import pandas as pd
# from tabulate import tabulate
import matplotlib.pyplot as plt
import eelbrain as eel


@staticmethod
def partition(mat):
    n, n = mat.shape
    n = n//2
    A = np.zeros((4, n, n))
    A[0] = mat[:n, :n]
    A[1] = mat[n:, n:]
    A[2] = mat[n:, :n]
    A[3] = mat[:n, n:]
    return A

@staticmethod
def unpartition(mat):
    n, n = mat[0].shape
    A = np.zeros((n*2, n*2))
    A[:n, :n] = mat[0]
    A[n:, n:] = mat[1]
    A[n:, :n] = mat[2]
    A[:n, n:] = mat[3]
    return A

@staticmethod
def reflect(mat):
    return mat[[1, 0, 3, 2], ...]

@staticmethod
def transform_and_weight(df, J):
    transform = df.T.values

    # raw link counts within each anatomical region (84x84 -> 68x68)
    B = transform @ J @ transform.T 
    
    # weights tell us how eigenmodes reside in each anatomical region
    # dim(weights) = 68x1
    weights = transform.sum(axis=1)[:, np.newaxis] 

    # weight matrix telling us how many possible pairs of eigenmodes can communicate
    # for region pairs. important since some regions are smaller than others so
    # a change in a few eigenmodes there produces a higher percentage change than
    # the same change in larger regions
    weights = weights @ weights.T
    return B, transform, weights

@staticmethod
def ico_to_desikan(rois, roi_names):
    # each anatomical roi is mapped to by an 84-dim vector in ico-1 source space
    # we will construct a mapping between ico-1 and anatomical parcellations
    # via a 68 x 84 matrix that we can apply to the ico-1 (84x84) connectivity data
    A = {i:np.zeros(84) for i in roi_names}

    for idx, mapping in enumerate(rois):
        for elem in mapping:
            # mapping is a list telling us how each ico-1 patch is divided
            # e.g., [(LTC-lh, 0.75), (PC-lh, 0.25)] means that the patch 
            # is 75% in the lh LTC and 25% in the lh PC
            A[elem[0]][idx] += elem[1]

    df = pd.DataFrame(A)
    df = df[roi_names]

    return df


@staticmethod
def gen_noise_models(J):
    B_noise = partition(np.ones_like(J))
    hemi_noise_floor = partition(J).mean(axis=1).mean(axis=1)
    for i in range(4):
        B_noise[i] *= hemi_noise_floor[i] 
    
    B_noise = unpartition(B_noise)

    return B_noise

@staticmethod
def get_roi_mapping(subject, src_target, subjects_dir, target_lobes, lobe_mapping):
    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
    label_names = [label.name for label in labels]

    lobe_labels = []
    roi_names_lh = []
    roi_names_rh = []

    for lobe_idx, lobe_ in enumerate(target_lobes):
        for hemi_idx, hemi in enumerate(['lh', 'rh']):
            temp_label = []
            for ROI_idx, ROI in enumerate(lobe_mapping[lobe_]):
                if ROI_idx == 0:
                    temp_label = labels[label_names.index(f"{ROI}-{hemi}")]
                else:
                    temp_label += labels[label_names.index(f"{ROI}-{hemi}")]
            name = f"{lobe_}-{hemi}"
            temp_label.name = name
            if hemi == 'lh':
                roi_names_lh.append(name)
            else:
                roi_names_rh.append(name)
            lobe_labels.append(temp_label)
    
    # need to separate hemi names this way since patch_to_ROI  
    # expects lh rh labels to be in sequential pairs for each region
    roi_names = roi_names_lh + roi_names_rh

    # map ico-1 sources (84) to anatomical sources (68)
    rois = patch_to_ROI(src_target, lobe_labels)

    return rois, roi_names

@staticmethod
def morph_symmetrical(J, src, subjects_dir):
    src = eel.SourceSpace.from_mne_source_spaces(src, 'ico-1', subjects_dir, parc='aparc')

    # connectivity matrices ought to be of shape (source space, source space), but eelbrain
    # doesn't allow for that. instead, we create a dummy dimension (case, source space) and 
    # transform only the latter to obtain (case, transformed source space). Then transpose 
    # case and source space dimensions and transform the latter axis again. The result is
    # of shape (transformed source space, transformed source space) which is still 84x84 
    # but with entries morphed so that left and right hemispheres are symmetrical.
    ndv = eel.NDVar(J, dims=(eel.Case, src))
    ndvs = eel.xhemi(ndv)
    arr_half = np.concatenate([ndvs[0].x, ndvs[1].x], axis=1).T
    ndv = eel.NDVar(arr_half, dims=(eel.Case, src))
    ndvs = eel.xhemi(ndv)
    arr_corrected = np.concatenate([ndvs[0].x, ndvs[1].x], axis=1).T
    return arr_corrected


def construct_link_matrix(self):
    self.conn_raw = dict()
    self.conn = dict()
    self.conn_hemi_raw = dict()
    self.conn_hemi = dict()

    for subject_vec in [['fsaverage', 'fsaverage_sym'], self.CONTROLS, self.PATIENTS]:
        for subject in subject_vec:
            if subject == 'fsaverage':
                self.experiment.e.set(subject=subject, match=False)
                src_target = self.experiment.e.load_src(src='ico-1')
                rois, roi_names = get_roi_mapping(subject, src_target, self.subjects_dir, self.target_lobes, self.lobe_mapping)
                df = ico_to_desikan(rois, roi_names)
                
                
                # 68 x 84 (ico->anatomical)
                self.fsaverage_anat_morph = df.T.values

                # 84 x 68 (anatomical->ico)
                self.fsaverage_ico_morph = np.linalg.pinv(self.fsaverage_anat_morph)
                continue
                
            if subject  == 'fsaverage_sym':
                self.experiment.e.set(subject=subject, match=False)
                src_target = self.experiment.e.load_src(src='ico-1')
                rois, roi_names = get_roi_mapping(subject, src_target, self.subjects_dir, self.target_lobes, self.lobe_mapping)
                df = ico_to_desikan(rois, roi_names)
                
                # 68 x 84 (ico->anatomical)
                self.fsaverage_sym_anat_morph = df.T.values

                # 84 x 68 (anatomical->ico)
                self.fsaverage_sym_ico_morph = np.linalg.pinv(self.fsaverage_sym_anat_morph)
                continue
      
            self.experiment.e.set(subject=subject)

            for visit_idx, visit in enumerate(self.visits):
                self.experiment.e.set(visit=visit)
                src_target = self.experiment.e.load_src(src='ico-1')
                rois, roi_names = get_roi_mapping(subject, src_target, self.subjects_dir, self.target_lobes, self.lobe_mapping)
                
                for session_idx, session in enumerate(self.sessions):
                    for trial_idx, trial in enumerate(self.trials):
                        for file_pattern in self.file_patterns:
                            file_name = eval(f"f'{file_pattern}'")
                            try:
                                with open(file_name, 'rb') as fp: 
                                    NLGC_obj = pickle.load(fp)

                                J = NLGC_obj.get_J_statistics()

                                # no zero link matrices to ensure averages aren't thrown off
                                if J.sum() == 0: 
                                    raise ValueError(f"link count {J.sum()} zero!")

                                if J.sum() < self.minlinkcount or J.sum() > self.maxlinkcount: 
                                    raise ValueError(f"link count {J.sum()} is outside of analysis range {self.minlinkcount}-{self.maxlinkcount}!")

                            except Exception as e:
                                print(f"caught exception: {e}")
                                print(f"continuing with NaN link matrix for {subject}, v={visit}, s={session}, t={trial}")
                                self.models.append([file_name, f"{subject}{visit}{session}{trial}", None])
                                J = np.empty([84, 84])
                                J[:] = np.nan
                                NLGC_obj = None

                            J_sym = morph_symmetrical(J, src_target, self.subjects_dir)

                            df = ico_to_desikan(rois, roi_names)
                            B, transform, weights = transform_and_weight(df, J)
                            J_noise = gen_noise_models(J)
                            B_noise, _, _ = transform_and_weight(df, J_noise)

                            if subject in self.reflect_subs:
                                B = unpartition(reflect(partition(B)))
                                B_noise = unpartition(reflect(partition(B_noise)))
                                weights = unpartition(reflect(partition(weights)))
                                part_nlabels = len(transform)//2
                                
                                # swap top and bottom halves
                                transform = np.concatenate([transform[part_nlabels:, :], transform[:part_nlabels, :]], axis=0) 
                               
                                # swap operation should yield the same weight matrix
                                assert(np.all((transform.sum(axis=1)[:, np.newaxis] @ transform.sum(axis=1)[:, np.newaxis].T) == weights))

                            self.models.append([file_name, f"{subject}{visit}{session}{trial}", NLGC_obj])
                            self.models_transform.append([subject, visit, session, trial, file_name, NLGC_obj, 
                                                          B, np.nan_to_num(B/weights), transform, weights, 
                                                          rois, roi_names, 
                                                          np.nan_to_num(B_noise/weights), J, J_sym, df])

                            self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'] = partition(B)
                            self.conn_hemi[f'{subject}{visit}{session}{trial}'] = partition(B/weights)

    self.models_transform = pd.DataFrame(self.models_transform)
    self.models_transform.columns = ['subject', 'visit', 'session', 'trial', 
              'file_name', 'NLGC_obj', 'B', 'B_norm', 'transform',
              'weights', 'rois', 'roi_names', 'B_noise', 'J', 'J_sym', 'ico_to_desikan']
        

