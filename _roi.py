from ._supp_funcs import patch_to_ROI
import mne
import itertools
import numpy as np
import pickle
import pandas as pd
# from tabulate import tabulate
import matplotlib.pyplot as plt


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

    return df #B, transform, weights, B_noise


@staticmethod
def gen_noise_models(df, J, noise_model_seed, n_noise_models, noise_model_dim):
    if not noise_model_seed:
        noise_model_seed = 0
    B_noise = np.zeros((n_noise_models, noise_model_dim, noise_model_dim))

    for nmi in range(n_noise_models):
        df_noise = df.copy()
        # increment random state by 1 each noise model run for reproducibility
        df_noise = df_noise.groupby(df_noise.index >= len(df_noise)//2)\
            .sample(frac=1, random_state=noise_model_seed+nmi)
        df_noise = df_noise.reset_index(drop=True)
        B_n, _, w_n = transform_and_weight(df_noise, J)
        B_noise[nmi] = np.nan_to_num(B_n/w_n)

    B_noise = np.mean(B_noise, axis=0)

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


def construct_link_matrix(self):
    self.conn_raw = dict()
    self.conn = dict()
    self.conn_hemi_raw = dict()
    self.conn_hemi = dict()

    for subject_vec in [['fsaverage'], self.CONTROLS, self.PATIENTS]:
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
            
            
            self.experiment.e.set(subject=subject)

            src_target = self.experiment.e.load_src(src='ico-1')

            rois, roi_names = get_roi_mapping(subject, src_target, self.subjects_dir, self.target_lobes, self.lobe_mapping)

            for visit_idx, visit in enumerate(self.visits):
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


                            df = ico_to_desikan(rois, roi_names)
                            B, transform, weights = transform_and_weight(df, J)
                            B_noise = gen_noise_models(df, J, self.noise_model_seed, self.n_noise_models, len(roi_names))

                            if subject in self.reflect_subs:
                                B = unpartition(reflect(partition(B)))
                                B_noise = unpartition(reflect(partition(B_noise)))
                                weights = unpartition(reflect(partition(weights)))
                                # TODO fill this in later
                                transform = None 

                            self.models.append([file_name, f"{subject}{visit}{session}{trial}", NLGC_obj])
                            self.models_transform.append([subject, visit, session, trial, file_name, NLGC_obj, 
                                                          B, np.nan_to_num(B/weights), transform, weights, 
                                                          rois, roi_names, 
                                                          B_noise])


                            self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'] = partition(B)
                            self.conn_raw[f'{subject}{visit}{session}{trial}'] = partition(B).mean(axis=0)

                            self.conn_hemi[f'{subject}{visit}{session}{trial}'] = partition(B/weights)
                            self.conn[f'{subject}{visit}{session}{trial}'] = partition(B/weights).mean(axis=0)

                            


    self.models_transform = pd.DataFrame(self.models_transform)
    self.models_transform.columns = ['subject', 'visit', 'session', 'trial', 
              'file_name', 'NLGC_obj', 'B', 'B_norm', 'transform',
              'weights', 'rois', 'roi_names', 'B_noise']




# def construct_link_matrix(self):
#     link_count = dict() 
#     lobe_weights = dict()

#     for subject_vec in [self.CONTROLS, self.PATIENTS]:
#         for subject in subject_vec:
#             self.experiment.e.set(subject=subject)

#             src_target = self.experiment.e.load_src(src='ico-1')

#             try:
#                 labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=self.subjects_dir)
#                 label_names = [label.name for label in labels]

#                 lobe_labels = []

#                 for lobe_idx, lobe_ in enumerate(self.target_lobes):
#                     for hemi_idx, hemi in enumerate(['lh', 'rh']):
#                         temp_label = []
#                         for ROI_idx, ROI in enumerate(self.lobe_mapping[lobe_]):
#                             if ROI_idx == 0:
#                                 temp_label = labels[label_names.index(f"{ROI}-{hemi}")]
#                             else:
#                                 temp_label += labels[label_names.index(f"{ROI}-{hemi}")]
#                         temp_label.name = f"{lobe_}-{hemi}"
#                         lobe_labels.append(temp_label)

#             except Exception as e:
#                 # check the mri/label folder for .lh and .rh parcellation files
#                 print(f"Ooops! Label issue with {subject}: {e}") 
#                 continue

#             for lobe_src, lobe_tar in itertools.product(self.target_lobes, repeat=2):
#                 link_count[f"{subject};{lobe_src}->{lobe_tar}"] = np.zeros((len(self.visits), len(self.sessions), len(self.trials), 4))
#             link_count[f"{subject};total"] = np.zeros((len(self.visits), len(self.sessions), len(self.trials), 4))

#             for visit_idx, visit in enumerate(self.visits):
#                 for session_idx, session in enumerate(self.sessions):
#                     for trial_idx, trial in enumerate(self.trials):
#                         for file_pattern in self.file_patterns:
#                             file_name = eval(f"f'{file_pattern}'")
#                             try:
#                                 with open(file_name, 'rb') as fp: 
#                                     NLGC_obj = pickle.load(fp)

#                                 J = NLGC_obj.get_J_statistics()
#                                 if J.sum() < self.minlinkcount or J.sum() > self.maxlinkcount:
#                                     raise ValueError(f"link count {J.sum()} outside of analysis range {self.minlinkcount}-{self.maxlinkcount}!")
                                
#                                 self.models.append([file_name, f"{subject}{visit}{session}{trial}", NLGC_obj])

                                
#                             except Exception as e:
#                                 print(f"caught exception: {e}")
#                                 print(f"continuing with zero link matrix for {subject}, v={visit}, s={session}, t={trial}")
#                                 self.models.append([file_name, f"{subject}{visit}{session}{trial}", None])
#                                 J = np.zeros([84, 84])
#                             this_visit_link_count = get_num_links(J.copy(), self.target_lobes, src_target, 
#                                                                     lobe_labels, normalization=False)
#                             for key, item in this_visit_link_count.items():
#                                 link_count[f"{subject};{key}"][visit_idx][session_idx][trial_idx] = item    
#                                 link_count[f"{subject};total"][visit_idx][session_idx][trial_idx] += item
#     self.link_count = link_count
#     self.pairwise_rois = list(this_visit_link_count.keys())
    
#     return self.link_count


# def tabulate_links(self, percentages=True, verbose=True):
#     if percentages:
#         if verbose:
#             print("\n These are <<percentages>> \n \n")
#     else:
#         if verbose:
#             print("\n These are <<raw link counts>> \n \n")

#     group_table = []
#     conn = dict()
#     conn_hemi = dict()
#     conn_raw = dict()
#     conn_hemi_raw = dict()

#     for subject_vec in [self.CONTROLS, self.PATIENTS]:
#         link_tables = []

#         for visit_idx, visit in enumerate(self.visits):
#             for session_idx, session in enumerate(self.sessions):
#                 for trial_idx, trial in enumerate(self.trials):
#                     this_table = []
#                     first_row = []

#                     first_row.append(f"visit={visit}")
#                     for subject in subject_vec:
#                         first_row.append(subject)
#                     this_table.append(first_row)

#                     for key in self.pairwise_rois:
#                         this_row = []
#                         this_row.append(key)
#                         this_subject_total = 0
#                         for subject in subject_vec:
#                             if verbose >= 2:
#                                 print(subject_vec, visit_idx, session_idx, trial_idx)
#                             this_link_count = self.link_count[f"{subject};{key}"][visit_idx][session_idx][trial_idx].sum() 

#                             # this_con = np.zeros((len(target_lobes), len(target_lobes)))
#                             # ll, rr, lr, rl
#                             this_con_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
#                             this_con_hemi_raw = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
#                             for j, source in enumerate(self.target_lobes):
#                                 for i, target in enumerate(self.target_lobes):
#                                     if verbose >= 3:
#                                         print(f'checking link {j}->{i}')
#                                     if self.link_count[f"{subject};total"][visit_idx][session_idx][trial_idx].sum() != 0:
#                                         this_con_hemi[:, i, j] = \
#                                         self.link_count[f"{subject};{source}->{target}"][visit_idx][session_idx][trial_idx]\
#                                         / self.link_count[f"{subject};total"][visit_idx][session_idx][trial_idx].sum()
#                                         this_con_hemi_raw[:, i, j] = \
#                                         self.link_count[f"{subject};{source}->{target}"][visit_idx][session_idx][trial_idx]

#                                 conn[f'{subject}{visit}{session}{trial}'] = this_con_hemi.sum(axis=0)
#                                 conn_raw[f'{subject}{visit}{session}{trial}'] = this_con_hemi_raw.sum(axis=0)
#                                 conn_hemi[f'{subject}{visit}{session}{trial}'] = this_con_hemi
#                                 conn_hemi_raw[f'{subject}{visit}{session}{trial}'] = this_con_hemi_raw

#                             if percentages:
#                                 if self.link_count[f"{subject};total"][visit_idx][session_idx][trial_idx].sum() != 0:
#                                     this_link_count /= self.link_count[f"{subject};total"][visit_idx][session_idx][trial_idx].sum()
#                                 this_row.append(round(this_link_count*100, 1))
#                             else:
#                                 this_row.append(round(this_link_count, 2))
#                         this_table.append(this_row)
#                     link_tables.append(this_table)
#         group_table.append(link_tables)

#     for group_idx, subject_vec in enumerate([self.CONTROLS, self.PATIENTS]):
#         status = 'patient' if subject_vec[0] in self.PATIENTS else 'control'
#         if verbose:
#             print('*'+status+'*\n')
#         for visit_idx, visit in enumerate(self.visits):
#             for session_idx, session in enumerate(self.sessions):
#                 if verbose:
#                     print(tabulate(group_table[group_idx][visit_idx], headers='firstrow'))
#                     print('\n\n')
#         if verbose:
#             print('\n\n\n')
#     self.conn_hemi = conn_hemi
#     self.conn_hemi_raw = conn_hemi_raw
#     self.conn = conn
#     self.conn_raw = conn_raw
#     return group_table


# def desikan_to_ico(self, region, lobe_mapping, verbose=False, subject='fsaverage'):
#     labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=self.subjects_dir)
#     label_names = [label.name for label in labels]

#     lobe_labels = []
#     target_lobes = list(lobe_mapping.keys())

#     for lobe_idx, lobe_ in enumerate(target_lobes):
#         for hemi_idx, hemi in enumerate(['lh', 'rh']):
#             temp_label = []
#             for ROI_idx, ROI in enumerate(lobe_mapping[lobe_]):
#                 if ROI_idx == 0:
#                     temp_label = labels[label_names.index(f"{ROI}-{hemi}")]
#                 else:
#                     temp_label += labels[label_names.index(f"{ROI}-{hemi}")]
#             temp_label.name = f"{lobe_}-{hemi}"
#             lobe_labels.append(temp_label)

#     link_count = dict()
#     for lobe_src, lobe_tar in itertools.product(target_lobes, repeat=2):
#         link_count[f"{lobe_src}->{lobe_tar}"] = np.zeros(4)
    
#     if subject == 'fsaverage':
#         self.experiment.e.set(subject=subject, match=False)
#     else:
#         self.experiment.e.set(subject=subject)
#     src_target = self.experiment.e.load_src(src='ico-1')

#     patch_to_labels = patch_to_ROI(src_target, lobe_labels)
#     idxs = []
#     for idx, i in enumerate(patch_to_labels):
#         # this patch straddles two or more regions, 
#         # don't consider it for the sake of simplicity
#         if i[0][1] != 1:
#             if verbose:
#                 print("skipping", idx, i, "straddles regions")
#             continue

#         # only want indices for this region
#         if i[0][0].split('-')[0] != region:
#             continue
#         if verbose:
#             print(idx, i)
#         idxs.append(idx)
#     return idxs


def lkm_to_matrix(self, lkm, verbose=False):
    d = dict()
    for reg in list(self.lobe_mapping.keys()):
        idxs = self.get_patch_idxs_in_region("fsaverage", reg, self.lobe_mapping, verbose)
        d[reg] = idxs

    A = np.zeros((84,84))


    for i in range(84):
        for j in range(84):
            if i < 42:
                hemi1 = 'lh'
            else:
                hemi1 = 'rh'
            if j < 42:
                hemi2 = 'lh'
            else:
                hemi2 = 'rh'

            reg1 = None
            reg2 = None
            for reg_idx, reg in enumerate(list(self.lobe_mapping.keys())):
                if i in d[reg]:
                    reg1 = reg_idx
                if j in d[reg]:
                    reg2 = reg_idx

            if reg1 is None or reg2 is None:
                continue

            idx = self._get_hemi_idx(hemi1, hemi2)
            val = lkm[idx][reg2, reg1]

            A[j, i] = val

    
    remap = []
    for key in list(self.lobe_mapping.keys()):
        remap.extend(d[key])
    remap = remap + list(np.setdiff1d(list(range(84)), remap))
    
    if verbose:
        plt.imshow(A)
        plt.show()
        plt.imshow(A[remap, :][:, remap])
        plt.show()
    return A, remap
        

