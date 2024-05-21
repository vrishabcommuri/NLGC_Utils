from ._supp_funcs import get_num_links
import mne
import itertools
import numpy as np
import pickle
from tabulate import tabulate
import matplotlib.pyplot as plt

def construct_link_matrix(self):
    link_count = dict() 

    for subject_vec in [self.CONTROLS, self.PATIENTS]:
        for subject in subject_vec:
            self.experiment.e.set(subject=subject)

            src_target = self.experiment.e.load_src(src='ico-1')

            try:
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

            except Exception as e:
                # check the mri/label folder for .lh and .rh parcellation files
                print(f"Ooops! Label issue with {subject}: {e}") 
                continue

            for lobe_src, lobe_tar in itertools.product(self.target_lobes, repeat=2):
                link_count[f"{subject};{lobe_src}->{lobe_tar}"] = np.zeros((len(self.visits), len(self.sessions), len(self.trials), 4))
            link_count[f"{subject};total"] = np.zeros((len(self.visits), len(self.sessions), len(self.trials), 4))

            for visit_idx, visit in enumerate(self.visits):
                for session_idx, session in enumerate(self.sessions):
                    for trial_idx, trial in enumerate(self.trials):
                        for file_pattern in self.file_patterns:
                            file_name = eval(f"f'{file_pattern}'")
                            try:
                                with open(file_name, 'rb') as fp: 
                                    NLGC_obj = pickle.load(fp)

                                J = NLGC_obj.get_J_statistics()
                                if J.sum() < self.minlinkcount or J.sum() > self.maxlinkcount:
                                    raise ValueError(f"link count {J.sum()} outside of analysis range {self.minlinkcount}-{self.maxlinkcount}!")
                                
                                self.models.append([file_name, f"{subject}{visit}{session}{trial}", NLGC_obj])

                                
                            except Exception as e:
                                print(f"caught exception: {e}")
                                print(f"continuing with zero link matrix for {subject}, v={visit}, s={session}, t={trial}")
                                self.models.append([file_name, f"{subject}{visit}{session}{trial}", None])
                                J = np.zeros([84, 84])
                            this_visit_link_count = get_num_links(J.copy(), self.target_lobes, src_target, 
                                                                    lobe_labels, normalization=False)
                            for key, item in this_visit_link_count.items():
                                link_count[f"{subject};{key}"][visit_idx][session_idx][trial_idx] = item    
                                link_count[f"{subject};total"][visit_idx][session_idx][trial_idx] += item
    self.link_count = link_count
    self.pairwise_rois = list(this_visit_link_count.keys())
    
    return self.link_count


def tabulate_links(self, percentages=True, verbose=True):
    if percentages:
        if verbose:
            print("\n These are <<percentages>> \n \n")
    else:
        if verbose:
            print("\n These are <<raw link counts>> \n \n")

    group_table = []
    conn = dict()
    conn_hemi = dict()
    conn_raw = dict()
    conn_hemi_raw = dict()

    for subject_vec in [self.CONTROLS, self.PATIENTS]:
        link_tables = []

        for visit_idx, visit in enumerate(self.visits):
            for session_idx, session in enumerate(self.sessions):
                for trial_idx, trial in enumerate(self.trials):
                    this_table = []
                    first_row = []

                    first_row.append(f"visit={visit}")
                    for subject in subject_vec:
                        first_row.append(subject)
                    this_table.append(first_row)

                    for key in self.pairwise_rois:
                        this_row = []
                        this_row.append(key)
                        this_subject_total = 0
                        for subject in subject_vec:
                            this_link_count = self.link_count[f"{subject};{key}"][visit_idx][session_idx][trial_idx].sum() 

                            # this_con = np.zeros((len(target_lobes), len(target_lobes)))
                            # ll, rr, lr, rl
                            this_con_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
                            this_con_hemi_raw = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
                            for j, source in enumerate(self.target_lobes):
                                for i, target in enumerate(self.target_lobes):
                                    if self.link_count[f"{subject};total"][visit_idx][session_idx][trial_idx].sum() != 0:
                                        this_con_hemi[:, i, j] = \
                                        self.link_count[f"{subject};{source}->{target}"][visit_idx][session_idx][trial_idx]\
                                        / self.link_count[f"{subject};total"][visit_idx][session_idx][trial_idx].sum()
                                        this_con_hemi_raw[:, i, j] = \
                                        self.link_count[f"{subject};{source}->{target}"][visit_idx][session_idx][trial_idx]

                                conn[f'{subject}{visit}{session}{trial}'] = this_con_hemi.sum(axis=0)
                                conn_raw[f'{subject}{visit}{session}{trial}'] = this_con_hemi_raw.sum(axis=0)
                                conn_hemi[f'{subject}{visit}{session}{trial}'] = this_con_hemi
                                conn_hemi_raw[f'{subject}{visit}{session}{trial}'] = this_con_hemi_raw

                            if percentages:
                                if self.link_count[f"{subject};total"][visit_idx][session_idx][trial_idx].sum() != 0:
                                    this_link_count /= self.link_count[f"{subject};total"][visit_idx][session_idx][trial_idx].sum()
                                this_row.append(round(this_link_count*100, 1))
                            else:
                                this_row.append(round(this_link_count, 2))
                        this_table.append(this_row)
                    link_tables.append(this_table)
        group_table.append(link_tables)

    for group_idx, subject_vec in enumerate([self.CONTROLS, self.PATIENTS]):
        status = 'patient' if subject_vec[0] in self.experiment.PATIENTS else 'control'
        if verbose:
            print('*'+status+'*\n')
        for visit_idx, visit in enumerate(self.visits):
            for session_idx, session in enumerate(self.sessions):
                if verbose:
                    print(tabulate(group_table[group_idx][visit_idx], headers='firstrow'))
                    print('\n\n')
        if verbose:
            print('\n\n\n')
    self.conn_hemi = conn_hemi
    self.conn_hemi_raw = conn_hemi_raw
    self.conn = conn
    self.conn_raw = conn_raw
    return group_table


def patch_to_ROI(src_target, labels):
    patch_to_labels = []

    for hemi_idx in [0, 1]:
        label_vert_dict = {label.name: label.get_vertices_used(src_target[hemi_idx]['vertno']) for label in
                           labels[hemi_idx::2]}
        # used_vert = np.sort(np.concatenate(tuple(label_vert_dict.values())))
        for vert in src_target[hemi_idx]['vertno']:
            match = None
            for key, item in label_vert_dict.items():
                if vert in item:
                    match = key
            patch_idx = np.where(src_target[hemi_idx]['vertno'] == vert)
            this_patch_info = []

            if match != None:
                this_patch_info = [(match, 1.0)]
            else:
                neighbours_assignment = np.asanyarray(
                    [len(label.get_vertices_used(src_target[hemi_idx]['pinfo'][patch_idx[0][0]]))
                     for label in labels[hemi_idx::2]], dtype=float)
                if np.all(neighbours_assignment == 0.0):
                    import random
                    rnd_label = random.randint(0, len(labels) // 2 - 1)
                    this_patch_info.append((labels[rnd_label * 2 + hemi_idx].name, 1.0))
                else:
                    neighbours_assignment /= np.sum(neighbours_assignment)
                    for label_idx, label in enumerate(labels[hemi_idx::2]):
                        if neighbours_assignment[label_idx] != 0.0:
                            this_patch_info.append((label.name, neighbours_assignment[label_idx]))
            patch_to_labels.append(this_patch_info)
    return patch_to_labels


def get_patch_idxs_in_region(self, subject, region, lobe_mapping, verbose):
    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=self.subjects_dir)
    label_names = [label.name for label in labels]

    lobe_labels = []
    target_lobes = list(lobe_mapping.keys())

    for lobe_idx, lobe_ in enumerate(target_lobes):
        for hemi_idx, hemi in enumerate(['lh', 'rh']):
            temp_label = []
            for ROI_idx, ROI in enumerate(lobe_mapping[lobe_]):
                if ROI_idx == 0:
                    temp_label = labels[label_names.index(f"{ROI}-{hemi}")]
                else:
                    temp_label += labels[label_names.index(f"{ROI}-{hemi}")]
            temp_label.name = f"{lobe_}-{hemi}"
            lobe_labels.append(temp_label)

    link_count = dict()
    for lobe_src, lobe_tar in itertools.product(target_lobes, repeat=2):
        link_count[f"{lobe_src}->{lobe_tar}"] = np.zeros(4)
    
    if subject == 'fsaverage':
        self.experiment.e.set(subject=subject, match=False)
    else:
        self.experiment.e.set(subject=subject)
    src_target = self.experiment.e.load_src(src='ico-1')

    patch_to_labels = patch_to_ROI(src_target, lobe_labels)
    idxs = []
    for idx, i in enumerate(patch_to_labels):
        # this patch straddles two or more regions, 
        # don't consider it for the sake of simplicity
        if i[0][1] != 1:
            continue

        # only want indices for this region
        if i[0][0].split('-')[0] != region:
            continue
        if verbose:
            print(idx, i)
        idxs.append(idx)
    return idxs


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
    return A
        

