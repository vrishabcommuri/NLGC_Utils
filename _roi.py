from ._supp_funcs import get_num_links
import mne
import itertools
import numpy as np
import pickle

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