class LinkAnalyzer:
    from ._roi import (construct_link_matrix, 
                       partition, unpartition, reflect)
    from ._plotting import (heatmap, heatmap_linkmatrix, circle_plot, railroad_plot, 
                           sankey, hammer_plot, hammer_plot_matrix, _hammer_plot)
    from ._stats import (group_whole_average, group_hemi_average,
                        t_test_between, _condition_average)
    from ._natures import get_natures, _get_natures
    from ._trf import fit_trfs, subset_trfs_by_region, get_trf_statmap

    def __init__(self, experiment, controls, patients, visits, sessions, trials,
                file_patterns="./Results/[{subject}]-[visit={visit}]-[beta].p",
                lobe_mapping=None, subjects_dir=None, exclusions=None,
                minlinkcount=0, maxlinkcount=84*84, 
                reflect_subs=[]):
        self.experiment = experiment
        if not exclusions is None:
            controls = [i for i in controls if i not in exclusions]
            patients = [i for i in patients if i not in exclusions]
        self.CONTROLS = controls
        self.PATIENTS = patients
        
        if isinstance(file_patterns, str):
            self.file_patterns = [file_patterns]
        else:
            self.file_patterns = file_patterns
        self.visits = visits
        self.sessions = sessions
        self.trials = trials
        self.visitmapping = dict()
        for i, v in enumerate(visits):
            self.visitmapping[v] = i
            
        self.conn = None
        self.conn_hemi = None
        self.conn_raw = None
        self.conn_hemi_raw = None
        
        self.models = []
        self.models_transform = []
        self.fsaverage_ico_morph = None
        self.fsaverage_anat_morph = None
        self.fsaverage_sym_ico_morph = None
        self.fsaverage_sym_anat_morph = None

        self.reflect_subs = reflect_subs
        
        self.subjects_dir = subjects_dir
        if self.subjects_dir is None:
            self.subjects_dir = "../data/mri"
        
        if not lobe_mapping is None:
            self.lobe_mapping = lobe_mapping
        else:
            self.lobe_mapping = { # lobe mapping parcellation
                "F1" : ["frontalpole", "rostralmiddlefrontal", 
                                       "parsorbitalis","lateralorbitofrontal", 
                                       "medialorbitofrontal", "rostralanteriorcingulate"],

                "F2" : ["superiorfrontal", 
                           "parsopercularis", "parstriangularis", 
                           "caudalanteriorcingulate", "caudalmiddlefrontal", 
                           "precentral", "paracentral"],

                "P" : ["postcentral", "supramarginal",
                              "superiorparietal", "posteriorcingulate", "isthmuscingulate",
                              "inferiorparietal", "precuneus"],

                "T" : ["superiortemporal", "bankssts", "transversetemporal", 
                              "middletemporal", "fusiform", "temporalpole",
                              "inferiortemporal", "entorhinal", "parahippocampal"],
            }
        self.target_lobes = list(self.lobe_mapping.keys())
        
        self.minlinkcount = minlinkcount
        self.maxlinkcount = maxlinkcount


        self.construct_link_matrix()
    


    @staticmethod
    def _get_hemi_idx(source_hemi, target_hemi):
        if (source_hemi, target_hemi) == ('rh', 'rh'):
            hemi_idx = 1
        elif (source_hemi, target_hemi) == ('lh', 'rh'):
            hemi_idx = 2
        elif (source_hemi, target_hemi) == ('rh', 'lh'):
            hemi_idx = 3
        else:
            hemi_idx = 0
        return hemi_idx
    
