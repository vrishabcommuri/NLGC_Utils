class LinkAnalyzer:
    from ._roi import construct_link_matrix, tabulate_links
    from ._plotting import heatmap, heatmap_linkmatrix, circle_plot, railroad_plot, sankey
    from ._stats import (t_test_between, group_averages_whole, group_norm_averages_whole,
                        group_norm_whole_list, group_norm_hemi_list,
                        group_averages_hemi, group_norm_averages_hemi, get_average_link_matrix)
    from ._natures import get_natures, _get_natures
    from ._trf import fit_trfs, subset_trfs_by_region, get_trf_statmap

    def __init__(self, experiment, controls, patients, visits, sessions, trials,
                file_patterns="./Results/[{subject}]-[visit={visit}]-[beta].p",
                lobe_mapping=None, subjects_dir=None, exclusions=None,
                minlinkcount=0, maxlinkcount=84*84):
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
    
    @staticmethod
    def _check_hemis(condition):
        allhemis = [('lh', 'lh'), ('rh', 'rh'), ('lh', 'rh'), ('rh', 'lh')]
        newcondition = []
        
        for v in condition:
            if len(v) != 5 or v[4] is None:
                newcondition.append([v[0], v[1], v[2], v[3], allhemis])
            else:
                newcondition.append(v)
        return newcondition