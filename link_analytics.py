

class AnalyzeLinks:
    def __init__(self, experiment, controls, patients, visits, sessions, file_patterns="./Results/[{subject}]-[visit={visit}]-[beta].p",
                 lobe_mapping=None, subjects_dir=None):
        self.experiment = experiment
        self.CONTROLS = controls
        self.PATIENTS = patients
        if isinstance(file_patterns, str):
            self.file_patterns = [file_patterns]
        else:
            self.file_patterns = file_patterns
        self.visits = visits
        self.sessions = sessions
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
        
        self._construct_link_matrix()
        
    def _construct_link_matrix(self):
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
                    link_count[f"{subject};{lobe_src}->{lobe_tar}"] = np.zeros((len(visits), len(sessions), 4))
                link_count[f"{subject};total"] = np.zeros((len(visits), len(sessions), 4))

                for visit_idx, visit in enumerate(self.visits):
                    for session_idx, session in enumerate(self.sessions):
                        for file_pattern in self.file_patterns:
                            file_name = eval(f"f'{file_pattern}'")
                            try:
                                with open(file_name, 'rb') as fp: 
                                    NLGC_obj = pickle.load(fp)
                                
                                self.models.append([file_name, NLGC_obj])

                                J = NLGC_obj.get_J_statistics()
                            except Exception as e:
                                print(f"caught exception: {e}")
                                print(f"continuing with zero link matrix for {subject}, v={visit}, s={session}")
                                self.models.append([file_name, None])
                                J = np.zeros([84, 84])
                            this_visit_link_count = get_num_links(J.copy(), target_lobes, src_target, 
                                                                  lobe_labels, normalization=False)
                            for key, item in this_visit_link_count.items():
                                link_count[f"{subject};{key}"][visit_idx][session_idx] = item    
                                link_count[f"{subject};total"][visit_idx][session_idx] += item
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
                            this_link_count = self.link_count[f"{subject};{key}"][visit_idx][session_idx].sum() 

                            # this_con = np.zeros((len(target_lobes), len(target_lobes)))
                            # ll, rr, lr, rl
                            this_con_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
                            this_con_hemi_raw = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
                            for j, source in enumerate(self.target_lobes):
                                for i, target in enumerate(self.target_lobes):
                                    if self.link_count[f"{subject};total"][visit_idx][session_idx].sum() != 0:
                                        this_con_hemi[:, i, j] = \
                                        self.link_count[f"{subject};{source}->{target}"][visit_idx][session_idx]\
                                        / self.link_count[f"{subject};total"][visit_idx][session_idx].sum()
                                        this_con_hemi_raw[:, i, j] = \
                                        self.link_count[f"{subject};{source}->{target}"][visit_idx][session_idx]

                                conn[f'{subject}{visit}{session}'] = this_con_hemi.sum(axis=0)
                                conn_raw[f'{subject}{visit}{session}'] = this_con_hemi_raw.sum(axis=0)
                                conn_hemi[f'{subject}{visit}{session}'] = this_con_hemi
                                conn_hemi_raw[f'{subject}{visit}{session}'] = this_con_hemi_raw

                            if percentages:
                                if self.link_count[f"{subject};total"][visit_idx][session_idx].sum() != 0:
                                    this_link_count /= self.link_count[f"{subject};total"][visit_idx][session_idx].sum()
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
            if len(v) != 4 or v[3] is None:
                newcondition.append([v[0], v[1], v[2], allhemis])
            else:
                newcondition.append(v)
        return newcondition
                
            
    def t_test_between(self, condition1, condition2, status1='C1', status2='C2', hemi=False):
        # if hemi is False, tests are performed over the whole brain
        # in this case, condition is of the form [(subject, visit, session), (subject, visit, session)]
        #
        # if hemi is True, tests are performed between particular hemispheres
        # in this case, condition is of the form [(subject, visit, session, [(hemi1, hemi2), (hemi1, hemi1)]), 
        # (subject, visit, session, [(hemi2, hemi1)])]
        
        if hemi == True:
            condition1, condition2 = self._check_hemis(condition1), self._check_hemis(condition2)
        
        if self.conn is None:
            self.tabulate_links(verbose=False)
            
        print(f'({status1} vs. {status2})')
        print('\n')
        for j, source in enumerate(self.target_lobes):
            for i, target in enumerate(self.target_lobes):
                a, b = [], []
                if hemi == False:
                    for subject, visit, session in condition1:
                        a.append(self.conn[f'{subject}{visit}{session}'][i,j])
                    for subject, visit, session, in condition2:
                        b.append(self.conn[f'{subject}{visit}{session}'][i,j])
                else:
                    for subject, visit, session, hemilist in condition1:
                        for srchemi, dsthemi in hemilist:
                            hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
                            a.append(self.conn_hemi[f'{subject}{visit}{session}'][hemi_idx,i,j])
                    for subject, visit, session, hemilist in condition2:
                        for srchemi, dsthemi in hemilist:
                            hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
                            b.append(self.conn_hemi[f'{subject}{visit}{session}'][hemi_idx,i,j])
                            
                            
                tval, pval = scipy.stats.ttest_ind(a, b)
                print(f'{source}->{target}: {min(1,len(target_lobes)**2*pval)}')
        print('--------------------------------------')
        print('\n')
        
        
    def _group_averages_whole(self, condition1, condition2):
        c1_group_avg = np.zeros((len(self.target_lobes), len(self.target_lobes)))
        c2_group_avg = np.zeros((len(self.target_lobes), len(self.target_lobes)))
        count = 0
        for subject, visit, session in condition1:
            links = self.conn[f'{subject}{visit}{session}']
            if links.sum() != 0:
                c1_group_avg += links
                count += links.sum()
        c1_group_avg /= count
        
        count = 0
        for subject, visit, session in condition2:
            links = self.conn[f'{subject}{visit}{session}']
            if links.sum() != 0:
                c2_group_avg += links
                count += links.sum()
        c2_group_avg /= count
        return c1_group_avg, c2_group_avg
    
    # TODO fix normalization logic
    def _group_averages_hemi(self, condition1, condition2):
        c1_group_avg_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
        c2_group_avg_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
        c1count = 0
        c2count = 0
        
        ########## c1 ##########
        for subject, visit, session, hemilist in condition1:
            for srchemi, dsthemi in hemilist:
                hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
                links = self.conn_hemi[f'{subject}{visit}{session}'][hemi_idx, :, :]
                if links.sum() != 0:
                    # some links for this hemisphere
                    c1_group_avg_hemi[hemi_idx, :, :] += links
                    # normalize by many links found for all hemispheres
                    c1count +=  self.conn[f'{subject}{visit}{session}'].sum()

        # normalize hemisphere connectomes
        for hemi_idx in range(4):
            if c1count != 0:
                c1_group_avg_hemi[hemi_idx, :, :] /= c1count
                
        ########## c2 ##########
        for subject, visit, session, hemilist in condition2:
            for srchemi, dsthemi in hemilist:
                hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
                links = self.conn_hemi[f'{subject}{visit}{session}'][hemi_idx,:, :]
                c2_group_avg_hemi[hemi_idx, :, :] += links
                c2count += self.conn[f'{subject}{visit}{session}'].sum()

        # normalize hemisphere connectomes
        for hemi_idx in range(4):
            if c2count != 0:
                c2_group_avg_hemi[hemi_idx, :, :] /= c2count
        
        return c1_group_avg_hemi, c2_group_avg_hemi
        
            
    def plot_heatmap(self, condition1, condition2, status1='C1', status2='C2', hemi=False, 
                     overlay_nums=False, vmin=0, vmax=1, diffvmin=-1, diffvmax=1, figsize=(10, 3)):
        if hemi == True:
            condition1, condition2 = self._check_hemis(condition1), self._check_hemis(condition2)
            
        if self.conn is None:
            self.tabulate_links(verbose=False)
    
    
        ###################################
        if hemi == False:
            c1_group_avg, c2_group_avg = self._group_averages_whole(condition1, condition2)
            c1_group_avg = c1_group_avg.T
            c2_group_avg = c2_group_avg.T
            
            fig, ax = plt.subplots(1, 3, figsize=figsize)

            im = ax[0].imshow(c1_group_avg, vmin=vmin, vmax=vmax)
            ax[0].set_title(f"{status1}")
            ax[0].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
            ax[0].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
            if overlay_nums == True:
                for srcidx, source in enumerate(self.target_lobes):
                    for dstidx, target in enumerate(self.target_lobes):
                        ax[0].text(srcidx, dstidx, 
                            round(c1_group_avg[dstidx, srcidx],2), ha="center", va="center", color="w")


            ax[1].imshow(c2_group_avg, vmin=vmin, vmax=vmax)
            ax[1].set_title(f"{status2}")
            ax[1].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
            ax[1].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
            if overlay_nums == True:
                for srcidx, source in enumerate(self.target_lobes):
                    for dstidx, target in enumerate(self.target_lobes):
                        ax[1].text(srcidx, dstidx, 
                            round(c2_group_avg[dstidx, srcidx],2), ha="center", va="center", color="w")


            im2 = ax[2].imshow(c1_group_avg - c2_group_avg, vmin=diffvmin, vmax=diffvmax, cmap='seismic')
            ax[2].set_title(f"{status1}-{status2}")
            ax[2].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
            ax[2].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
            if overlay_nums == True:
                for srcidx, source in enumerate(self.target_lobes):
                    for dstidx, target in enumerate(self.target_lobes):
                        ax[2].text(srcidx, dstidx, 
                            round(c1_group_avg[dstidx, srcidx]\
                                  -c2_group_avg[dstidx, srcidx],2), ha="center", va="center", color="w")
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.0075, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.95, 0.15, 0.0075, 0.7])
            fig.colorbar(im2, cax=cbar_ax)
            plt.show()
    
        ###################################
        else:
            c1_group_avg_hemi, c2_group_avg_hemi = self._group_averages_hemi(condition1, condition2)
            for i in range(4):
                c1_group_avg_hemi[i] = c1_group_avg_hemi[i].T
                c2_group_avg_hemi[i] = c2_group_avg_hemi[i].T
            
            fig, ax = plt.subplots(2, 6, figsize=figsize)

            for srchemi, dsthemi, i, j in [('lh', 'lh', 0, 0), ('rh', 'rh', 1, 1), 
                                           ('lh', 'rh', 0, 1), ('rh', 'lh', 1, 0)]:
                hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
                im = ax[i, j].imshow(c1_group_avg_hemi[hemi_idx], vmin=vmin, vmax=vmax)
                ax[i, j].set_title(f"{status1}:{srchemi}->{dsthemi}")
                ax[i, j].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
                ax[i, j].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
                if overlay_nums == True:
                    for srcidx, source in enumerate(self.target_lobes):
                        for dstidx, target in enumerate(self.target_lobes):
                            ax[i, j].text(srcidx, dstidx, 
                                round(c1_group_avg_hemi[hemi_idx, dstidx, srcidx],2), ha="center", va="center", color="w")

                ax[i, j+2].imshow(c2_group_avg_hemi[hemi_idx], vmin=vmin, vmax=vmax)
                ax[i, j+2].set_title(f"{status2}:{srchemi}->{dsthemi}")
                ax[i, j+2].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
                ax[i, j+2].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
                if overlay_nums == True:
                    for srcidx, source in enumerate(self.target_lobes):
                        for dstidx, target in enumerate(self.target_lobes):
                            ax[i, j+2].text(srcidx, dstidx, 
                                round(c2_group_avg_hemi[hemi_idx, dstidx, srcidx],2), ha="center", va="center", color="w")

                im2 = ax[i, j+4].imshow(c1_group_avg_hemi[hemi_idx] \
                                  - c2_group_avg_hemi[hemi_idx],
                                  vmin=diffvmin, vmax=diffvmax, cmap='seismic')
                ax[i, j+4].set_title(f"{status1}-{status2}:{srchemi}->{dsthemi}")
                ax[i, j+4].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
                ax[i, j+4].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
                if overlay_nums == True:
                    for srcidx, source in enumerate(self.target_lobes):
                        for dstidx, target in enumerate(self.target_lobes):
                            ax[i, j+4].text(srcidx, dstidx, 
                                round(c1_group_avg_hemi[hemi_idx, dstidx, srcidx]\
                                      -c2_group_avg_hemi[hemi_idx, dstidx, srcidx],2), ha="center", va="center", color="w")
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.81, 0.15, 0.005, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.005, 0.7])
            fig.colorbar(im2, cax=cbar_ax)

            plt.show()
            
    def plot_circleplot(self, condition1, condition2, status1='C1', status2='C2', hemi=False):
        if hemi == True:
            condition1, condition2 = self._check_hemis(condition1), self._check_hemis(condition2)
            
        if self.conn is None:
            self.tabulate_links(verbose=False)
    
    
        ###################################
        if hemi == False:
            c1_group_avg, c2_group_avg = self._group_averages_whole(condition1, condition2)
            
            
            fig, ax = plt.subplots(1, 3, figsize=(10, 3))
            for i in range(3):
                ax[i] = plt.subplot(130+(i+1), projection='polar')

            viz.plot_connectivity_circle(c1_group_avg, self.target_lobes, ax=ax[0], show=False)
            ax[0].set_title(f"{status1}")


            viz.plot_connectivity_circle(c2_group_avg, self.target_lobes, ax=ax[1], show=False)
            ax[1].set_title(f"{status2}")

            viz.plot_connectivity_circle(c1_group_avg - c2_group_avg, self.target_lobes, ax=ax[2], show=False)
            ax[2].set_title(f"{status1}-{status2}")
            plt.show()
    
        ###################################
        else:
            c1_group_avg_hemi, c2_group_avg_hemi = self._group_averages_hemi(condition1, condition2)
                
            
            fig, ax = plt.subplots(2, 6, figsize=(20, 7))
            for row in range(2):
                for col in range(6):
                    ax[row, col] = plt.subplot(2,6, 0+(6*row+col+1), projection='polar')

            for srchemi, dsthemi, i, j in [('lh', 'lh', 0, 0), ('rh', 'rh', 1, 1), 
                                           ('lh', 'rh', 0, 1), ('rh', 'lh', 1, 0)]:
                hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
                viz.plot_connectivity_circle(c1_group_avg_hemi[hemi_idx], self.target_lobes, ax=ax[i, j], show=False)
                ax[i, j].set_title(f"{status1}:{srchemi}->{dsthemi}")

                viz.plot_connectivity_circle(c2_group_avg_hemi[hemi_idx], self.target_lobes, ax=ax[i, j+2], show=False)
                ax[i, j+2].set_title(f"{status2}:{srchemi}->{dsthemi}")
                
                
                viz.plot_connectivity_circle(c1_group_avg_hemi[hemi_idx] \
                                  - c2_group_avg_hemi[hemi_idx], self.target_lobes, ax=ax[i, j+4], show=False)
                ax[i, j+4].set_title(f"{status1}-{status2}:{srchemi}->{dsthemi}")

            plt.show()
            
            
    def get_average_link_matrix(self, condition1, condition2, hemi=False):
        if hemi == True:
            condition1, condition2 = self._check_hemis(condition1), self._check_hemis(condition2)
            
        if self.conn is None:
            self.tabulate_links(verbose=False)
    
    
        ###################################
        if hemi == False:
            c1_group_avg, c2_group_avg = self._group_averages_whole(condition1, condition2)
            return c1_group_avg, c2_group_avg
        
        ###################################
        else:
            c1_group_avg_hemi, c2_group_avg_hemi  = self._group_averages_hemi(condition1, condition2)
            return c1_group_avg_hemi, c2_group_avg_hemi
            
    @staticmethod
    def _get_vertices(n, rad, offset, centerxy=(0.5, 0.5)):
        # generate n vertices evenly spaced along a circle with radius rad
        # offset by offset degrees
        increment = 360 / n
        xs = []
        ys = []
        thetas = []
        for i in range(n):
            theta = i*increment
            thetas.append(theta)
            x = np.round(rad*np.cos(theta*np.pi/180), 2)
            y = np.round(rad*np.sin(theta*np.pi/180), 2)
            xs.append(x+centerxy[0])
            ys.append(y+centerxy[1])
        return xs, ys, [rad]*len(xs), thetas

    # @staticmethod
    # def hist_midpoint(ts):
        # helper function for trace_bezier_midpoint
    #     halfdiff = (np.max(list(ts.keys())) - np.min(list(ts.keys())))/2
    #     midpt =  np.min(list(ts.keys())) + halfdiff
    #     idx = np.argmin(np.abs(np.array(list(ts.keys())) - midpt))
    #     return ts[np.array(list(ts.keys()))[idx]]

    @staticmethod 
    def _trace_bezier_midpoint(arrow):
        path = arrow.get_path().vertices

        ## used the method below to find the optimal bezier t, but it is finnicky,
        ## so i'll use the best t it found and leave the commented code for future runs
        #####################################################
    #     xs = {}
    #     ys = {}
    #     for t in np.linspace(0.0, 1, 300):
    #         i, j = mpl.bezier.BezierSegment(path).point_at_t(t)
    #         xs[i] = t
    #         ys[i] = t
            
    #     txfinal = hist_midpoint(xs)
    #     tyfinal = hist_midpoint(ys)
    #     tfinal = np.mean([txfinal, tyfinal])
    #     print(txfinal, tyfinal)
        #####################################################0.7357859531772575
        return *mpl.bezier.BezierSegment(path).point_at_t(0.72), \
                *mpl.bezier.BezierSegment(path).point_at_t(0.7357859531772575)

    @staticmethod
    def _push_wedge(factor, wx, wy, centerxy=(0.5, 0.5)):
        cx, cy = centerxy
        # radially push the wedge outwards from center
        
        # center the coords at origin     
        wx = wx - cx
        wy = wy - cy
        wx = factor*wx + cx
        wy = factor*wy + cy
        return wx, wy
            

    def _add_connection(self, ax, posa, posb, bidirectional=False, inner=True, arc=0.2, 
                    pushfactor={'inner':0.75, 'outer':1.15}, centerxy=(0.5, 0.5),
                    color={'inner':1, 'outer':1}):
        inner = 2*inner-1
        style = "Simple, tail_width=6, head_width=4, head_length=8"
        kw = dict(arrowstyle=style)


        arrow = mpatches.FancyArrowPatch(posa, posb,
                                    connectionstyle=f"arc3,rad={-int(inner)*arc}", color=color['inner'], **kw)

        ax.add_patch(arrow)
        
        
        mid_dx, mid_dy, midx, midy = self._trace_bezier_midpoint(arrow)
        mid_dx, mid_dy = self._push_wedge(pushfactor['inner'], mid_dx, mid_dy, centerxy)
        midx, midy = self._push_wedge(pushfactor['inner'], midx, midy, centerxy)

        ax.arrow(midx, midy, 0.001*(mid_dx-midx), 0.001*(mid_dy-midy), width=0.01, color=color['inner'])
        
        if bidirectional:
            arrow = mpatches.FancyArrowPatch(posa, posb,
                                    connectionstyle=f"arc3,rad={-(-int(inner))*arc}", color=color['outer'], **kw)
            ax.add_patch(arrow)
            mid_dx, mid_dy, midx, midy = self._trace_bezier_midpoint(arrow)
            mid_dx, mid_dy = self._push_wedge(pushfactor['outer'], mid_dx, mid_dy, centerxy)
            midx, midy = self._push_wedge(pushfactor['outer'], midx, midy, centerxy)
            ax.arrow(midx, midy, 0.001*(midx-mid_dx), 0.001*(midy-mid_dy), width=0.01, color=color['outer'])
            

    @staticmethod
    def _draw_selfloop_arrow(ax, centX, centY, radius, angle_, theta2_, color_):
        from numpy import radians as rad
        #========Line
        arc = mpatches.Arc([centX,centY],radius,radius,angle=angle_,
            theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=7, color=color_)
        ax.add_patch(arc)


        #========Create the arrow head
        endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
        endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))
        
        head = mpatches.RegularPolygon(
                (endX, endY),            # (x,y)
                3,                       # number of vertices
                radius=radius/8,                # radius
                orientation=rad(angle_+theta2_),     # orientation
                color=color_)
        

        ax.add_patch(head)
        ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
        # Make sure you keep the axes scaled or else arrow will distort

    def railroad_plot(self, lkm, rad=0.3, centerxy=(0.5, 0.5), nverts=3, region=['P','F', 'T', 'O', 'S'], 
                    colors=['tab:blue', 'tab:green', 'tab:red', 'purple', 'pink'],
                    pushfactor={'inner':0.715, 'outer':1.147}, figsize=(10, 10), cmap=mpl.colormaps['binary'],
                    ax_=None):
        # create a railroad plot from a link matrix. 
        # the link matrix should have normalized values, i.e., each cell should be a percentage in [0, 1].
        
        # links from behrad go row:dst col:src but we need row:src col:dst.
        lkm = lkm.T
        
        if ax_ is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            ax = ax_

        xs, ys, _, _ = self._get_vertices(nverts, rad, 0, centerxy)
        xs_selfloop, ys_selfloop, _, thetas_selfloop = self._get_vertices(nverts, rad+0.1, 0, centerxy)

        for i in range(nverts):
            a = lkm[i, (i+1)%nverts]
            b = lkm[(i+1)%nverts, i]
            assert(a <= 1 and a >= 0 and b <= 1 and b >= 0)
            if a == 1:
                a = a - 0.00001
            if b == 1:
                b = b - 0.00001

            color_i = cmap(a)
            color_o = cmap(b)
            self._add_connection(ax, (xs[i], ys[i]), (xs[(i+1)%nverts], ys[(i+1)%nverts]), 
                        bidirectional=True, arc=0.25, pushfactor={'inner':0.715, 'outer':1.147}, 
                        color={'inner':color_i, 'outer':color_o})


        for i in range(nverts):
            c = lkm[i, i]
            assert(c <= 1 and c >= 0)
            if c == 1:
                c = c - 0.00001
            color_self = cmap(c)
            self._draw_selfloop_arrow(ax, xs_selfloop[i], ys_selfloop[i], 0.15, -180+thetas_selfloop[i], 315, 
                color_=color_self)
        
        for idx, (i, j) in enumerate(zip(xs, ys)):
            circle = mpatches.Circle((i, j), 0.05, linewidth=3, alpha=1, color=colors[idx])
            ax.add_patch(circle)
            
            if ax_ is None:
                plt.text(i, j, region[idx], size=35,
                    ha="center", va='center_baseline', color='white', fontname='sans-serif', weight='heavy')
            else:
                ax.text(i, j, region[idx], size=35,
                    ha="center", va='center_baseline', color='white', fontname='sans-serif', weight='heavy')

        if ax_ is None:
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
        return ax
    
    @staticmethod
    def _tally_natures(model, natures):
        links = np.where(model.get_J_statistics() > 0.95)
        A = model._model_f[0]._parameters[0]
        neigs = A.shape[1]//84
        order = A.shape[0]

        # only support order 2 models for now. this is 
        # because 2 ar coefficients have a nice frequency-domain
        # interpretation (i.e., 2-tap AR filter -- just highpass 
        # or lowpass) whereas more than 2 coefficients gives rise 
        # to more complicated filter structures (e.g., different 
        # kinds of bandpass filters) that complicate the macro-level
        # interpretation
        assert(order == 2) 

        aes = dict()
        for eigno in range(neigs):
            for orderno in range(order):
                aes[f'a{orderno}e{eigno}'] = A[orderno][eigno::neigs, eigno::neigs] * ~np.eye(84, dtype=bool)


        for idx, i in enumerate(links[0]):
            j = links[1][idx]

            for eigno in range(neigs):
                a_s = []
                for orderno in range(order):
                    a_s.append(aes[f"a{orderno}e{eigno}"][i, j])
                if all(np.abs(a_s) > 0.01):
                    nature_sign = np.sign(a_s[0]*a_s[1])
                    if nature_sign > 0 and a_s[0] > 0:
                        natures['facilitative'] += 1
                    elif nature_sign > 0 and a_s[0] < 0:
                        natures['suppressive'] += 1
                    else:
                        natures['sharpening'] += 1

                else:
                    natures['none'] += 1
        return natures
        
    
    def nature_of_links(self, condition):
        natures = {
            'facilitative': 0,
            'suppressive': 0,
            'sharpening': 0,
            'none': 0,
        }
        
        for subject, visit, session in condition:
            for modelname, model in self.models:
                for fp in self.file_patterns:
                    if modelname != eval(f"f'{file_pattern}'"):
                        continue
                    natures = self._tally_natures(model, natures)
                
        return natures