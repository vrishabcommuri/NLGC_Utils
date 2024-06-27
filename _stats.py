import numpy as np
import scipy
from ._roi import partition, unpartition
import pandas as pd

# def t_test_between(self, condition1, condition2, status1='C1', status2='C2', hemi=False):
#     # if hemi is False, tests are performed over the whole brain
#     # in this case, condition is of the form [(subject, visit, session, trial), (subject, visit, session, trial)]
#     #
#     # if hemi is True, tests are performed between particular hemispheres
#     # in this case, condition is of the form [(subject, visit, session, trial, [(hemi1, hemi2), (hemi1, hemi1)]), 
#     # (subject, visit, session, trial, [(hemi2, hemi1)])]
    
#     if hemi == True:
#         condition1, condition2 = self._check_hemis(condition1), self._check_hemis(condition2)
        
#     print(f'({status1} vs. {status2})')
#     print('\n')
#     for j, source in enumerate(self.target_lobes):
#         for i, target in enumerate(self.target_lobes):
#             a, b = [], []
#             if hemi == False:
#                 for subject, visit, session, trial in condition1:
#                     a.append(self.conn[f'{subject}{visit}{session}{trial}'][i,j])
#                 for subject, visit, session, trial in condition2:
#                     b.append(self.conn[f'{subject}{visit}{session}{trial}'][i,j])
#             else:
#                 for subject, visit, session, trial, hemilist in condition1:
#                     for srchemi, dsthemi in hemilist:
#                         hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
#                         a.append(self.conn_hemi[f'{subject}{visit}{session}{trial}'][hemi_idx,i,j])
#                 for subject, visit, session, trial, hemilist in condition2:
#                     for srchemi, dsthemi in hemilist:
#                         hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
#                         b.append(self.conn_hemi[f'{subject}{visit}{session}{trial}'][hemi_idx,i,j])
                        
                        
#             tval, pval = scipy.stats.ttest_ind(a, b)
#             print(f'{source}->{target}: {min(1,len(self.target_lobes)**2*pval)}')
#     print('--------------------------------------')
#     print('\n')


def _condition_average(self, condition, colnorm=True, matnorm=False):
    condition_matrices = []

    df_cond = pd.DataFrame(condition).astype(str)
    df_cond.columns = ['subject', 'visit', 'session', 'trial']
    
    # trials get averaged before we normalize columns
    for name, group in df_cond.groupby(["subject", "visit", "session"]):
        # join here concatenates column values (sub, visit, sess, trial) into one string for lookup
        grouped_matrices = group.apply(lambda x: self.conn_hemi[''.join(x.values)], axis=1)

        # formatting
        grouped_matrices = np.array([unpartition(i).astype(float) for i in grouped_matrices])

        if self.force_nonzero_trials:
            # all trials must have nonzero J statistics (i.e., non-NaN) else output will be NaN
            avg_mat = np.mean(grouped_matrices, axis=0)  
        else:
            # allow zero J statistic trials; convert all NaN trials to zeros 
            # (but don't include zeros in average)
            avg_mat = np.nanmean(grouped_matrices, axis=0)  
        
        # column norm
        if colnorm:
            avg_mat /= np.sum(avg_mat, axis=0)
        if matnorm:
            avg_mat /= np.sum(avg_mat)

        condition_matrices.append(avg_mat)

    condition_avg = np.nanmean(condition_matrices, axis=0)
    return partition(condition_avg)

def _hemi_average(self, condition1, condition2, colnorm, matnorm):
    c1_avg = self._condition_average(condition1, colnorm=colnorm, matnorm=matnorm)
    c2_avg = self._condition_average(condition2, colnorm=colnorm, matnorm=matnorm)

    # convert NaNs to zeros
    c1_avg = partition(np.nan_to_num(unpartition(c1_avg)))
    c2_avg = partition(np.nan_to_num(unpartition(c2_avg)))

    return c1_avg, c2_avg

def group_averages_hemi(self, condition1, condition2):
    c1_avg, c2_avg = self._hemi_average(condition1, condition2, False, True)
    return c1_avg, c2_avg

def group_norm_averages_hemi(self, condition1, condition2):
    c1_avg, c2_avg = self._hemi_average(condition1, condition2, True, True)
    return c1_avg, c2_avg

def group_averages_whole(self, condition1, condition2):
    c1_avg, c2_avg = self.group_averages_hemi(condition1, condition2)
    return np.mean(c1_avg, axis=0), np.mean(c2_avg, axis=0)


def group_norm_averages_whole(self, condition1, condition2):
    c1_avg, c2_avg = self.group_norm_averages_hemi(condition1, condition2)
    return np.mean(c1_avg, axis=0), np.mean(c2_avg, axis=0)

# def group_whole_list(self, condition1, condition2, norm=False):
#     c1_group_avg = []
#     c2_group_avg = []
#     for subject, visit, session, trial in condition1:
#         if norm:
#             links = self.conn[f'{subject}{visit}{session}{trial}']
#         else:
#             links = self.conn_raw[f'{subject}{visit}{session}{trial}']
#         c1_group_avg.append(links)
    
#     count = 0
#     for subject, visit, session, trial in condition2:
#         if norm:
#             links = self.conn[f'{subject}{visit}{session}{trial}']
#         else:
#             links = self.conn_raw[f'{subject}{visit}{session}{trial}']
#         c2_group_avg.append(links)
        
#     return c1_group_avg, c2_group_avg
    

# def group_averages_hemi(self, condition1, condition2):
#     c1_group_avg_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
#     c2_group_avg_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
#     c1count = 0
#     c2count = 0
    
#     ########## c1 ##########
#     for subject, visit, session, trial, hemilist in condition1:
#         for srchemi, dsthemi in hemilist:
#             hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
#             links = self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'][hemi_idx, :, :]
#             # links for this hemisphere
#             c1_group_avg_hemi[hemi_idx, :, :] += links
        
#         # normalize by how many links found for all hemispheres
#         c1count += self.conn_raw[f'{subject}{visit}{session}{trial}'].sum()

#     # normalize hemisphere connectomes
#     for hemi_idx in range(4):
#         if c1count != 0:
#             c1_group_avg_hemi[hemi_idx, :, :] /= c1count
            
#     ########## c2 ##########
#     for subject, visit, session, trial, hemilist in condition2:
#         for srchemi, dsthemi in hemilist:
#             hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
#             links = self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'][hemi_idx,:, :]
#             c2_group_avg_hemi[hemi_idx, :, :] += links
        
#         c2count += self.conn_raw[f'{subject}{visit}{session}{trial}'].sum()

#     # normalize hemisphere connectomes
#     for hemi_idx in range(4):
#         if c2count != 0:
#             c2_group_avg_hemi[hemi_idx, :, :] /= c2count
    
#     return c1_group_avg_hemi, c2_group_avg_hemi


# def group_norm_averages_hemi(self, condition1, condition2):
#     c1_group_avg_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
#     c2_group_avg_hemi = np.zeros((4, len(self.target_lobes), len(self.target_lobes)))
#     c1count = 0
#     c2count = 0
    
#     ########## c1 ##########
#     for subject, visit, session, trial, hemilist in condition1:
#         for srchemi, dsthemi in hemilist:
#             hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
#             links_normed = self.conn_hemi[f'{subject}{visit}{session}{trial}'][hemi_idx, :, :]
#             # links for this hemisphere
#             c1_group_avg_hemi[hemi_idx, :, :] += links_normed
        
#         # normalize by how many links found for all hemispheres
#         c1count += 1

#     # normalize hemisphere connectomes
#     for hemi_idx in range(4):
#         if c1count != 0:
#             c1_group_avg_hemi[hemi_idx, :, :] /= c1count
            
#     ########## c2 ##########
#     for subject, visit, session, trial, hemilist in condition2:
#         for srchemi, dsthemi in hemilist:
#             hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
#             links_normed = self.conn_hemi[f'{subject}{visit}{session}{trial}'][hemi_idx,:, :]
#             c2_group_avg_hemi[hemi_idx, :, :] += links_normed
        
#         c2count += 1

#     # normalize hemisphere connectomes
#     for hemi_idx in range(4):
#         if c2count != 0:
#             c2_group_avg_hemi[hemi_idx, :, :] /= c2count
    
#     return c1_group_avg_hemi, c2_group_avg_hemi


# def group_hemi_list(self, condition1, condition2, norm=False):
#     c1_group_avg_hemi = []
#     c2_group_avg_hemi = []
    
#     ########## c1 ##########
#     for subject, visit, session, trial, hemilist in condition1:
#         for srchemi, dsthemi in hemilist:
#             hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
#             if norm:
#                 links = self.conn_hemi[f'{subject}{visit}{session}{trial}'][hemi_idx, :, :]
#             else:
#                 links = self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'][hemi_idx, :, :]
#             # links for this hemisphere
#             c1_group_avg_hemi.append(links)
    
            
#     ########## c2 ##########
#     for subject, visit, session, trial, hemilist in condition2:
#         for srchemi, dsthemi in hemilist:
#             hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
#             if norm:
#                 links = self.conn_hemi[f'{subject}{visit}{session}{trial}'][hemi_idx, :, :]
#             else:
#                 links = self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'][hemi_idx, :, :]
#             c2_group_avg_hemi.append(links)
        
#     return c1_group_avg_hemi, c2_group_avg_hemi


# def get_average_link_matrix(self, condition1, condition2, hemi=False):
#     if hemi == True:
#         condition1, condition2 = self._check_hemis(condition1), self._check_hemis(condition2)

#     ###################################
#     if hemi == False:
#         c1_group_avg, c2_group_avg = self.group_averages_whole(condition1, condition2)
#         return c1_group_avg, c2_group_avg
    
#     ###################################
#     else:
#         c1_group_avg_hemi, c2_group_avg_hemi = self.group_averages_hemi(condition1, condition2)
#         return c1_group_avg_hemi, c2_group_avg_hemi


def condition_to_models(self, condition, norm=False):
    matrices = []
    for subject, visit, session, trial in condition:
        for name, key, model in self.models:
            if model is None:
                continue
        
            if key == f"{subject}{visit}{session}{trial}":
                matrices.append(model.get_J_statistics())
    if norm:
        matrices = [i/i.sum() for i in matrices]
    return matrices

