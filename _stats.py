import numpy as np
import scipy
from ._roi import partition, unpartition
import pandas as pd

def t_test_between(self, condition1, condition2, status1='C1', status2='C2'):
    print(f'({status1} vs. {status2})\n')
    print("bonferroni-corrected t-test p values")
    for j, source in enumerate(self.target_lobes):
        for i, dest in enumerate(self.target_lobes):
            for shemi_idx, srchemi in enumerate(['lh', 'rh']):
                for dhemi_idx, dsthemi in enumerate(['lh', 'rh']):
                    hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
                    a, b = [], []
                    for subject, visit, session, trial in condition1:
                        a.append(self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'][hemi_idx,i,j])
                    for subject, visit, session, trial in condition2:
                        b.append(self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'][hemi_idx,i,j])
                        
                        
                    tval, pval = scipy.stats.ttest_ind(a, b)
                    # bonferroni-corrected pval
                    print(f'{srchemi}-{source} -> {dsthemi}-{dest}: {min(1,(2*len(self.target_lobes))**2 * pval)}')
    print('--------------------------------------')
    print('\n')


def _condition_average(self, condition, areanorm=False, matnorm=False, trim=0.0, verbose=0):
    condition_matrices = []
    count = 0
    
    for subject, visit, session, trial in condition:
        if areanorm:
            cm = np.copy(self.conn_hemi[f'{subject}{visit}{session}{trial}'])
        else:
            cm = np.copy(self.conn_hemi_raw[f'{subject}{visit}{session}{trial}'])

        if matnorm:
            cm /= np.sum(cm)

        if not np.any(np.isnan(cm)):
            condition_matrices.append(cm)

        count += 1
        if verbose > 1:
            print("averaging", subject, visit, session, trial)
    
    if verbose > 0:
        print(f"averaged {count}/{len(condition)} trials")
    
    return scipy.stats.trim_mean(condition_matrices, trim, axis=0)


def group_hemi_average(self, condition1, condition2, areanorm=False, matnorm=False, trim=0.0, verbose=0):
    c1_avg = self._condition_average(condition1, areanorm=areanorm, matnorm=matnorm, trim=trim, verbose=verbose)
    c2_avg = self._condition_average(condition2, areanorm=areanorm, matnorm=matnorm, trim=trim, verbose=verbose)

    return c1_avg, c2_avg


def group_whole_average(self, condition1, condition2, areanorm=False, matnorm=False, trim=0.0, verbose=0):
    c1_avg, c2_avg = self.group_hemi_average(condition1, condition2, areanorm, matnorm, trim=trim, verbose=verbose)

    # TODO: should this average hemis first, then trim?
    return np.mean(c1_avg, axis=0), np.mean(c2_avg, axis=0)


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

