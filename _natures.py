import numpy as np 

def blank_natures_partition(link_natures, partition):
    if partition == ('lh', 'lh'):
        link_natures[:, 42:, ...] = 0
        link_natures[42:, :, ...] = 0
    elif partition == ('rh', 'rh'):
        link_natures[:, :42, ...] = 0
        link_natures[:42, :, ...] = 0
    elif partition == ('rh', 'lh'):
        link_natures[:, 42:, ...] = 0
        link_natures[:42, :, ...] = 0
    elif partition == ('lh', 'rh'):
        link_natures[:, :42, ...] = 0
        link_natures[42:, :, ...] = 0
    else:
        raise Exception(f"invalid hemisphere partition {partition}")

    return link_natures

def _get_natures(model, factor, partition=None):
    a1s = []
    a2s = []
    b1s = []
    b2s = []

    A = model._model_f[0]._parameters[0]
    neigs = A.shape[1]//84
    order = A.shape[0]
    assert(order == 2) 

    link_natures = np.zeros((84, 84, neigs, neigs, 2))
    for ej in range(neigs):
        for ei in range(neigs):
            for orderno in range(order):
                link_natures[..., ei, ej, orderno] = A[orderno][ei::neigs, ej::neigs]  
                if not partition is None:
                    link_natures = blank_natures_partition(link_natures, partition)

    for ej in range(neigs):
        for ei in range(neigs):
            row, col = np.where(model.get_J_statistics() >= factor)

            # if we want to look at all link natures, not just significant GC links,
            # we have to exclude the B coefficients (diagonal of A matrix)
            if factor == 0.0:
                row, col = np.where(~np.eye(84, 84, dtype=bool))
            a1s.extend((link_natures[..., ei, ej, 0])[row, col])
            a2s.extend((link_natures[..., ei, ej, 1])[row, col])

            # B coefficients are on diagonal of A matrix
            if factor == 0.0:
                row, _ = np.where(np.eye(84, 84, dtype=bool))
            b1s.extend((link_natures[..., ei, ej, 0])[row, row]) # get coeffs for active sources
            b2s.extend((link_natures[..., ei, ej, 1])[row, row])
    
    return a1s, a2s, b1s, b2s


def get_natures(self, condition, hemi=False, factor=0.95, verbose=False):
    a1s = []
    a2s = []
    b1s = []
    b2s = []
    
    #####################
    if hemi == False:
        for subject, visit, session, trial in condition:
            for idx, (name, key, model) in enumerate(self.models):
                if key != f"{subject}{visit}{session}{trial}":
                    continue
                if verbose:
                    print(name, key)

                if model is None:
                    if verbose:
                        print(f"model {name} does not exist")
                    continue

                a1s_, a2s_, b1s_, b2s_ = _get_natures(model, factor)
                
                a1s.extend(a1s_)
                a2s.extend(a2s_)
                b1s.extend(b1s_)
                b2s.extend(b2s_)

    #####################
    else:
        condition = self._check_hemis(condition)
        for subject, visit, session, trial, hemilist in condition:
            for partition in hemilist:
                for idx, (name, key, model) in enumerate(self.models):
                    if key != f"{subject}{visit}{session}{trial}":
                        continue
                    if verbose:
                        print(name, key)

                    if model is None:
                        if verbose:
                            print(f"model {name} does not exist")
                        continue

                    a1s_, a2s_, b1s_, b2s_ = _get_natures(model, factor, partition)
                    
                    a1s.extend(a1s_)
                    a2s.extend(a2s_)
                    b1s.extend(b1s_)
                    b2s.extend(b2s_)

    return a1s, a2s, b1s, b2s