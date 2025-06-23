'''
functions to model cell-cell communication at the sub-cluster level
option to include downstream target genes for refined CCI rediction
'''

import numpy as np
import pandas as pd
from numba import jit

############################################################################
#
#### basic functions to model CCI
#
############################################################################

@jit(nopython=True)
def mass_act(R, L, K=0.5):
    return (L*R)/(K + L*R)

@jit(nopython=True)
def alpha(R, L):
    if L==0. or R==0.:
        return 0
    else:
        return np.exp(-1 / (L * R))

@jit(nopython=True)
def beta(T_act):
    if T_act==0.:
        return 0
    else:
        return np.exp(-1/T_act)

@jit(nopython=True)
def beta_RNAVel(T_act):
    if T_act<=0.:
        return 0
    else:
        return np.exp(-1/T_act)

@jit(nopython=True)
def gamma(T_inh):
    return np.exp(-T_inh)

@jit(nopython=True)
def K1(L, R, T_act):
    if (L==0. and T_act==0.) or (R==0. and T_act==0.):
        return 0.
    else:
        return alpha(L, R)/( alpha(L, R) + beta(T_act) )

@jit(nopython=True)
def K1_RNAVel(L, R, V_act):
    # this line is not correct anymore for RNA velocity case:
    if (L==0. and V_act==0.) or (R==0. and V_act==0.):
        return 0.
    else:
        return alpha(L, R)/( alpha(L, R) + beta_RNAVel(V_act) )

@jit(nopython=True)
def K2(L, R, T_inh):
    return alpha(L, R)/( alpha(L, R) + gamma(T_inh) )

############################################################################
#
#### functions to model individual cell CCI with/without the target genes
#
############################################################################
@jit(nopython=True)
def P_pair_LR(R, L, model='mass_action'):
    if model=='mass_action':
        return mass_act(R, L)
    else:
        return alpha(R, L)

@jit(nopython=True)
def P_pair_LRU(R, L, T_act, model='mass_action'):
    # return alpha(R, L)*K1(L, R, T_act)*beta(T_act)
    if model=='mass_action':
        return mass_act(R, L)*beta(T_act)
    else:
        return alpha(R, L)*beta(T_act)

@jit(nopython=True)
def P_pair_LRU_RNAVel(R, L, V_act):
    return alpha(L, R)*K1_RNAVel(L, R, V_act)*beta_RNAVel(V_act)

@jit(nopython=True)
def P_pair_LRD(R, L, T_inh):
    return alpha(L, R)*K2(L, R, T_inh)*gamma(T_inh)

@jit(nopython=True)
def P_pair_all(R, L, T_act, T_inh):
    return alpha(L, R)*K1(L, R, T_act)*beta(T_act)*K2(L, R, T_inh)*gamma(T_inh)


def sign_combs(labels, combs, lig, rec, t_act=None, include_target=False, min_cells=5, model='mass_action', normalize=True):
    ccc_mat = np.zeros((len(combs), len(combs)))
    for i in range(len(combs)):
        lig_sel = lig[labels == combs[i]]
        for j in range(len(combs)):
            rec_sel = rec[labels == combs[j]]
            if include_target:
                tar_sel = t_act[labels == combs[j]]
            if lig_sel.size<min_cells or rec_sel.size<min_cells:
                continue
            else:
                l, r = np.mean(lig_sel), np.mean(rec_sel)
                if include_target:
                    t = np.mean(tar_sel)
                    ccc_mat[i][j] = P_pair_LRU(r, l, t, model=model)
                else:
                    ccc_mat[i][j] = P_pair_LR(r, l, model=model)
    if normalize:
        ccc_mat = ccc_mat/np.sum(ccc_mat)
    return ccc_mat

'''
to include the possibility for RNA velocity instead of targets, remove the include_target argument and instead define:
downstream (list): None (default), target, velocity
it will need some checks (velocities exist...) to avoid errors 
'''
def compute_ccc_matrix(adata, pathway, key='clusters', include_target=False, min_cells=5, model='mass_action', conversion=True, moments=False):
    assert model=='mass_action' or model=='diffusion', "Invalid model parameter, choose between model=='mass action' or model=='diffusion'"
    # generate list of cell labels (state + mode)
    # some state-mode combination might not exist, but still need to be defined to have a well-defined matrix
    states = list(adata.obs[key])
    modes = list(adata.obs[pathway + '_modes'])
    labels = np.asarray([states[i] + '-' + str(modes[i]) for i in range(len(states))])

    # define all state-mode combinations:
    combs = []
    for s in sorted(set(states)):
        for m in sorted(set(modes)):
            combs.append(s + '-' + str(m))

    lig, rec = np.asarray(adata.obs[pathway + '_lig']), np.asarray(adata.obs[pathway + '_rec'])
    if conversion:
        lig, rec = 10 ** lig, 10 ** rec
    lig, rec = lig / np.amax(lig), rec / np.amax(rec)

    if include_target:
        # compute the target array
        tar_list = adata.uns['TF'][pathway]
        if moments:
            t_act = adata[:, tar_list].layers['Mu'].mean(axis=1)+adata[:, tar_list].layers['Ms'].mean(axis=1)
        else:
            t_act = adata[:, tar_list].X.toarray().mean(axis=1)
        if pathway + 'tar' not in adata.obs.keys():
            adata.obs[pathway + '_tar'] = np.asarray(t_act)
        if conversion:
            t_act = 10 ** t_act
        t_act = t_act / np.amax(t_act)

        ccc_mat = sign_combs(labels, combs, lig, rec, t_act=t_act, min_cells=min_cells, include_target=include_target,
                             model=model, normalize=True)
    else:
        ccc_mat = sign_combs(labels, combs, lig, rec, t_act=None, min_cells=min_cells, include_target=include_target,
                             model=model, normalize=True)

    if 'ccc_mat' not in adata.uns.keys():
        adata.uns['ccc_mat'] = {}
    adata.uns['ccc_mat'][pathway] = {'states': combs, 'mat': ccc_mat}

    flat = np.ndarray.flatten(ccc_mat)

# run the ccc matrix calculation using only the cell state annotations
def compute_ccc_matrix_CellchatBenchmark(adata, pathway, key='clusters', include_target=False, min_cells=5, model='mass_action', conversion=True, moments=False):
    assert model == 'mass_action' or model == 'diffusion', "Invalid model parameter, choose between model=='mass action' or model=='diffusion'"
    # generate list of cell labels (state + mode)
    # some state-mode combination might not exist, but still need to be defined to have a well-defined matrix

    labels = np.asarray(adata.obs[key])
    combs = sorted(list(set(labels)))

    lig, rec = np.asarray(adata.obs[pathway + '_lig']), np.asarray(adata.obs[pathway + '_rec'])
    if conversion:
        lig, rec = 10 ** lig, 10 ** rec
    lig, rec = lig / np.amax(lig), rec / np.amax(rec)

    if include_target:
        # compute the target array
        tar_list = adata.uns['TF'][pathway]
        if moments:
            t_act = adata[:, tar_list].layers['Mu'].mean(axis=1) + adata[:, tar_list].layers['Ms'].mean(axis=1)
        else:
            t_act = adata[:, tar_list].X.toarray().mean(axis=1)
        if pathway + 'tar' not in adata.obs.keys():
            adata.obs[pathway + '_tar'] = np.asarray(t_act)
        if conversion:
            t_act = 10 ** t_act
        t_act = t_act / np.amax(t_act)

        ccc_mat = sign_combs(labels, combs, lig, rec, t_act=t_act, min_cells=min_cells, include_target=include_target,
                             model=model, normalize=True)
    else:
        ccc_mat = sign_combs(labels, combs, lig, rec, t_act=None, min_cells=min_cells, include_target=include_target,
                             model=model, normalize=True)

    return combs, ccc_mat


def pathway_strength(adata, pathway, key='clusters'):
    combs = adata.uns['ccc_mat'][pathway]['states']
    ccc_mat = adata.uns['ccc_mat'][pathway]['mat']
    outward, inward = np.zeros(len(combs)), np.zeros(len(combs))
    for i in range(len(combs)):
        outward[i] =  np.sum(ccc_mat[i])
        inward[i] = np.sum(ccc_mat[:, i])

    n_modes = len(set(adata.obs[pathway + '_modes']))
    n_states = int(len(combs)/n_modes)
    incoming, outgoing = np.zeros(n_states), np.zeros(n_states)
    for i in range(n_states):
        incoming[i] = np.sum(inward[i*n_modes:(i+1)*n_modes])
        outgoing[i] = np.sum(outward[i * n_modes:(i + 1) * n_modes])

    df = pd.DataFrame(data=[incoming, outgoing], index=['incoming', 'outgoing'],
                      columns=sorted(list(set(adata.obs[key])))).transpose()

    if 'sign_strength' not in adata.uns.keys():
        adata.uns['sign_strength'] = {}
    adata.uns['sign_strength'][pathway] = df



'''

older version of the functions can be deleted eventually

'''
#
# @jit(nopython=True)
# def singlecell_prob_LR(R, L, model='mass_action'):
#     P = np.zeros((R.size, R.size))
#     for i in range(R.size):
#         for j in range(R.size):
#             P[i][j] = P_pair_LR(R[j], L[i], model=model)
#     P = P/np.sum(P)
#     return P
#
# @jit(nopython=True)
# def singlecell_prob_LRU(R, L, T_act, model='mass_action'):
#     P = np.zeros((R.size, R.size))
#     for i in range(R.size):
#         for j in range(R.size):
#             P[i][j] = P_pair_LRU(R[j], L[i], T_act[j], model=model)
#     P = P/np.sum(P)
#     return P
#
# def singlecell_prob_LRU_RNAVel(R, L, V_act):
#     P = np.zeros((R.size, R.size))
#     for i in range(R.size):
#         for j in range(R.size):
#             P[i][j] = P_pair_LRU_RNAVel(R[j], L[i], V_act[j])
#     P = P/np.sum(P)
#     return P
#
# @jit(nopython=True)
# def singlecell_prob_LRD(R, L, T_inh):
#     P = np.zeros((R.size, R.size))
#     for i in range(R.size):
#         for j in range(R.size):
#             P[i][j] = P_pair_LRD(R[j], L[i], T_inh[j])
#     P = P/np.sum(P)
#     return P
#
# @jit(nopython=True)
# def singlecell_prob_all(R, L, T_act, T_inh):
#     P = np.zeros((R.size, R.size))
#     for i in range(R.size):
#         for j in range(R.size):
#             P[i][j] = P_pair_all(R[j], L[i], T_act[j], T_inh[j])
#     P = P/np.sum(P)
#     return P
#
# # select block of mat between sender_type and receiver_type
# def subcluster_sign(P, types, sender_type, receiver_type):
#     P_sel = P.copy()
#     P_sel = P_sel[types==sender_type]
#     P_sel = P_sel[:, types==receiver_type]
#     prob = np.sum(P_sel)
#     return prob
#
# def subcluster_mat(labels, combs, R, L, T_act=[0], T_inh=[0], normalize=True, model='mass_action'):
#     # step 1 - compute the cell-cell interaction matrix
#     if len(T_act)==1 and len(T_inh)==1:
#         print('no genes')
#         p_single = singlecell_prob_LR(R, L, model=model)
#     elif len(T_inh)==1:
#         print('only up')
#         p_single = singlecell_prob_LRU(R, L, T_act, model=model)
#     elif len(T_act)==1:
#         print('only down')
#         p_single = singlecell_prob_LRD(R, L, T_inh)
#     else:
#         print('all genes')
#         p_single = singlecell_prob_all(R, L, T_act, T_inh)
#
#     # step 2 - aggregate based on the passed labels
#     # if order:
#     #     l=order
#     # else:
#     #     l = sorted(list(set(labels)))
#     p = np.zeros((len(combs), len(combs)))
#     for i in range(len(combs)):
#         for j in range(len(combs)):
#             p[i][j] = subcluster_sign(p_single, np.asarray(labels), combs[i], combs[j])
#             # print(l[i], l[j], p[i][j])
#     if normalize:
#         p = p/np.sum(p)
#     return p
#
#
#
# def subcluster_mat_RNAVel(labels, R, L, V_act, order=False):
#     # step 1 - compute the cell-cell interaction matrix
#     p_single = singlecell_prob_LRU_RNAVel(R, L, V_act)
#
#     # step 2 - aggregate based on the passed labels
#     if order:
#         l=order
#     else:
#         l = sorted(list(set(labels)))
#     p = np.zeros((len(l), len(l)))
#     for i in range(len(l)):
#         for j in range(len(l)):
#             p[i][j] = subcluster_sign(p_single, np.asarray(labels), l[i], l[j])
#             # print(l[i], l[j], p[i][j])
#     return p
