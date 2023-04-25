""" Algorithms from pseudo-code """

import Attack_fig6.constants as cons
import Attack_fig6.helpers as helpers
import numpy as np
import random


def attack_nn(grads, params, layer_dims, small_grads):
    """ Run the attack algorithm """

    labels, small_grads, Index_sets = algo_B1(grads)
    activation_patterns = algo_B3(grads, params, layer_dims, Index_sets, small_grads)

    return activation_patterns


def algo_B1(grads):
    """
    Algorithm B.1

    Input: The gradient of (the weight the last hidden layer W_H), which is G_H_bar

    Output: Reconstructed labels,
            loss vectors 'small_grads',
            Index sets
    """

    # The last Hidden layer
    # Our FCN has only one hidden layer, its size is 512
    G_H_bar = grads['Layer_0_w'].T
    nr_of_classes = cons.OUT_SIZE

    # r_c c=1 to K
    r_c = []

    # Compute ratio vector r_c = [G_H_bar]_c / [G_H_bar]_1 for each class
    for i in range(nr_of_classes):
        r_c.append(G_H_bar[i, :]/G_H_bar[0, :])

    # !!! Find all duplicates in r_2
    # and group them by the duplicated value
    index_sets = helpers.find_duplicate_indices(r_c[1])
    print(f'Estimated M is {len(index_sets.keys())}')

    # sample-wise
    ratio_gcm_g1m = [[] for _ in range(cons.BATCH_SIZE_TRAIN)]

    # Get all values of g_c_m_bar / g_1_m_bar
    for c in range(nr_of_classes):
        for m in range(cons.BATCH_SIZE_TRAIN):
            try:
                j = random.choice(index_sets[m].tolist())
                ratio_gcm_g1m[m].append(r_c[c][j])
            except:
                pass

    # Upper bound delta_m, sample-wise
    Deltas = []
    # Final output g_c_m, sample-wise
    g_c_m = []

    for m in range(cons.BATCH_SIZE_TRAIN):
        Y_m = algo_B2(ratio_gcm_g1m[m])

        Delta_m = 1 / ratio_gcm_g1m[Y_m]
        Deltas.append(Delta_m)

        g_1_m = 2 * Delta_m / 3

        g_c_m.append([ratio * g_1_m for ratio in ratio_gcm_g1m[m]])

    return Y_m, g_c_m, index_sets


def algo_B2(small_grads):
    """   B.2 Algorithm

    Input: loss vector for the m-th sample 'small_grads' from 'algo_B1'

    Output: Reconstructed labels
    """
    return np.where(small_grads < 0)[1]


def algo_B3(grads, params, layer_dims, Ind_ExANs_H, small_grads):
    """   
    Algorithm B.3

    Input: Gradients weights, layer wise,
           Index sets of ExAN at last ReLU,
           Reconstructed loss vector for the m-th sample 'small_grads' from 'algo_B1'

    Output: Reconstructed attivation pattern
    """

    grads = np.array([grads['Layer_0_w'], grads['Layer_1_w']])
    H = grads.shape[0]
    I_cur = Ind_ExANs_H

    # Ugly as hell
    # we should find a better way to store the activation patterns
    D_pats = []
    for h in range(H):
        D_pats.append(np.zeros((cons.BATCH_SIZE_TRAIN, layer_dims[h+1], layer_dims[h+1])))


    for i in range(H-1, 0, -1):
        for m in range(2):
            # choose a random element j from I_cur[m]
            j_exan = random.choice(np.where(I_cur[m] > 0.)[0])
            # store the non-zero grads corresponding to the index of an ExAN
            D_pats[i-1][m] = np.diag(grads[i-1][j_exan,:])   

    # construct the index set I_cur from D_pats

    # solve binary equation for D_H^m

    small_grads = small_grads
    W_H = params['Layer_1_w']
    D_H = D_pats[-1]
    I_dH = ''
    grad_bias = grads['Layer_0_b']

    """
    mean_sample(sum_classes(small_grads @ W_H @ D_H @ I_dH)) - grad_bias = 0
    eq = 0
    """
    eq = np.mean(np.sum(small_grads @ W_H @ D_H @ I_dH, axis=1), axis=1) - grad_bias

    return D_pats 