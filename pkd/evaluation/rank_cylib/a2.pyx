# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from __future__ import print_function

import cython
import numpy as np
cimport numpy as np
from collections import defaultdict
import random


cpdef a2(float[:, :] distmat, long[:] q_pids, long[:] g_pids,
                       long max_rank):

    cdef long num_q = distmat.shape[0]
    cdef long num_g = distmat.shape[1]

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
    
    cdef:
        np.ndarray[long, ndim=2] indices = np.argsort(distmat, axis=1)
        np.ndarray[long, ndim=2] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int64)

        np.ndarray[float, ndim=2] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
        np.ndarray[float, ndim=1] all_AP = np.zeros(num_q, dtype=np.float32)
        float num_valid_q = 0.0  # number of valid query

        np.ndarray[float, ndim=1] raw_cmc = np.zeros(num_g, dtype=np.float32)  # binary vector, positions with value 1 are correct matches
        np.ndarray[float, ndim=1] cmc = np.zeros(num_g, dtype=np.float32)
        long num_g_real, rank_idx
        unsigned long meet_condition

        float num_rel
        np.ndarray[float, ndim=1] tmp_cmc = np.zeros(num_g, dtype=np.float32)
        float tmp_cmc_sum

    # Print the number of unique query and gallery identities
    #print(f'Number of queries: {num_q}')
    #print(f'Number of gallery samples: {num_g}')
    
    unique_q_pids = np.unique(q_pids)
    unique_g_pids = np.unique(g_pids)
    
    #print(f'Number of unique query identities: {len(unique_q_pids)}')
    #print(f'Number of unique gallery identities: {len(unique_g_pids)}')
    
    for q_idx in range(num_q):
        # get query pid
        q_pid = q_pids[q_idx]

        num_g_real = 0
        meet_condition = 0
        
        # Cast matches[q_idx] to float
        raw_cmc = matches[q_idx].astype(np.float32)
        
        # Check if the query identity appears in the gallery
        if q_pid in unique_g_pids:
            meet_condition = 1
        
        if not meet_condition:
            # this condition is true when query identity does not appear in gallery
            # print(f'Query PID {q_pid} does not appear in gallery')
            continue
        
        # compute cmc
        function_cumsum(raw_cmc, cmc, num_g_real)
        for g_idx in range(num_g_real):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1
        
        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        num_valid_q += 1.0
    
        # compute average precision
        function_cumsum(raw_cmc, tmp_cmc, num_g_real)
        num_rel = 0
        tmp_cmc_sum = 0
        for g_idx in range(num_g_real):
            tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.0)) * raw_cmc[g_idx]
            num_rel += raw_cmc[g_idx]
        all_AP[q_idx] = tmp_cmc_sum / num_rel
    
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    # compute averaged cmc
    cdef np.ndarray[float, ndim=1] avg_cmc = np.zeros(max_rank, dtype=np.float32)
    
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q
    
    cdef float mAP = 0.0
    
    for q_idx in range(num_q):
        mAP += all_AP[q_idx]
    mAP /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), mAP


# Compute the cumulative sum
cdef void function_cumsum(np.ndarray[float, ndim=1] src, np.ndarray[float, ndim=1] dst, long n):
    
    cdef long i
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]
