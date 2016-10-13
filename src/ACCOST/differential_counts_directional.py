"""
differential_counts.py

Module to test differential contact counts.

"""

import warnings
import logging
import sys
import csv
import numpy as np
import scipy
from scipy import sparse
from numpy.random import negative_binomial
from scipy.stats import nbinom
from scipy.misc import logsumexp
#from test-fit-variance import set_up_vars_means
import contact_counts
from NBfit import LogPolyEstimator,LocfitEstimator


smallval = 10**-8 

def set_up_estimators(matrix,lengths_rows,lengths_cols,filter_lengths=2./3,filter_diagonal=True,filter_inter=True,min_counts=2):
    """
    This is a function to:

    1) Set up masks so that zeros, low mappability bins, etc (TODO: split this into a separate function)
    2) calculate mean, variance, w, and z for each length/genomic distance
        
    """
    (rows,cols,norm_counts) = scipy.sparse.find(matrix.data)
    
    assert matrix.masked, "Must mask bins without a bias before setting up estimators"
    
    
    # mask rows and columns in bad bins (poor mappability)
    biasmat = np.ma.masked_array(np.outer(matrix.biases,matrix.biases),matrix.masked_counts.mask)
    
    #calculate normalized counts
    norm_counts = matrix.masked_counts/biasmat
    recip_biases = 1/biasmat
    
    logging.debug("min lengths rows inside: %d" % min(lengths_rows))

    vars = []
    means = []
    lengths_stats = []
    z_factors = []

    max_length_to_use = filter_lengths*max(lengths_rows)

    for l in lengths_rows:
        if l > max_length_to_use:
            continue
        if filter_diagonal and l == 0:
            continue
        if filter_inter and l < 0:
            continue
        cur_i = lengths_rows[l]
        cur_j = lengths_cols[l]
        cur_counts = norm_counts[cur_i,cur_j]
        nonzero = np.nonzero(cur_counts)

        # get rid of zeros. there is a ton of numpy weirdness between versions
        if cur_counts.ndim < 2:
            cur_counts = cur_counts[np.newaxis,:][nonzero]
        else:
            cur_counts = cur_counts[nonzero]
        cur_recip_biases = recip_biases[cur_i,cur_j]
        if cur_recip_biases.ndim < 2:
            cur_recip_biases = cur_recip_biases[np.newaxis,:][nonzero]
        else:
            cur_recip_biases = cur_recip_biases[nonzero] # get rid of zeros
        cur_counts = cur_counts.copy()
        cur_counts = cur_counts.compressed().flatten()
        cur_recip_biases = cur_recip_biases.compressed().flatten()
        num_cur_counts = max(cur_counts.shape[0],cur_counts.shape[1])
        if num_cur_counts >= min_counts: # if we have enough unmasked counts for these bins
            v = np.var(cur_counts,ddof=1)
            u = np.mean(cur_counts)
            z = ( u / num_cur_counts ) * np.sum(cur_recip_biases)
            vars.append(v)
            means.append(u)
            lengths_stats.append(l)
            z_factors.append(z)
    vars = np.array(vars)
    z_factors = np.array(z_factors)
    means = np.array(means)
    lengths_stats = np.array(lengths_stats)
    w = vars - z_factors
    return w,vars,means,z_factors,lengths_stats
    #return np.array(lengths_stats),np.array(vars),np.array(means),np.array(z_factors)

def length_to_matrix(mat_shape, lengths, lengths_rows, lengths_cols, length_indexed, mask_zeros):
    """
    Given a length (genomic distance)-indexed dictionary or array, create the
    contact count matrix version of values and return it.
    
    """
    
    mat = np.zeros(mat_shape)
    
    for idx,l in enumerate(lengths):
        logging.debug("broadcasting (%d, %d)" % (idx,l))
        i = lengths_rows[l]
        j = lengths_cols[l]
        mat[i,j] = length_indexed[idx]
    
    if mask_zeros:
        mat = np.ma.masked_equal(mat,0)

    return mat
        
def get_null_NB_params(matrix,q0,prefix):
    """
    This function calculations dispersion (r) and probability of success (p)
    under the null distribution from the fitted g() and
 
    """
    assert matrix.fitted, "Must fit variance before calculating NB params"
    biases = matrix.biases
    biasmat = np.outer(biases,biases)
    
    logging.info("saving biasmat and normalized matrices")
    logging.info(matrix.data.shape)
    logging.info(biasmat.shape)
    np.savetxt(prefix + "_normalized.txt",np.divide(matrix.data.todense(),biasmat),delimiter='\t')
    np.savetxt(prefix + "_biasmat.txt",biasmat,delimiter='\t')
    
    est = matrix.est
    lengths_with_counts = matrix.fitted_lengths
    mat_shape = biasmat.shape
    
    logging.info("predicting raw variance")
    f_q0 = est.predict(q0)
    r_recip = np.zeros(q0.shape)
    q0_2 = np.multiply(q0,q0)
    positive_indices = np.where(q0_2>0)

    sigma = q0 + np.multiply(np.multiply(biasmat,biasmat),f_q0)
    #np.savetxt("sigma.txt",sigma,delimiter="\t")
    
    logging.info("calculating r and p")
    r_recip[positive_indices] = np.divide(f_q0[positive_indices],q0_2[positive_indices])
    r_recip[np.where(q0_2<=smallval)] = smallval
     
    r = 1/np.maximum(r_recip,smallval)
    p = np.divide(r,(r + np.multiply(biasmat,q0)))
    
    r[np.where(r<smallval)] = smallval
    np.clip(p,smallval,1-smallval)
    
    np.savetxt("q0_2.txt",q0_2,fmt="%5e",delimiter="\t")
    np.savetxt("q0.txt",q0,fmt="%5e",delimiter="\t")
    np.savetxt("f_q0.txt",f_q0,fmt="%5e",delimiter="\t")
    np.savetxt("r_recip.txt",r_recip,fmt="%5f",delimiter="\t")
    np.savetxt("r.txt",r,fmt="%5f",delimiter="\t")
    np.savetxt("p.txt",p,fmt="%5f",delimiter="\t")
    
    
    logging.info("min/max r: %.10e %.10e" % (np.min(r),np.max(r)))
    logging.info("min/max p: %.10e %.10e" % (np.min(p),np.max(p)))
    matrix.r = r
    matrix.p = p
    matrix.mu = np.multiply(biasmat,q0)
    matrix.sigma = sigma
    matrix.has_NB_params = True
    matrix.mat_shape = mat_shape

    return matrix

def standardize_lengths(matA,matB):
    """
    Make sure that the lengths in fitted_lengths are the same and in the same order
    
    also standardizes the indices of the 
    
    matrix.g_w_l
    matrix.q_l
    matrix.z_l
    
    """
    logging.info("standardizing lengths")
    
    lengths_A = matA.fitted_lengths
    lengths_B = matB.fitted_lengths

    new_g_A = []
    new_g_B = []
    
    new_q_A = []
    new_q_B = []
    
    new_z_A = []
    new_z_B = []
    
    new_v_A = []
    new_v_B = []
    
    new_w_A = []
    new_w_B = []
    
    new_lengths = []
    for idx, (l_A, l_B) in enumerate(zip(lengths_A,lengths_B)):
        if l_A == l_B:
            new_lengths.append(l_A)
            new_g_A.append(matA.g_fitted[idx])
            new_g_B.append(matB.g_fitted[idx])
            new_q_A.append(matA.q_l[idx])
            new_q_B.append(matB.q_l[idx])
            new_z_A.append(matA.z_l[idx])
            new_z_B.append(matB.z_l[idx])
            new_w_A.append(matA.w_l[idx])
            new_w_B.append(matB.w_l[idx])
            new_v_A.append(matA.v_l[idx])
            new_v_B.append(matB.v_l[idx])
        elif l_A in lengths_B: #in B but in a different spot
            idx_B = np.where(lengths_B==l_A)[0][0]
            new_lengths.append(l_A)
            new_g_A.append(matA.g_fitted[idx])
            new_g_B.append(matB.g_fitted[idx_B])
            new_q_A.append(matA.q_l[idx])
            new_q_B.append(matB.q_l[idx_B])
            new_z_A.append(matA.z_l[idx])
            new_z_B.append(matB.z_l[idx_B])
            new_w_A.append(matA.w_l[idx])
            new_w_B.append(matB.w_l[idx_B])
            new_v_A.append(matA.v_l[idx])
            new_v_B.append(matB.v_l[idx_B])

        #otherwise don't add this length since it's not in both
            
    matA.g_fitted = np.array(new_g_A)
    matA.q_l = np.array(new_q_A)
    matA.z_l = np.array(new_z_A)
    matA.w_l = np.array(new_w_A)
    matA.v_l = np.array(new_v_A)
    
    matB.g_fitted = np.array(new_g_B)
    matB.q_l = np.array(new_q_B)
    matB.z_l = np.array(new_z_B)
    matB.w_l = np.array(new_w_B)
    matB.v_l = np.array(new_v_B)
    
    matA.fitted_lengths = new_lengths
    matB.fitted_lengths = new_lengths
    
    return matA,matB

def pval(counts_A, dispersion_A, p_success_A, counts_B, dispersion_B, p_success_B):
    """
    Given two observed counts and the dispersions and probability of success for each NS distribution,
    calculate the p-value for those counts
    """
    # probability of observed data
    
    logging.debug("p_a: %f p_b: %f r_a: %f r_b: %f" % (p_success_A,p_success_B,dispersion_A,dispersion_B))
    logging.debug("counts A: %d  counts B: %d" % (counts_A,counts_B))
    
    log_p_counts_A = nbinom.logpmf(counts_A, n=dispersion_A, p=p_success_A)
    log_p_counts_B = nbinom.logpmf(counts_B, n=dispersion_B, p=p_success_B)
    
    log_p_counts = log_p_counts_A + log_p_counts_B
    
    # now we will calculate the p-value,
    # conditioning on the total count
    total_count = counts_A + counts_B
    numerator = []
    denominator = []
    for a in range(int(total_count)+1):
        b = total_count - a
        log_p_a = nbinom.logpmf(a,dispersion_A,p_success_A)
        log_p_b = nbinom.logpmf(b,dispersion_B,p_success_B)
        log_p_joint = log_p_a + log_p_b
        logging.debug("a: %f b: %f log_p_a: %f log_p_b %f p_counts: %f p_joint: %f" % (a,b,log_p_a,log_p_b,log_p_counts,log_p_joint))
        if log_p_joint <= log_p_counts:
            numerator.append(log_p_joint)
        denominator.append(log_p_joint)
    log_num_sum = logsumexp(numerator)
    log_dem_sum = logsumexp(denominator)
    if log_num_sum != 0 and log_dem_sum != 0:
        p_val = log_num_sum - log_dem_sum
    else:
        p_val = np.nan
    logging.debug("log_num_sum: %f log_dem_sum: %f log_p_val: %f" % (log_num_sum,log_dem_sum,p_val))
    return p_val,log_p_counts_A,log_p_counts_B

def get_normalized_sum(matA,matB):
    biases_A = matA.biases
    biasmat_A = np.outer(biases_A,biases_A)
    q_A = matA.data / biasmat_A
    
    biases_B = matB.biases
    biasmat_B = np.outer(biases_B,biases_B)
    q_B = matB.data / biasmat_B
    
    sum_mat = q_A + q_B
    return sum_mat


def get_percentile(sum_mat,lengths,lengths_rows,lengths_cols,perc=80):
    intrachromosomal_sums = np.array([])
    for l in lengths:
        if l < 0:
            continue
        print l
        print lengths_rows[l]
        print lengths_cols[l]
        intrachromosomal_sums = np.append(intrachromosomal_sums,(sum_mat[lengths_rows[l],lengths_cols[l]]).flatten())
    val = mynanpercentile(intrachromosomal_sums,perc)
    return val
    

def mynanpercentile(a,perc):
    part = a.ravel()
    c = np.isnan(part)
    s = np.where(~c)[0]
    return np.percentile(part[s],perc)
    

def get_q0(matA,matB): 
    biases_A = matA.biases
    biasmat_A = np.outer(biases_A,biases_A)
    q_A = matA.data / biasmat_A
    
    biases_B = matB.biases
    biasmat_B = np.outer(biases_B,biases_B)
    q_B = matB.data / biasmat_B
    
    q0 = 0.5 * (q_A + q_B)
    logging.debug("q0 shape: " + str(q0.shape))
    return q0

def pval_mat(matA,matB,lengths_mat,q0,normalized_cutoff):
    assert matA.has_NB_params and matB.has_NB_params, "Must calculate NB params before calculating p-values"

    # calculate p-value matrix
    log_p_vals = np.empty(matA.data.shape)
    log_p_vals[:] = np.nan
    log_p_A = np.empty(matA.data.shape)
    log_p_A[:] = np.nan
    log_p_B = np.empty(matA.data.shape)
    log_p_B[:] = np.nan
    nan_reasons = np.zeros(matA.data.shape)
    min_counts_mat = np.zeros(matA.data.shape)
    directions = np.zeros(matA.data.shape)
    

    
    biases_A = matA.biases
    biasmat_A = np.outer(biases_A,biases_A)
    biases_B = matB.biases
    biasmat_B = np.outer(biases_B,biases_B)

    
    for i in range(matA.data.shape[0]):
        for j in range(matA.data.shape[1]):
            if i>j:
                continue # force upper triangular
            if lengths_mat[i,j] == -1:
                continue # skip interchromosomal
            #if abs(i-j)<5:
            #    continue #skip diagonal & 4 off-diagonals
            logging.debug("(i,j) = (%d,%d)" % (i,j))
            if i % 10 == 0 and j % 10 == 0:
                logging.info('i,j:' + str((i,j)))
            counts_A = matA.data[i,j]
            counts_B = matB.data[i,j]
            norm_counts_A = counts_A / biasmat_A[i,j]
            norm_counts_B = counts_B / biasmat_B[i,j]
            sum_norm_counts = norm_counts_A + norm_counts_B
            if sum_norm_counts < normalized_cutoff:
                nan_reasons[i,j] = 99
                continue
            if counts_A == counts_B:
                log_p_vals[i,j] = 0
            else:
                # NB params
                dispersion_A = matA.r[i,j]
                dispersion_B = matB.r[i,j]
                p_success_A = matA.p[i,j]
                p_success_B = matB.p[i,j]
                logging.debug("q0: %f" % q0[i,j])
                logging.debug("mu_a: %f mu_b: %f sigma_a: %f sigma_b: %f" % (matA.mu[i,j],matB.mu[i,j],matA.sigma[i,j],matB.sigma[i,j]))
                #min_counts = get_min_counts(p_success_A,p_success_B,dispersion_A,dispersion_B,target_min_pval)
                #min_counts = 0
                #min_counts_mat[i,j] = min_counts
                if counts_A is np.ma.masked or counts_B is np.ma.masked:
                    #p_vals[i,j] = np.nan
                    log_p_vals[i,j] = np.nan
                    nan_reasons[i,j]  = 1 # counts_masked
                elif dispersion_A is np.ma.masked or dispersion_B is np.ma.masked:
                    #p_vals[i,j] = np.nan
                    log_p_vals[i,j] = np.nan
                    nan_reasons[i,j] = 2 #dispersion_masked
                elif p_success_A is np.ma.masked or p_success_B is np.ma.masked:
                    #p_vals[i,j] = np.nan
                    log_p_vals[i,j] = np.nan
                    nan_reasons[i,j] = 3 #p_success_masked
                else:
                    p,p_A,p_B = pval(counts_A, dispersion_A, p_success_A, counts_B, dispersion_B, p_success_B)
                    #p_vals[i,j] = p
                    log_p_vals[i,j] = p
                    log_p_A[i,j] = p_A
                    log_p_B[i,j] = p_B
                    if p == np.nan:
                        nan_reasons[i,j] = 4
    #np.savetxt("min_counts.txt",min_counts_mat,delimiter='\t')
    np.savetxt("ln_p_B.txt",log_p_A,delimiter='\t')
    np.savetxt("ln_p_A.txt",log_p_B,delimiter='\t')
    return log_p_vals,nan_reasons,directions

def load_lengths(precalculated_length_file):
    logging.info("Loading lengths")
    lengths_rows = {}
    lengths_cols = {}
    with open(precalculated_length_file) as fh:
        reader = csv.reader(fh,delimiter='\t')
        for line in reader:
            l,i,j = [int(x) for x in line]
            if l in lengths_rows:
                lengths_rows[l].append(i)
                lengths_cols[l].append(j)
            else:
                lengths_rows[l] = [i]
                lengths_cols[l] = [j]
    return lengths_rows,lengths_cols

def fit_mat(matrix,lengths_rows,lengths_cols):
    w,vars,means,z,l = set_up_estimators(matrix,lengths_rows,lengths_cols,filter_lengths=2./3,filter_diagonal=True,filter_inter=True,min_counts=4)
    
    X = means
    y = w
    
    if len(X.shape)<2:
        X = X[:,np.newaxis]
    if len(y.shape)<2:
        y = y[:,np.newaxis]
    
    # filter negatives
    X = X[np.where(X>0)]
    y = y[np.where(y>0)]
    #est = LogPolyEstimator(degree=3)
    est = LocfitEstimator()
    est.fit(X,y)
   
    w_all,v_all,means_all,z_all,l_all = set_up_estimators(matrix,lengths_rows,lengths_cols,min_counts=2,filter_diagonal = True,filter_inter=False,filter_lengths = 1)
   
    X_all = means_all
    y_all = w_all

    # set negatives to 0
    X[np.where(X<0)] = 0
    y[np.where(y<0)] = 0
    
    if len(X_all.shape)<2:
        X_all = X_all[:,np.newaxis]
    if len(y_all.shape)<2:
        y_all = y_all[:,np.newaxis]

    ypred = est.predict(X_all)
    
    matrix.w_l = w_all
    matrix.v_l = v_all
    matrix.est = est
    matrix.g_fitted = ypred
    matrix.q_l = means_all
    matrix.z_l = z_all
    matrix.fitted_lengths = l_all
    matrix.fitted = True
    return matrix

def main():
    logging.basicConfig(format='%(levelname)s from %(filename)s %(funcName)s :: %(message)s', level=logging.DEBUG)
    warnings.simplefilter("error")

    binfile = sys.argv[1]
    precalculated_length_file = sys.argv[2]
    precalculated_length_file_reversed = sys.argv[3]
    matfile_A = sys.argv[4] # normalized counts
    matfile_B = sys.argv[5] # normalized counts
    biasfile_A = sys.argv[6]
    biasfile_B = sys.argv[7]
    outprefix = sys.argv[8]
    target_min_pval = float(sys.argv[9])
    mappability_thresh = 0.25
     
    # set up bins   
    (allBins,allBins_reversed,badBins) = contact_counts.generate_bins(binfile,mappability_thresh)

    # read input contact count matrices
    matrixA = contact_counts.contactCountMatrix(allBins,allBins_reversed,badBins)
    matrixA.load_from_file(matfile_A)
    matrixA.load_ICE_biases_vec(biasfile_A)

    matrixB = contact_counts.contactCountMatrix(allBins,allBins_reversed,badBins)
    matrixB.load_from_file(matfile_B)
    matrixB.load_ICE_biases_vec(biasfile_B)
  
    # mask matrices
    matrixA.mask_matrix(mask_zeros=True, mask_no_bias=True, mask_low_mappability=True)
    matrixB.mask_matrix(mask_zeros=True, mask_no_bias=True, mask_low_mappability=True)
    
    
    #np.savetxt(outprefix+"_counts_A.txt",matrixA.data.todense(),fmt='%d',delimiter='\t')
    #np.savetxt(outprefix+"_counts_B.txt",matrixB.data.todense(),fmt='%d',delimiter='\t')
    
    lengths_mat = np.loadtxt(precalculated_length_file,delimiter="\t")
    lengths_cols,lengths_rows = load_lengths(precalculated_length_file_reversed)
    
    matrixA = fit_mat(matrixA,lengths_rows,lengths_cols)
    matrixB = fit_mat(matrixB,lengths_rows,lengths_cols)
   
    matrixA,matrixB = standardize_lengths(matrixA,matrixB)
    assert matrixA.fitted_lengths == matrixB.fitted_lengths
    #logging.info(sorted(matrixA.fitted_lengths))
    #logging.info(sorted(matrixB.fitted_lengths))
    
    np.savetxt("A_w_l.txt",matrixA.w_l,delimiter='\t')
    #np.savetxt("B_w_l.txt",matrixB.w_l,delimiter='\t')
    np.savetxt("A_l.txt",matrixA.fitted_lengths,delimiter='\t')
    #np.savetxt("B_l.txt",matrixB.fitted_lengths,delimiter='\t')
    np.savetxt("A_g_fitted.txt",matrixA.g_fitted,delimiter='\t')
    #np.savetxt("B_g_fitted.txt",matrixB.g_fitted,delimiter='\t')
    np.savetxt("A_q_l.txt",matrixA.q_l,delimiter='\t')
    #np.savetxt("B_q_l.txt",matrixB.q_l,delimiter='\t')
    #sys.exit()
    
    q0 = get_q0(matrixA,matrixB)
    np.savetxt(outprefix+"_q0.txt",q0,delimiter='\t')
    
    logging.info("get 80th percentile of the contact counts")
    sum_mat = get_normalized_sum(matrixA,matrixB)
    #perc_80 = get_percentile(sum_mat,matrixA.fitted_lengths,lengths_rows,lengths_cols,perc=0)
    perc_80 = get_percentile(sum_mat,matrixA.fitted_lengths,lengths_rows,lengths_cols,perc=80)
    logging.info("80th percentile is: %f" % perc_80)    

    logging.info("calculating NB params")
    matrixA = get_null_NB_params(matrixA,q0,"A")
    matrixB = get_null_NB_params(matrixB,q0,"B")    
    
     
    #biases_A = matrixA.biases
    #biases_B = matrixB.biases
    #biasmat = np.outer(biases_A,biases_A) 
    #np.savetxt("matA_norm.txt",np.multiply(matrixA.data,biasmat))
    #biasmat = np.outer(biases_B,biases_B) 
    #np.savetxt("matB_norm.txt",np.multiply(matrixB.data,biasmat))
     
    np.savetxt(outprefix+"_A_r.txt",matrixA.r,delimiter='\t')
    np.savetxt(outprefix+"_A_p.txt",matrixA.p,delimiter='\t')
    #np.savetxt(outprefix+"_B_r.txt",matrixB.r,delimiter='\t')
    #np.savetxt(outprefix+"_B_p.txt",matrixB.p,delimiter='\t')
    
    logging.info("calculating p-value mat") 
    ln_pvals,nan_reasons,directions = pval_mat(matrixA,matrixB,lengths_mat,q0,perc_80)
   
    #np.savetxt(outprefix+"_directions.txt",directions,fmt="%d",delimiter='\t')
    np.savetxt(outprefix+"_ln_pvals.txt",ln_pvals,fmt='%.3f',delimiter='\t') 
    #np.savetxt(outprefix+"_log_pvals.txt",pvals,fmt='%.3f',delimiter='\t') 
    #np.savetxt(outprefix+"_pvals.txt",pvals,delimiter='\t') 
    np.savetxt(outprefix+"_nan_reasons.txt",nan_reasons,fmt='%d',delimiter='\t')
    


if __name__ == "__main__":
    main()
