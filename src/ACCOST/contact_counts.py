
import numpy as np
import csv
import gzip
import logging
import copy
import scipy
from scipy import sparse
from scipy.sparse import coo_matrix
import sklearn.cross_validation


class contactCountMatrix:
    """
    Class to represent contact count matrices. Has functionality for modeling distribution of counts.
    
    Args:
       allBins: dictionary of chromosome + midpoints to bin index
       allBins_reversed: list of indices, each pointing to a (chr,mid) tuple
       badBins: list of indices of bad (low mappability) bins
    """
    def __init__(self,allBins,allBins_reversed,badBins):
        self.allBins = allBins
        self.allBins_reversed = allBins_reversed
        self.badBins = badBins
        self.data = None
        self.biases = None
        
    def load_from_file(self,filename):
        """
        Load a contact count matrix from a file.

        Matrix should be in tab delimited text format:
        
        chr10    5000    chr10    5000    2
        chr10    5000    chr10    15000    6
        chr10    5000    chr10    105000    2
        
        Gzipped (detected by having '.gz' at the end of the filename) data is allowed.

        Args:
           filename: the name of the file containing the matrix to load
        """
        if filename.endswith('.gz'):
            fh = gzip.open(filename)
        else:
            fh = open(filename)
        reader = csv.reader(fh,delimiter='\t')
        
        logging.info('Loading matrix from %s',filename)
        n = 0
        rows = []
        cols = []
        counts = []
        rowset = set()
        colset = set()
        for line in reader:
            (chr1,mid1,chr2,mid2,count) = line
            mid1 = int(mid1)
            mid2 = int(mid2)
            count = float(count)
            #enforce lower triangular
            if int(chr1[3:]) > int(chr2[3:]) or (chr1 == chr2 and int(mid1) > int(mid2)):
                chrtemp = chr1
                chr1 = chr2
                chr2 = chrtemp
                midtemp = mid1
                mid1 = mid2
                mid2 = midtemp
            assert int(chr1[3:]) <= int(chr2[3:]), "Matrix should be lower triangular (chrs: %s %s)" % (chr1,chr2)
            assert int(chr1[3:]) < int(chr2[3:]) or (chr1 == chr2 and mid1 <= mid2), "Matrix should be lower triangular (mids: %s %s)" % (mid1,mid2)
            assert self.allBins[chr1][mid1] <= self.allBins[chr2][mid2], "???? bin indices not lower triangular %d %d" % (self.allBins[chr1][mid1],self.allBins[chr2][mid2]) 
            rows.append(self.allBins[chr1][mid1])
            cols.append(self.allBins[chr2][mid2])
            counts.append(float(count))
            rowset.add(self.allBins[chr1][mid1])
            colset.add(self.allBins[chr2][mid2])
            n+=1
        for i in range(len(self.allBins_reversed)):
            if i not in rowset:
                rows.append(i)
                cols.append(0)
                counts.append(0.0)
            if i not in colset:
                cols.append(i)
                rows.append(0)
                counts.append(0.0)
        data = scipy.sparse.coo_matrix((counts,(rows,cols)))
        logging.debug("min and max of rows: %d %d" % (np.min(rows),np.max(cols)))
        logging.debug("min and max of cols: %d %d" % (np.min(cols),np.max(cols)))
        logging.debug("data shape: " + str(np.shape(data)))
        logging.debug("dense data shape: " + str(np.shape(data.todense())))
        self.data = scipy.sparse.csc_matrix(data)
        self.N = n
    
    def load_ICE_biases_vec(self,filename):
        """
        Load an ICE bias vector from a raw file, one bias per line.
        
        Gzipped (detected by having '.gz' at the end of the filename) data is allowed.
        
        This function sets the biases and biasBins attributes of the contact count matrix.
        biasBins is a list of boolens corresponding to indices where the biases are defined.

        Args:
            filename: the name of the file containing the bias vector

        """
        logging.info("loading biases from %s" % filename)
        biases = np.loadtxt(filename)
        self.biases = biases
        logging.debug("biases shape: " + str(np.shape(biases)))
        logging.debug("data shape: " + str(np.shape(self.data)))
        biasBins = np.array(range(len(biases)))
        biasMask = np.array([False]*len(biases))
        self.biasBins = biasBins
        self.biasMask = biasMask

 
    def load_ICE_biases(self,filename,trim_counts=False):
        """
        Load an ICE bias vector from a file.
        
        Biases should be in tab delimited text format:

        chr1    5000    1.0
        chr1    15000    1.0
        chr1    25000    1.0
        chr1    35000    0.710793549712
        chr1    45000    0.313981161587
        
        Gzipped (detected by having '.gz' at the end of the filename) data is allowed.
        
        This function sets the biases and biasBins attributes of the contact count matrix.
        biasBins is a list of boolens corresponding to indices where the biases are defined.
 
        Args:
            filename: the name of the file containing the bias vector
        """
        logging.info("loading biases from %s" % filename)
        if filename.endswith('.gz'):
            fh = gzip.open(filename)
        else:
            fh = open(filename)
        reader = csv.reader(fh,delimiter='\t')
        n = 0
        biases = np.zeros(len(self.allBins_reversed))
        biasBins = []
        
        for line in reader:
            (chr,mid,bias) = line
            mid = int(mid)
            if mid in self.allBins[chr]:
                index = self.allBins[chr][mid]
                biases[index] = bias
                biasBins.append(index)
                n+=1
        biasMask = []
        for i in range(np.shape(self.data)[0]):
            if i in biasBins:
                biasMask.append(False)
            else:
                biasMask.append(True)
        self.biases = np.array(biases)
        logging.debug("biases shape: " + str(np.shape(biases)))
        logging.debug("data shape: " + str(np.shape(self.data)))
        self.biasBins = biasBins
        self.biasMask = biasMask
    
    def mask_matrix(self, mask_zeros=False, mask_no_bias=True, mask_low_mappability=False):
        """Mask bins without a bias or with low mappability bins"""
        # only use counts for which a bias is known
        biasMask = np.maximum.outer(self.biasMask,self.biasMask)
        # mask zeros
        if mask_zeros:
            zeroCounts = self.data != 0
            zeroCounts = ~(zeroCounts.todense())
            mask = np.logical_or(biasMask,zeroCounts)
        else:
            mask = biasMask
        
        masked_counts = np.ma.masked_array(self.data.todense(), mask)
        
        # mask rows and columns in bad bins (poor mappability)
        if mask_low_mappability:
            masked_counts[self.badBins,:] = np.ma.masked
            masked_counts[:,self.badBins] = np.ma.masked
        
        self.masked_counts = masked_counts 
        self.masked = True
        

def generate_binpairs(allBins):
    """
    Generate all bin pairs from a dictionary of bins.

    Args:
       allBins: dictionary of chr to mid to index, generated by generate_bins
    
    Returns:
       binpairs: dictionary of chr1 to mid1 to chr2 to mid2 to 0
    """
    raise NotImplementedError("Shouldn't need this!")
    logging.info('Generating bin pairs')
    binpairs = {}
    binpairs_reversed = []
    for chr1 in allBins.keys():
        binpairs[chr1] = {}
        for mid1 in allBins[chr1].keys():
            binpairs[chr1][mid1] = {}
            for chr2 in allBins.keys():
                if chr1 <= chr2:
                    binpairs[chr1][mid1][chr2] = {}
                    for mid2 in allBins[chr2].keys():
                        if chr1 < chr2 or (chr1 == chr2 and mid1 <= mid2):
                            binpairs[chr1][mid1][chr2][mid2] = 0
                            binpairs_reversed.append((chr1,mid1,chr2,mid2))
    return binpairs,binpairs_reversed

def generate_bins(midsFile,lowMappThresh):
    """
    Generate data structures to hold binned data and keep track of "bad" (low mappability) bins.

    Args:
       midsFile: name of the file with bin midpoints and mappability data. Tab delimited file with format:
           <chr ID>        <mid>   <anythingElse> <mappabilityValue> <anythingElse>+ 
               chr10   50000   NA      0.65    ...
       lowMappThresh: threshold at which we want to discard those bins

    Returns:
       allBins: dictionary of chromosome + midpoints to bin index
       allBins_reversed: list of indices, each pointing to a (chr,mid) tuple
       badBins: list of indices of bad (low mappability) bins

    """
    badBins = [] # list of indices with mappabilties < lowMappThresh
    allBins = {} # chr and mid to index
    allBins_reversed = [] # index to chr and mid    

    fh = open(midsFile)
    reader = csv.reader(fh, delimiter='\t')
    i = 0
    logging.info('Reading bin midpoints and mappability from %s',midsFile)
    for line in reader: # might need a check here to avoid header line
        chr = line[0]
        mid = int(line[1])
        mappability = float(line[3])
        logging.debug("chr: %s mid: %8d map: %1.3f" % (chr,mid,mappability))
        if chr not in allBins:
            allBins[chr] = {}
        allBins[chr][mid] = i
        allBins_reversed.append((chr,mid))
        if mappability <= lowMappThresh:
            badBins.append(i)
        i+=1
    fh.close()
    return(allBins,allBins_reversed,badBins)

def dict_to_sparse(counts,allBins_reversed):
    """
    Convert count matrix in nested dictionary format to sparse scipy coordinate matrix format.

    Args:
       counts: nested dictionary of counts (ie data[chr1][mid1][chr2][mid2])
       allBins_reversed: index of ordered bins. list of (chr,mid) tuples
    
    Returns:
       sparse_counts: sparse format matrix in scipy COO (coordinate) format
    """
    raise NotImplementedError("Shouldn't need this!")
    rows = []
    cols = []
    data = []
    n = 0
    for i,(chr1,mid1) in enumerate(allBins_reversed):
        for j,(chr2,mid2) in enumerate(allBins_reversed):
            if mid1 in counts[chr1]:
                if chr1 <= chr2 and mid2 in counts[chr1][mid1][chr2]:
                    if counts[chr1][mid1][chr2][mid2] > 0:
                        rows.append(i)
                        cols.append(j)
                        data.append(counts[chr1][mid1][chr2][mid2])
                        n += 1
                        logging.info("%04d %04d %04.4f" % (i,j,counts[chr1][mid1][chr2][mid2]))
    logging.debug("n= %4d" % n)
    sparse_counts = sparse.coo_matrix((np.array(data,dtype=np.float64),(np.array(rows,dtype=np.int16),np.array(cols,dtype=np.int16))), dtype=np.float64)
    (r,c,d) = sparse.find(sparse_counts)
    logging.info(r[1:10])
    logging.info(c[1:10])
    logging.info(d[1:10])
    return sparse_counts

def sparse_to_dict(counts_sparse,allBins,allBins_reversed):
    """
    Convert contact count matrix in scipy sparse format to nested dictionary format.
    
    Args:
        counts_sparse: scipy sparse matrix of contact counts
    
    Returns:
        counts: dictionary of contact counts
    """
    raise NotImplementedError("Shouldn't need this!")
    logging.info("counts_sparse shape: " + str(np.shape(counts_sparse)))
    logging.info("allBins length: %04d" % len(allBins))
    logging.info("allBins_reversed length: %04d" % len(allBins_reversed))
    counts,binpairs_reversed = generate_binpairs(allBins)
    counts_sparse_csc = counts_sparse.tocsc()
    for i,(chr1,mid1) in enumerate(allBins_reversed):
        for j,(chr2,mid2) in enumerate(allBins_reversed):
            if chr1 < chr2 or (chr1 == chr2 and mid1 <= mid2):
                if i < np.shape(counts_sparse_csc)[0] and j < np.shape(counts_sparse_csc)[1]:
                    counts[chr1][mid1][chr2][mid2] = counts_sparse_csc[i,j]
                    logging.debug("%04d, %04d, %04.4f" % (i,j,counts_sparse_csc[i,j]))
    return counts


def get_lengths(allBins_reversed):
    """
    Calculate distance between all bins (not necessarily with
    a count matrix attached)
    
    Distance is defined by: | mid2 - mid1 |
    
    Interchromosomal lengths are set to -1    

    Args:
        TODO
    Returns:
        TODO
    """
    logging.info('Generating lengths')
    n = len(allBins_reversed)
    lengths = np.zeros((n,n))
    lengths_reversed = {}
    lengths_reversed[-1] = [] # we know this one exists!
    for (i,(chr1,mid1)) in enumerate(allBins_reversed):
        for (j,(chr2,mid2)) in enumerate(allBins_reversed):
            if chr1 != chr2:
                lengths[i,j] = -1
                lengths_reversed[-1].append((i,j))
            else:
                dist = abs(mid2-mid1)
                #print(mid1,mid2,dist)
                lengths[i,j] = dist
                if dist in lengths_reversed:
                    lengths_reversed[dist].append((i,j))
                else:
                    lengths_reversed[dist] = [(i,j)]
    return lengths,lengths_reversed


def get_lengths_for_matrix(allBins_reversed,mat,check_bias_indices=False):
    """
    Calculate distance between all bins in the full count matrix.
    
    Distance is defined by: | mid2 - mid1 |
    
    Interchromosomal lengths are set to -1    

    Args:
        TODO
    Returns:
        TODO
    """
    logging.info('Generating lengths')
    n = len(allBins_reversed)
    lengths = np.zeros((n,n))
    lengths_reversed = {}
    lengths_reversed[-1] = [] # we know this one exists!
    for (i,(chr1,mid1)) in enumerate(allBins_reversed):
        for (j,(chr2,mid2)) in enumerate(allBins_reversed):
            if i<np.shape(mat.data)[0] and j<np.shape(mat.data)[1]:
                if check_bias_indices and (i in mat.biasBins and j in mat.biasBins):
                    if chr1 != chr2:
                        lengths[i,j] = -1
                        lengths_reversed[-1].append((i,j))
                    else:
                        dist = abs(mid2-mid1)
                        lengths[i,j] = dist
                        if dist in lengths_reversed:
                            lengths_reversed[dist].append((i,j))
                        else:
                            lengths_reversed[dist] = [(i,j)]
    return lengths,lengths_reversed










