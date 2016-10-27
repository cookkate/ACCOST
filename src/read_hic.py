# import matplotlib.pyplot as plt
import numpy as np

from mirnylib import genome
from mirnylib import h5dict
from mirnylib import plotting
from hiclib import binnedData

genome_db = genome.Genome('/net/noble/vol1/data/reference_genomes/mm9/chromosomes', readChrms=['#', 'X'])

# genome_db = genome.Genome('/net/noble/vol1/data/reference_genomes/mm9/chromosomes', readChrms=['1', '2'])

# Read resolution from the dataset.
raw_heatmap = h5dict.h5dict('~/2016ACCOST/data/1000000.either.hdf5', mode='r') 
resolution = int(raw_heatmap['resolution'])

print str(resolution)
print str(raw_heatmap.get_dataset('heatmap'))

data = binnedData.binnedData(resolution, genome_db)
data.simpleLoad('~/2016ACCOST/data/1000000.either.hdf5', '~/2016ACCOST/data/keys')

print str(data.dataDict.keys())

