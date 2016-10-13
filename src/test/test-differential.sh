
#real data
#python differential_counts.py ../data/test_data/pfal3D7-pfal3D7.MboI.w10000 ../data/test_data/pfal3D7_w10000_lengths_reversed.tab /net/noble/vol2/home/ferhatay/public_html/proj/2013publishedHiCdata/data/Ay-GR2014/interactionCounts/beforeICE/10000/RINGS.gz /net/noble/vol2/home/ferhatay/public_html/proj/2013publishedHiCdata/data/Plasmo-Sexual2014/interactionCounts/beforeICE/10000/HiC_Pf-Ring-HP1-KD-Combined.gz /net/noble/vol2/home/ferhatay/public_html/proj/2013publishedHiCdata/data/Ay-GR2014/interactionCounts/afterICE/10000/RINGS.biases.gz /net/noble/vol2/home/ferhatay/public_html/proj/2013publishedHiCdata/data/Plasmo-Sexual2014/interactionCounts/afterICE/10000/HiC_Pf-Ring-HP1-KD-Combined.biases.gz test_output

# HP1 KD chromosome 2
#python differential_counts.py ../data/test_data/pfal3D7_chr2_w10000_mappability.txt ../data/test_data/pfal3D7_chr2_lengths_reversed.tab ../data/test_data/RING_chr2_raw.txt ../data/test_data/HP1KD_chr2_raw.txt ../data/test_data/RING_chr2_biases.txt ../data/test_data/HP1KD_chr2_biases.txt test_output 0.01

# this is the one that actually runs!

# HP1 KD chromosome 2 with bias vector
python differential_counts_directional.py ../data/test_data/pfal3D7_chr2_w10000_mappability.txt ../data/test_data/pfal3D7_chr2_lengths.tab ../data/test_data/pfal3D7_chr2_lengths_reversed.tab ../data/test_data/RING_chr2_raw.txt ../data/test_data/HP1KD_chr2_raw.txt ../data/test_data/RING_chr2_biases_vec_2.txt ../data/test_data/HP1KD_chr2_biases_vec_2.txt test_output 0.01


# Ring/troph chromosome 2
#tpython differential_counts.py ../data/test_data/pfal3D7_chr2_w10000_mappability.txt ../data/test_data/pfal3D7_chr2_lengths_reversed.tab ../data/test_data/RING_chr2_raw.txt ../data/test_data/TROPH_chr2_raw.txt ../data/test_data/RING_chr2_biases.txt ../data/test_data/TROPH_chr2_biases.txt


# Ring/troph chromosome 2, smaller set
#python differential_counts.py ../data/test_data/pfal3D7_chr2_small.txt ../data/test_data/pfal3D7_chr2_small_lengths_reversed.tab ../data/test_data/RING_chr2_small_raw.txt ../data/test_data/TROPH_chr2_small_raw.txt ../data/test_data/RING_chr2_small_biases.txt ../data/test_data/TROPH_chr2_small_biases.txt

# earlygam/gam chromosome 14
#python differential_counts.py ../data/test_data/pfal3D7-pfal3D7.MboI.w10000 ../data/test_data/pfal3D7_w10000_lengths_reversed.tab ../data/test_data/earlygam_chr14_raw_sorted.txt ../data/test_data/gam_chr14_raw_sorted.txt ../data/test_data/earlygam_chr14_biases.txt ../data/test_data/gam_chr14_biases.txt test_output
