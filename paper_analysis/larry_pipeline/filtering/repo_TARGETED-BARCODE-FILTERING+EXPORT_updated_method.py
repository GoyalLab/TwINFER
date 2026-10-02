#!/usr/bin/env python
# [UNREVIEWED: rescued 2026-09-30 from scratchpad_triage; logic not yet reviewed. Placement is a best guess, see RESCUE_MAP.tsv]
# coding: utf-8


import math
import pandas as pd
import scanpy as sc
from Bio import pairwise2
import re
import Levenshtein
import scanpy
from scipy.spatial.distance import hamming
import numpy as np
from multiprocessing import Pool
from collections import Counter
import shutil
import os
import pickle
import anndata


# This is our lineage barcode sequence - replace this sequence with the sequence you want to match to - variable barcodes should be specified by N
bc_seq = 'ctcggcatggacgagctgtacaagtaaagcggccgcgtcgagagatgggggaggctaactgagctagcttggagacTGCGTATGATNNNNCTNNNNACNNNNTCNNNNGTGCTGATACCGTTATTAACATATGACAACTCAATTAAACACTGaccggtactagcttagtcatgcaccgggctgcaggaattcga'

bc_seq = bc_seq.upper()

# Input your lineage barcode sequence 
barcode = 'AT....CT....AC....TC....GT'
partial_barcode_left = 'TC....GTGCTGATACCGTTATT'
partial_barcode_right = 'TGGAGACTGCGTATGAT....CT'

# Indicate Anndata file path
T1_filepath = '../data/202100909_Lineage_BC_T1.h5ad'
T2_filepath = '../data/202100909_Lineage_BC_T2.h5ad'

# Make sequence regular expression
bc_seq_re = ''
for x in bc_seq:
    if x == 'N':
        bc_seq_re += '.'
    else:
        bc_seq_re += x
print('Length + BC Sequence:', len(bc_seq_re), bc_seq_re)



# DEFINE FUNCTIONS
#
#
def create_barcode_df_from_fastq(files, save_dir):
    for q, y in enumerate(files):
        # iterate through the fastq file and only include lines with barcode and UMI
        ls = []
        for i in range(len(y.index)):
            x = y['Sequence'][i]
            if ('A' in x and 'G' in x and 'T' in x and 'C' in x) and len(x)==131 and not x.startswith('@') and not x.startswith('#'):
                ls[i-1] = True
                ls.append(True)
            else:
                ls.append(False)
        y['Filter'] = ls
        y = y[y['Filter']]
        
	# Separate UMI and Cell barcode
        ls = []
        ls1 = []
        for x in y['Sequence']:
            if '_' in x:
                sp = x.split('_')
                sp1 = sp[2].split('2')
                ls.extend([sp[1], sp[1]])
                ls1.extend([sp1[0], sp1[0]])

        y['Cell Barcode'] = ls
        y['UMI'] = ls1
    		
        # Remove UMI/Cell barcode line
        ls = []
        for x in y['Sequence']:
            if x.startswith('@'):
                ls.append(False)
            else:
                ls.append(True)
        y['Filter'] = ls
        y = y[y['Filter']]
        y.drop('Filter', inplace=True, axis=1)

        files[q] = y

    # Save checkpoint
    for i, x in enumerate(files):
        x.to_csv(save_dir+'S'+str(i)+'-barcode_df_0.csv')
	
    return files


# This function aligns sequences to the lineage barcode vector sequence
def align_to_barcode_pb(h, files, bc_seq_re, localms_param=[1, -0.5, -1, -0.5]):
    print(h, 'dataset')
    y = files[h]
    ls = []
    ls1 = []
    scores = []
    pb = []
    barcodes = []
    y_len = len(y)
    for x in y['Sequence']:
        # compute alignment score with biopython local alignment tool
        alignments = pairwise2.align.localms(bc_seq_re, x, localms_param[0], localms_param[1], localms_param[2], localms_param[3])
        if alignments:
            scores.append(alignments[0].score)
            if alignments[0].score > 60:

                # Determine if the barcode aligns with right end, middle, or left end of sequence
                re1 = re.search(barcode, x)
                rel = re.search(partial_barcode_left, x)
                rer = re.search(partial_barcode_right, x)
                if re1:
                    barcodes.append(re1.group())
                    pb.append(False)
                    ls.append(True)
                elif rel:
                    barcodes.append(x[0:rel.span()[0]] + rel.group()[0:8]) # NOTE: appended group may need to be altered for different lineage barcode seq
                    pb.append(True)
                    ls.append(True)
                elif rer:
                    barcodes.append(rer.group()[-8:] + x[rer.span()[1]:])
                    pb.append(True)
                    ls.append(True)
                else:
                    barcodes.append('N/A')
                    pb.append(False)
                    ls.append(False)
            else:
                barcodes.append('N/A')
                ls.append(False)
                pb.append(False)
        else:
            ls.append(False)
            barcodes.append(False)
            pb.append(False)

    y['Filter'] = ls
    y['Barcode'] = barcodes
    y['Partial Barcode?'] = pb
    y['Alignment Score'] = scores
    y = y[y['Filter']]
    files[h]=y
    y.to_csv(save_dir+'S'+str(h)+'-barcodes_df_aligned.csv')


# This should only be used if no partial barcodes were extracted previously
def align_to_barcode(h, files, bc_seq_re, localms_param=[1, -0.5, -1, -0.5]):
    print(h, 'dataset')
    y = files[h]
    ls = []
    ls1 = []
    scores = []
    pb = []
    barcodes = []
    y_len = len(y)
    for x in y['Sequence']:
	# compute alignment score with biopython local alignment tool
        alignments = pairwise2.align.localms(bc_seq_re, x, localms_param[0], localms_param[1], localms_param[2], localms_param[3])
        if alignments:
            scores.append(alignments[0].score)
            re1 = re.search(barcode, x)
            if alignments[0].score > 60 and re1:
                barcodes.append(re1.group())
                ls.append(True)
                pb.append(False)
            else:
                barcodes.append('N/A')
                ls.append(False)
                pb.append(False)
        else:
            barcodes.append(False)
            ls.append(False)
            pb.append(False)

    y['Filter'] = ls
    y['Barcode'] = barcodes
    y['Partial Barcode?'] = pb
    y['Alignment Score'] = scores
    y = y[y['Filter']]
    y = y[[len(x) <= 26 for x in y['Barcode']]] # this removes any potential partial barcodes or barcodes that are not the correct length
    files[h]=y
    y.to_csv(save_dir+'S'+str(h)+'-barcodes_df_aligned.csv')


# Function for launching alignment
def run_barcode_alignment(files, bc_seq_re, localms_param=[1, -0.5, -1, -0.5], n_pcs=8):
    print('start barcode alignment')
    if __name__=='__main__':
    	with Pool(processes=n_pcs) as pool:
            args = [(i, files, bc_seq_re, localms_param) for i in range(len(files))]
            pool.starmap(align_to_barcode_pb, args)


# This function will essentially merge barcodes that have a similar sequence - I think we can simplify this by just 
# keeping barcodes that have multiple reads
# Previous lineage barcoding paper only kept reads that had at least 3 of a UMI-CB-LBC combo
def merge_similar_barcodes(q):
    y=files[q]
    ls = []
    cb = []
    drops = []
    i = 0
    len_q = len(y)
    print(q, 'dataset', len_q)
    for x in y.index:
        if y['UMI'][x] in ls:
            cbx = y['Cell Barcode'][x]
            cb2 = cb[ls.index(y['UMI'][x])]
            i2 = ls.index(y['UMI'][x])
            #print(cbx, cb2, cbx==cb2, x, i2)
            # print(s1l1[s1l1['UMI'] == s1l1['UMI'][x]])
            if cbx==cb2:
                drops.append(x)
            elif Levenshtein.distance(cbx, cb2) > 1:
                drops.append(x)
                drops.append(i2)
                # print('Levenshtein Distance', Levenshtein.distance(cbx, cb2))
                i+=1
            elif Levenshtein.distance(cbx, cb2) == 1:
                if 'N' in cbx:
                    drops.append(x)
                elif 'N' in cb2:
                    drops.append(i2)
                elif y['Partial Barcode?'][x]:
                    drops.append(x)
                elif cbx not in T1.obs_names:
                    drops.append(x)
                elif cb2 not in T1.obs_names:
                    drops.append(i2)
        cb.append(y['Cell Barcode'][x])
        ls.append(y['UMI'][x])
    drops = [x for x in drops if x in y.index]
    drops_out = [x for x in drops if x not in y.index]
    print("Number of UMI's that have Levenshtein distance > 1: ", i)
    print('Number of drops', len(drops))
    y.drop(drops, axis=0, inplace=True)
    y.to_csv(save_dir+'S'+str(q+1)+'-barcodes-merged.csv')
    for z in drops_out:
        del y.iloc[z]
    files[q] = y


def start_merge_similar_barcodes_function():
    if __name__ == '__main__':
    	with Pool(processes=8) as pool:
            pool.map(merge_similar_barcodes, range(len(files)))


# this function will filter out any sequecnes that have < threshold CB_BC_UMI combos
# this will also only keep 1 row for each CB_BC_UMI combo that is valid
# this should merge all sequencing into the same file, possible that combo multiples are in different files
def filter_umi_cb_lbc_combos(files, threshold=3):
    # concatenate dataset
    f = pd.concat([df.assign(origin='S'+str(i)) for i,df in enumerate(files)], ignore_index=True)
    print(f)
    f = f.drop(labels=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    f['CB_BC_UMI'] = f['Cell Barcode'] + f['Barcode'] + f['UMI']
    cb_bc_umi = dict(Counter(list(f['CB_BC_UMI'])))
    cb_bc_umi = [k for k, i in cb_bc_umi.items() if i >= threshold] # filter the dict for combos that had greater than threshold
    f = f[f['CB_BC_UMI'].isin(cb_bc_umi)]
    print(f)
    # keep only 1 row per valid combo
    print('keep one per valid combo')
    print(f)
    keep_ls = []
    #for x in f['CB_BC_UMI']:
    #    tmp = f[f['CB_BC_UMI']==x]
    #    keep_ls.append(tmp.index[0])
    #    print(len(keep_ls))
    #f = f[f.index.isin(keep_ls)]
    f = f.loc[f.groupby('CB_BC_UMI')['Alignment Score'].idxmax()]
    f.reset_index(inplace=True, drop=True)
    print(f)
    # Save file
    f.to_csv(save_dir+'all-barcodes.csv')


def launch_umi_cb_lbc_filter(files, threshold=3, n_pcs=8):
    if __name__=='__main__':
        with Pool(processes=n_pcs) as pool:
            args = [(i, files, threshold) for i in range(len(files))]
            pool.starmap(filter_umi_cb_lbc_combos, args)

# only include reads with CB in the dataset
def filter_bcs_not_in_adata(adata, bc, save='all-barcodes-in-scrnaseq.csv'):
    adata_cbs = [x.split('-')[0] for x in adata.obs_names]
    ls = []
    for x in bc['Cell Barcode']:
        ls.append(x in adata_cbs)
    bc['In scRNAseq?'] = ls
    bc = bc[bc['In scRNAseq?']]
    bc.to_csv(save)

#
#


# Load fastq files containing barcodes

S0 = pd.read_csv('../lineage-bc/barcodes-targeted/YM-1-169-T1-I1_S1_L001_R2_001_barcodes.fastq',
                         names=['Sequence'])
S1 = pd.read_csv('../lineage-bc/barcodes-targeted/YM-1-169-T1-I2_S2_L001_R2_001_barcodes.fastq',
                         names=['Sequence'])
S2 = pd.read_csv('../lineage-bc/barcodes-targeted/YM-1-169-T1-I3_S3_L001_R2_001_barcodes.fastq',
                         names=['Sequence'])
S3 = pd.read_csv('../lineage-bc/barcodes-targeted/YM-1-169-T1-I4_S4_L001_R2_001_barcodes.fastq',
                         names=['Sequence'])
S4 = pd.read_csv('../lineage-bc/barcodes-targeted/YM-1-169-T2-I1_S5_L001_R2_001_barcodes.fastq',
                         names=['Sequence'])
S5 = pd.read_csv('../lineage-bc/barcodes-targeted/YM-1-169-T2-I2_S6_L001_R2_001_barcodes.fastq',
                         names=['Sequence'])
S6 = pd.read_csv('../lineage-bc/barcodes-targeted/YM-1-169-T2-I3_S7_L001_R2_001_barcodes.fastq',
                         names=['Sequence'])
S7 = pd.read_csv('../lineage-bc/barcodes-targeted/YM-1-169-T2-I4_S8_L001_R2_001_barcodes.fastq',
                         names=['Sequence'])


# Create a files list from the files that have been read

files = [S0, S1, S2, S3, S4, S5, S6, S7]

# Specify which fastq files correspond to which timepoint in your anndata
T1_fastq = ['S0', 'S1', 'S2', 'S3']
T2_fastq = ['S0', 'S2', 'S3', 'S4']

# Provide a directory where you would like the files to be saved
save_dir = './lineage-bc-update/'



# RUN PIPELINE

#1
print('Step 1')
#files = create_barcode_df_from_fastq(files, save_dir)

# read in files
files1 = []
for x in range(len(files)):
    f = pd.read_csv(save_dir+'S'+str(x)+'-barcode_df_0.csv')
    files1.append(f)
files = files1

#2
print('Step 2')
#run_barcode_alignment(files, bc_seq_re, localms_param=[1, -0.5, -1, -0.5], n_pcs=8)

# Read in if you used the above save checkpoint in the align_to_barcode function
files1 = []
for x in range(len(files)):
    files1.append(pd.read_csv(save_dir+'S'+str(x)+'-barcodes_df_aligned.csv'))

files=files1

#3
print('Step 3')
filter_umi_cb_lbc_combos(files, threshold=3)

# Read in the results of the previous step
bc = pd.read_csv(save_dir+'all-barcodes.csv', index_col=0)


# Read in adata
T1 = sc.read(T1_filepath)
T2 = sc.read(T2_filepath)
# Combine adatas
adata = anndata.concat([T1, T2])

# 4
print('Step 4')
filter_bcs_not_in_adata(adata, bc, save=save_dir+'all-barcodes-in-scrnaseq.csv')


