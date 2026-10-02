import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numba
import tqdm
import scipy
import os
import sys
import joblib
import scanpy as sc
import os
import itertools
from itertools import product
from collections import defaultdict
from itertools import combinations
from scipy.stats import mannwhitneyu
from pathlib import Path
from adjustText import adjust_text
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
import json

# Calculation functions
import importlib
import sys
import os

from TwINFER_function_scripts import correlation_analysis_functions
from TwINFER_function_scripts import correlation_analysis_helpers

importlib.reload(correlation_analysis_functions)
importlib.reload(correlation_analysis_helpers)

from TwINFER_function_scripts.correlation_analysis_functions import (

    calculate_pairwise_gene_gene_correlation_matrix,
    check_system_in_steady_state,
    check_gene_gene_correlation_threshold,
    calculate_twin_random_pair_correlations,
    differentiate_single_state_reg_and_multiple_states,
    identify_reg_if_multiple_states,
    get_cross_correlations,
    identify_actual_directed_edges
)

# Helper functions
from TwINFER_function_scripts.correlation_analysis_helpers import (
    extract_param_index,
    read_input_matrix,
    get_param_data,
    plot_matrix_as_heatmap,
    print_summary,
    plot_network
)

import matplotlib.font_manager as fm
# ============================================================
# Fonts / style
# ============================================================
font_paths = [
    "fonts/Arial.ttf",
    "fonts/Arial Bold.ttf",
    "fonts/Arial Italic.ttf",
    "fonts/Arial Bold Italic.ttf",
]

for fp in font_paths:
    try:
        fm.fontManager.addfont(fp)
        print("✔ Loaded font:", fp)
    except Exception as e:
        print("⚠️  Could not load:", fp, "|", e)

# ==== LaTeX + SVG text mode (Illustrator-safe) ====
plt.rcParams['pdf.fonttype'] = 42  # For PDF export
plt.rcParams['ps.fonttype'] = 42   # For PostScript (EPS) export
plt.rcParams['font.sans-serif'] = ["Arial"]
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['svg.fonttype'] = "none"
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['axes.labelsize'] = 18     # x/y labels
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 12     # x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12    # y-axis tick labels
plt.rcParams['legend.fontsize'] = 12    # legend text

data_path = "real_data/"
path_to_plots = f"/home/gzu5140/Keerthana_b1042/grnInference/plots/Top_50_most_variable_genes_12032026/"
os.makedirs(path_to_plots, exist_ok=True)
path_to_plot_data = Path("/home/gzu5140/Keerthana_b1042/grnInference/analysisData/Top_50_most_variable_genes_12032026/")
path_to_plot_data.mkdir(exist_ok=True)

#Load the h5ad file
import subprocess
from pathlib import Path

repo = Path("/home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER")

cmd = """
module load git-lfs
git lfs pull --include=real_data/LSK_d2_d4_d6.h5ad
"""

result = subprocess.run(
    ["bash", "-lc", cmd],
    cwd=repo,
    capture_output=True,
    text=True,
)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)
print("Return code:", result.returncode)

# Define the full path to the data file
file_path = f'{data_path}LSK_d2_d4_d6.h5ad'
adata = sc.read_h5ad(file_path)
adata.obs_names_make_unique()

gene_list_Neutrophil = ['Muc13', 'Srgn', 'Ccl9', 'Plac8', 'Snrpf','Prtn3','Elane', 'Igfbp4', 'Ap3s1', 'Ctsg'] #Neutrophil
gene_list_Monocyte = ['Rbms1', 'Tuba1b','Sirpa', 'Ttf1', 'H3f3b', 'Set', 'Tk1', 'Fkbp4', 'Hspd1', 'Emb'] #Monocyte
gene_list_Regulator_TF = ['Gata1', 'Gata2', 'Gfi1', 'Fli1', 'Spi1', 'Tal1',  'Cebpa', 'Jun', 'Egr1', 'Nab2', 'Klf1', 'Zfpm1'] #TF involved in hematopoiesis regulation
gene_list_50_genes = ['Ybx1', 'Ttf1', 'Hmgb1', 'Ybx3', 'Nfkb1', 'Hmgb2', 'Srebf2', 'Smarce1', 'Csde1', 'Jund', 'Tsc22d1', 'Max', 'Sp1', 'Bcl11a', 'Irf9', 'Myc', 'Mef2c', 'Ssrp1', 'Tcf4', 'Cbfb', 'Trp53', 'Arid1a', 'Stat3', 'Sub1', 'Mta2', 'Xbp1', 'Cers2', 'Etv6', 'Ubtf', 'Ikzf1', 'Foxp1', 'Smarcc2', 'Dnajc2', 'Zmiz1', 'Nfe2l2', 'Gtf2i', 'Atf4', 'Sox4', 'Cdc5l', 'Zfp422', 'Usf2', 'Mta1', 'Gata2', 'Mafg', 'Creb1', 'Atf2', 'Klf6', 'Runx1', 'Mbd3', 'Rfx7']
gene_list_50_most_variable = ['MEIS1', 'RAD51', 'GFI1B', 'SPI1', 'HCFC1', 'MEF2C', 'IKZF1', 'JUN', 'IRF1', 'MYC', 'HDAC2', 'STAT3', 'NOTCH1', 'FOS', 'SATB1', 'ESR1', 'NFAT5', 'EBF1', 'GATA2', 'GFI1', 'ETS1', 'CBX7', 'CBFB', 'FOXO1', 'NR4A1', 'BMI1', 'IRF8', 'ELF4', 'BACH2', 'BHLHE40', 'POU2AF1', 'JUNB', 'NR1H3', 'STAT6', 'TCF7', 'SMAD4', 'STAT1', 'BCL11B', 'EGR2', 'STAT2', 'ELF1', 'CTSG', 'ELANE', 'MPO', 'MPL', 'CCL9', 'DUSP1', 'TYROBP', 'CLEC12A', 'LTB', 'LY6A', 'RGS1', 'IGKC', 'PROCR', 'RAB44', 'PDZK1IP1', 'SELL', 'ESAM', 'F630028O10RIK', 'MYL10', 'MAGED1', 'TRIM47', 'CD48', 'TBXAS1', 'BC035044', 'NEURL3', 'CST7', 'HACD4', 'MS4A3', 'RTP4', 'PGLYRP2', 'HK3', 'IFI44', 'NKG7', 'MMRN1', 'ATP8B4', 'MSMO1', 'GIMAP5', 'SERPINA3G', 'SLC18A2', 'ANXA3', 'NCEH1', 'DENND2D', 'PCP4L1', 'GIMAP1', 'LAPTM4B', 'PQLC3', 'RHOF', 'PEAR1', 'IL21R', 'LY6C2', 'LDHB', 'MS4A6B', 'RAB32', 'AP3S1', 'GBP2', 'PADI4', 'MYO1F', 'CASP1', 'CTSL', 'ALAS1', 'ITIH5', 'EMILIN1', 'ZFP386', 'TCP11L2', 'CISH', 'VILL', 'PPP1R15A', 'AFAP1L1', 'SMPDL3A', 'TCN2', 'OASL2', 'FES', 'LBP', 'ADGRG3', 'MFGE8', 'GLIPR2', 'IGTP', '2610035D17RIK', 'FUT7', 'CSGALNACT2', 'ORAI3', 'CD79B', 'MAPKAPK3', 'SIGIRR', 'TGM2', 'CSF2RB', 'OSBPL1A', 'TNFRSF21', 'MS4A6C', 'ADSSL1', 'ANXA2', 'DENND1C', 'KRT18', 'IGHM', 'SLC35B4', 'CAMK1', '1110059E24RIK', 'SNX4', 'GLB1']
gene_list_50_most_variable = [g.lower() for g in gene_list_50_most_variable]
gene_list_500_most_variable = ['meis1', 'rad51', 'gfi1b', 'spi1', 'hcfc1', 'mef2c', 'ikzf1', 'jun', 'irf1', 'myc', 'hdac2', 'stat3', 'notch1', 'fos', 'satb1', 'esr1', 'nfat5', 'ebf1', 'gata2', 'gfi1', 'ets1', 'cbx7', 'cbfb', 'foxo1', 'nr4a1', 'bmi1', 'irf8', 'elf4', 'bach2', 'bhlhe40', 'pou2af1', 'junb', 'nr1h3', 'stat6', 'tcf7', 'smad4', 'stat1', 'bcl11b', 'egr2', 'stat2', 'elf1', 'ctsg', 'elane', 'mpo', 'mpl', 'ccl9', 'dusp1', 'tyrobp', 'clec12a', 'ltb', 'ly6a', 'rgs1', 'igkc', 'procr', 'rab44', 'pdzk1ip1', 'sell', 'esam', 'f630028o10rik', 'myl10', 'maged1', 'trim47', 'cd48', 'tbxas1', 'bc035044', 'neurl3', 'cst7', 'hacd4', 'ms4a3', 'rtp4', 'pglyrp2', 'hk3', 'ifi44', 'nkg7', 'mmrn1', 'atp8b4', 'msmo1', 'gimap5', 'serpina3g', 'slc18a2', 'anxa3', 'nceh1', 'dennd2d', 'pcp4l1', 'gimap1', 'laptm4b', 'pqlc3', 'rhof', 'pear1', 'il21r', 'ly6c2', 'ldhb', 'ms4a6b', 'rab32', 'ap3s1', 'gbp2', 'padi4', 'myo1f', 'casp1', 'ctsl', 'alas1', 'itih5', 'emilin1', 'zfp386', 'tcp11l2', 'cish', 'vill', 'ppp1r15a', 'afap1l1', 'smpdl3a', 'tcn2', 'oasl2', 'fes', 'lbp', 'adgrg3', 'mfge8', 'glipr2', 'igtp', '2610035d17rik', 'fut7', 'csgalnact2', 'orai3', 'cd79b', 'mapkapk3', 'sigirr', 'tgm2', 'csf2rb', 'osbpl1a', 'tnfrsf21', 'ms4a6c', 'adssl1', 'anxa2', 'dennd1c', 'krt18', 'ighm', 'slc35b4', 'camk1', '1110059e24rik', 'snx4', 'glb1', 'anxa1', 'pstpip1', 'ptgs1', 'ctso', 'tor3a', 'plek', 'il15', 'cpt1a', 'tmem43', 'gng11', 'pafah2', 'cpa3', 'myo1g', 'sla', 'vps16', 'spns3', 'atg12', 'idh1', 'serpinb1a', 'nampt', 'rnf13', 'atp13a1', 'sfxn3', 'stx7', 'f11r', 'cpq', 'mfsd1', 'tcirg1', 'tespa1', 'ebi3', 'ube2l6', 'ica1', 'cd74', 'csad', 'vldlr', 'trim44', 'icam1', 'zfp35', 'grina', 'pak1', 'ctps2', 'colgalt1', 'slc28a2', 'flt3', 'csf3r', 'stxbp4', 'cd53', 'ric8a', 'ncf1', 'unc93b1', 'slc39a6', 'g6pdx', 'snap23', 'myct1', 'slco3a1', 'sqle', 'casp4', 'bap1', 'epsti1', 'gimap6', 'st8sia4', 'hells', 'dok2', 'gypc', 'mettl7a1', 'lgals8', 'gbp3', 'spns2', 'prps2', 'tie1', 'ythdf3', 'jak3', 'tmem206', 'gosr2', 'plac8', 'sdhaf2', 'myadm', 'gng12', 'slc2a1', 'cdt1', 'fcer1g', 'pfkp', 'pttg1ip', 'tfpi', 'lcp2', 'cln5', 'prss57', 'serpinb9', 'cd300a', 'stk25', 'uba3', 'rnase6', 'acot9', 'sh2d5', 'fdft1', 'elmo2', 'ctla2b', 'rnf115', 'atad1', 'zfp24', 'dhrs3', 'bcl2', 'pcmtd2', 'st6galnac4', 'arhgef18', 'zfp871', 'fuca2', 'fzr1', 'bace1', 'foxred1', 'irak3', 'plekhb2', 'abcd1', 'nde1', 'swap70', 'trim35', 'erlin1', 'dpp4', 'tmx3', 'nek9', 'trp53i11', 'zfp397', 'pacsin2', 'tyms', 'wars', 'ada', 'itpk1', 'scarf1', 'zfp62', 'man2a2', 'abcg2', 'crk', 'arhgap27', 'prim1', 'gyg', 'pold1', 'usp11', 'cd9', 'fig4', 'flot2', 'fads1', 'ahcyl1', 'arl6ip6', 'rgs2', 'dnajc5', 'pecam1', 'prodh', 'uhrf1', 'pqlc1', 'pim2', 'cdk19', 'cd82', 'ubap1', 'samd9l', 'maged2', 'inpp1', 'rnf125', 'usp8', 'gbp7', 'cers4', 'nfam1', 'ranbp3', 'mpnd', 'gcnt2', 'f2rl3', 'pld3', 'wdr5', 'milr1', 'lonp2', 'sgms1', 'mtap', 'itfg1', 'ap1g2', 'sdf2l1', 'panx1', 'tmem97', 'fgd2', 'erap1', 'il15ra', 'lyn', 'itm2a', 'poglut1', 'mtfr1', 'plppr3', 'rab31', 'gpr65', 'appl2', 'gmfb', 'capn7', 'cdca4', 'usp37', 'ppil4', 'cnot10', 'cerk', 'fam136a', 'exoc2', '1600014c10rik', 'plin3', 'slfn8', 'dhx58', 'inpp5b', 'phactr2', 'ube2g2', 'dhfr', 'akap8l', 'wdr4', 'dus1l', 'fam76b', 'mfsd14b', 'vps33b', 'ncln', 'tmem229b', 'tk2', 'mast3', 'emcn', 'paqr7', 'map4k2', 'wdr45b', 'rnasel', 'pmpca', 'ptdss1', 'ak3', 'relt', 'far1', 'slc19a1', 'lrmp', 'rgcc', 'man2b2', 'zfp451', 'casp2', 'ccdc50', 'plekho2', 'itsn1', 'prmt7', 'nup107', 'tars', 'klhl7', 'atp6ap2', 'calu', 'parp9', 'morc3', 'il12a', 'arap1', 'psat1', 'gusb', 'hars2', 'rasa4', 'tpst2', 'clcn4', 'tmem129', 'chmp3', 'psen1', 'ado', 'shmt1', 'ssh3', 'cstf2', 'ftsj3', 'abcb8', 'fam91a1', 'acox1', 'tmx4', 'pxk', 'trmt2a', 'slc30a5', 'myo5a', 'ldah', 'haus2', 'thumpd1', 'ndrg1', 'mis12', 'usp3', 'gsr', 'tmem71', 'zfp945', 'msh6', 'ppat', 'socs5', 'bag2', 'tfr2', 'cul2', 'dcaf13', 'metap1', 'bid', 'zfp367', 'prkab1', 'gm5111', 'cd69', 'apoe', 'trmt2b', 'ubr7', 'coa5', 'sesn1', 'golt1b', 'ccl4', 'armc8', 'hacd3', 'plaa', 'lig1', 'cyb5r1', 'chka', 'leprot', 'usp10', 'arid5a', 'tubgcp3', 'slc25a46', 'vav3', 'pno1', 'mrpl37', 'zfp157', 'calcrl', 'ccnb2', 'snx30', 'csgalnact1', 'tmx1', 'tbc1d9b', 'arl8b', 'fcho2', 'top2a', 'emid1', 'mad2l1', 'plxnc1', 'il27ra', 'galnt6', 'lonp1', 'arhgef6', 'iigp1', 'tmem164', 'igsf6', 'iars', 'gdi1', 'stard7', '4833439l19rik', 'sgpl1', 'wdr55', 'rangap1', 'xaf1', 'rassf4', 'prpf4', 'por', 'kat7', 'pcmtd1', 'rpa1', 'man1a', 'wdr75', 'tnip3', 'epb41l4b', 'hist1h2bc', 'tmem30a', 'eml4', 'rfc5', 'prep', 'hexa', 'fam168b', 'casp12', 'nudcd3', 'irgm2', 'arhgef39', 'unk', 'klhl24', 'crnkl1', 'slc6a6', 'mlkl', 'ctsf', 'rab37', 'gmcl1', 'trmt11', 'rfc4', 'ino80d', 'kif23', 'cd52', 'gmnn', 'ppp6c', 'agps', 'stt3b', 'arl11', 'nudt19', 'lztr1', 'gria3', 'neu1']
#Gene set to use for current analysis
gene_set_name = "Top_50_most_variable_genes" #For Neutrophils, modify this to Neutrophil
curr_gene_list = gene_list_50_most_variable #For neutrophil, modify this to gene_list_Neutrophil
gene_subset = [s + '_mRNA' for s in curr_gene_list]

t1 = 2
t2 = 4
t3 = 6
adata.var_names = pd.Index([g.lower() for g in adata.var_names])

#Subset the data into different time points - includes both barcoded and not barcoded cells (hence, all)
adata.obs['cell_id'] = adata.obs.index
adata_t1_all = adata[(adata.obs['Time point'] == t1)].copy()
adata_t2_all = adata[(adata.obs['Time point'] == t2)].copy()
adata_t3_all = adata[(adata.obs['Time point'] == t3)].copy()


# All cells at time t1 to calculate gene correlation
def make_all_cells_table(adata_t, timepoint, gene_subset, curr_gene_list):
    df = pd.DataFrame({
        'clone_id': adata_t.obs['clone_id'].values,
        'cell_id':  adata_t.obs['cell_id'].values,
    })
    df['pair_id']   = df['cell_id'].astype(str) + f"_single_{timepoint}"
    df['replicate'] = 1
    df['time_step'] = timepoint

    expr = pd.DataFrame(
        adata_t[df.cell_id, [g.lower() for g in curr_gene_list]].X.toarray(),
        columns=gene_subset,
        index=df.index
    )
    return pd.concat([df, expr], axis=1)

import pandas as pd

# verify
print(adata.var_names[:5])
gene_subset = [s + '_mRNA' for s in curr_gene_list]
t1_data_all_cells = make_all_cells_table(
    adata_t1_all, t1, gene_subset, curr_gene_list
)
t2_data_all_cells = make_all_cells_table(
    adata_t2_all, t2, gene_subset, curr_gene_list
)
t3_data_all_cells = make_all_cells_table(
    adata_t3_all, t3, gene_subset, curr_gene_list
)

#Identifying barcoded cells at time t1
adata_t1 = adata_t1_all[(adata_t1_all.obs['clone_id'] != -1)].copy()
adata_t1_clones_undiff = adata_t1[adata_t1.obs['Cell type annotation'] == 'Undifferentiated']

# Print the results
print(f"Number of barcoded cells: {adata_t1.shape[0]}")
print(f"Number of undifferentiated barcoded cells: {adata_t1_clones_undiff.shape[0]}")

adata_t2 = adata_t2_all[(adata_t2_all.obs['clone_id'] != -1)].copy()
adata_t3 = adata_t3_all[(adata_t3_all.obs['clone_id'] != -1)].copy()

# Initialize dictionary to store cell type counts and percentages for each clone_id
clone_cell_type = {}

# Loop through each unique clone_id
for clone_id in adata_t1_clones_undiff.obs.clone_id.unique():
    # Get unique cell types for t2 and t3 as lists
    unique_t2_cell_types = adata_t2[adata_t2.obs.clone_id == clone_id].obs['Cell type annotation'].unique().tolist()
    unique_t3_cell_types = adata_t3[adata_t3.obs.clone_id == clone_id].obs['Cell type annotation'].unique().tolist()

    # Concatenate the lists and convert to a set (to avoid duplicates)
    cell_type = set(unique_t2_cell_types + unique_t3_cell_types)
    if len(cell_type) < 1:
        continue

    # Get the number of cells for each cell type at t2 and t3
    t2_cell_counts = adata_t2[adata_t2.obs.clone_id == clone_id].obs['Cell type annotation'].value_counts()
    t3_cell_counts = adata_t3[adata_t3.obs.clone_id == clone_id].obs['Cell type annotation'].value_counts()

    # Initialize dictionary to store counts and percentages
    cell_type_info = {}

    # Count and calculate percentages for t2
    total_cells_t2 = len(adata_t2[adata_t2.obs.clone_id == clone_id])
    for cell in cell_type:
        t2_count = t2_cell_counts.get(cell, 0)
        t2_percentage = (t2_count / total_cells_t2) * 100 if total_cells_t2 > 0 else 0
        cell_type_info[cell] = {'t2_count': t2_count, 't2_percentage': t2_percentage}

    # Count and calculate percentages for t3
    total_cells_t3 = len(adata_t3[adata_t3.obs.clone_id == clone_id])
    for cell in cell_type:
        t3_count = t3_cell_counts.get(cell, 0)
        t3_percentage = (t3_count / total_cells_t3) * 100 if total_cells_t3 > 0 else 0
        if cell in cell_type_info:
            cell_type_info[cell].update({'t3_count': t3_count, 't3_percentage': t3_percentage})
        else:
            cell_type_info[cell] = {'t3_count': t3_count, 't3_percentage': t3_percentage}

    # Store the information in the dictionary
    clone_cell_type[clone_id] = cell_type_info

# Convert the dictionary into a pandas DataFrame for easier inspection
cell_type_df = pd.DataFrame.from_dict({(clone_id, cell_type): values
                                       for clone_id, clone_info in clone_cell_type.items()
                                       for cell_type, values in clone_info.items()},
                                      orient='index')


clone_dominant_cell_type = {}
clone_cell_type_composition = {}
count = 0
for clone_id, cell_type_info in clone_cell_type.items():

    # --- compute max at each time ---
    max_t2 = max(info["t2_percentage"] for info in cell_type_info.values())
    max_t3 = max(info["t3_percentage"] for info in cell_type_info.values())

    dom_t2 = sorted([
        ct for ct, info in cell_type_info.items()
        if info["t2_percentage"] == max_t2
    ])

    dom_t3 = sorted([
        ct for ct, info in cell_type_info.items()
        if info["t3_percentage"] == max_t3
    ])


    # --- choose timepoint ---
    if max_t3 > max_t2:
        chosen_types = dom_t3
        time_key = "t3_percentage"

    elif max_t2 > max_t3:
        # t2 wins unless it is purely Undifferentiated
        if dom_t2 == ["Undifferentiated"] and max_t3 > 0:
            count +=1
            chosen_types = dom_t3
            time_key = "t3_percentage"
        else:
            chosen_types = dom_t2
            time_key = "t2_percentage"

    else:  # equal max
        chosen_types = dom_t3
        time_key = "t3_percentage"

    # --- resolve ties ---
    if len(chosen_types) > 1:
        non_undiff = [ct for ct in chosen_types if ct != "Undifferentiated"]
        if len(non_undiff) > 0:
            chosen_types = non_undiff
        # else: keep Undifferentiated

    # --- final assignment (SCALAR ONLY) ---
    dominant_type = chosen_types[0] if len(chosen_types) > 0 else None
    clone_dominant_cell_type[clone_id] = dominant_type

    # --- composition string ---
    composition_parts = []
    for ct, info in cell_type_info.items():
        pct = info[time_key]
        if pct > 0:
            composition_parts.append(f"{ct}: {pct:.0f}%")

    clone_cell_type_composition[clone_id] = ", ".join(composition_parts)

# Now, assign the dominant cell type(s) to each clone in adata.obs
adata_t1_clones_undiff.obs["dominant_cell_type"] = None
adata_t1_clones_undiff.obs["cell_type_composition"] = None

for clone_id, dom_type in clone_dominant_cell_type.items():
    adata_t1_clones_undiff.obs.loc[
        adata_t1_clones_undiff.obs.clone_id == clone_id,
        "dominant_cell_type"
    ] = dom_type

    adata_t1_clones_undiff.obs.loc[
        adata_t1_clones_undiff.obs.clone_id == clone_id,
        "cell_type_composition"
    ] = clone_cell_type_composition[clone_id]

clone_to_dom = (
    adata_t1_clones_undiff.obs
    .dropna(subset=["dominant_cell_type"])
    .drop_duplicates(subset=["clone_id"])
    .set_index("clone_id")["dominant_cell_type"]
)

# --------------------------------------------------
# Get list of dominant cell types
# --------------------------------------------------
dominant_cell_types = sorted(clone_to_dom.unique())

# Get unique cell types INCLUDING None
cell_type_list = adata_t1_clones_undiff.obs['dominant_cell_type'].unique()

# Print the total number of cells
print(f"Number of cells: {adata_t1_clones_undiff.shape[0]}")

# Print the number of cell types and list them
print(f"Number of cell types (including None): {len(cell_type_list)}")
print(f"Cell types: {cell_type_list}")

# Count number of cells per dominant cell type INCLUDING None
cell_type_counts = (
    adata_t1_clones_undiff.obs['dominant_cell_type']
    .value_counts(dropna=False)
)

# Print counts
for cell_type, count in cell_type_counts.items():
    print(f"{cell_type}: {count}")

lymphoid_cells = ['Lymphoid']
adata_t1_clones_undiff_filter = adata_t1_clones_undiff[~adata_t1_clones_undiff.obs['dominant_cell_type'].isin(lymphoid_cells)]

#Twin pairs at each time point and across time point
use_undifferentiated = True
if use_undifferentiated:
    adata_t1 = adata_t1_clones_undiff_filter.copy()
else:
    adata_t1 = adata_t1_all[(adata_t1_all.obs['clone_id'] != -1)].copy()

adata_t2 = adata_t2_all[(adata_t2_all.obs['clone_id'] != -1)].copy()

#Remove any lymphoid-lineage cells
lymphoid_cells = ['Lymphoid', 'pDC']
adata_t2 = adata_t2[~adata_t2.obs['Cell type annotation'].isin(lymphoid_cells)].copy()
adata_t3 = adata_t3_all[(adata_t3_all.obs['clone_id'] != -1)].copy()
adata_t3 = adata_t3[~adata_t3.obs['Cell type annotation'].isin(lymphoid_cells)].copy()

# Save cell IDs in .obs
adata_t1.obs['cell_id'] = adata_t1.obs_names
adata_t2.obs['cell_id'] = adata_t2.obs_names
adata_t3.obs['cell_id'] = adata_t3.obs_names

# Pick subset of genes
gene_subset = [s + '_mRNA' for s in curr_gene_list]

# Create tables for t1, t2 and t3 twin pairs
for adata_t, timepoint in zip([adata_t1,adata_t2,adata_t3], ['t1','t2', 't3']):
    rows = []
    for clone_id, group in adata_t.obs.groupby('clone_id'):
        cells = group['cell_id'].tolist()
        pair_counter = 0
        for c1, c2 in itertools.combinations(cells, 2):
            pair_id = f"{clone_id}_p{pair_counter}_{timepoint}"
            rows.append({
                'clone_id': clone_id,
                'pair_id': pair_id,
                'cell_id': c1,
                'replicate': 1
            })
            rows.append({
                'clone_id': clone_id,
                'pair_id': pair_id,
                'cell_id': c2,
                'replicate': 2
            })
            pair_counter += 1

    if timepoint == 't1':
        t1_data = pd.DataFrame(rows)
    elif timepoint == 't2':
        t2_data = pd.DataFrame(rows)
    else:
        t3_data = pd.DataFrame(rows)

t1_data['time_step'] = np.repeat(t1, len(t1_data))
t2_data['time_step'] = np.repeat(t2, len(t2_data))
t3_data['time_step'] = np.repeat(t3, len(t3_data))

# t1_data[gene_subset] = adata_t1[t1_data.cell_id, curr_gene_list].X.toarray()
# t2_data[gene_subset] = adata_t2[t2_data.cell_id, curr_gene_list].X.toarray()
# t3_data[gene_subset] = adata_t3[t3_data.cell_id, curr_gene_list].X.toarray()

expr_t1 = pd.DataFrame(adata_t1[t1_data.cell_id, curr_gene_list].X.toarray(),
                        columns=gene_subset, index=t1_data.index)
t1_data = pd.concat([t1_data, expr_t1], axis=1)

expr_t2 = pd.DataFrame(adata_t2[t2_data.cell_id, curr_gene_list].X.toarray(),
                        columns=gene_subset, index=t2_data.index)
t2_data = pd.concat([t2_data, expr_t2], axis=1)

expr_t3 = pd.DataFrame(adata_t3[t3_data.cell_id, curr_gene_list].X.toarray(),
                        columns=gene_subset, index=t3_data.index)
t3_data = pd.concat([t3_data, expr_t3], axis=1)

# ### Create tables for across t twin pairs
across_t_clones = list(set(adata_t1.obs.clone_id).intersection(adata_t2.obs.clone_id))
adata_t1_sub = adata_t1[adata_t1.obs.clone_id.isin(across_t_clones)]
adata_t2_sub = adata_t2[adata_t2.obs.clone_id.isin(across_t_clones)]

rows_t1 = []
rows_t2 = []
for clone_id in across_t_clones:
    cells_t1 = adata_t1_sub[adata_t1_sub.obs.clone_id == clone_id].obs['cell_id'].tolist()
    cells_t2 = adata_t2_sub[adata_t2_sub.obs.clone_id == clone_id].obs['cell_id'].tolist()
    pair_counter = 0
    for cell_t1 in cells_t1:
        for cell_t2 in cells_t2:
            pair_id = f"{clone_id}_p{pair_counter}_across_t"
            rows_t1.append({
                'clone_id': clone_id,
                'pair_id': pair_id,
                'cell_id': cell_t1,
                'replicate': 1,
                'time_step': t1
            })
            rows_t2.append({
                'clone_id': clone_id,
                'pair_id': pair_id,
                'cell_id': cell_t2,
                'replicate': 2,
                'time_step': t2
            })

            pair_counter += 1

base_t1 = pd.DataFrame(rows_t1)
across_t_data_t1 = pd.concat([
    base_t1,
    pd.DataFrame(
        adata_t1[base_t1['cell_id'], curr_gene_list].X.toarray(),
        columns=gene_subset,
        index=base_t1.index
    )
], axis=1)

base_t2 = pd.DataFrame(rows_t2)
across_t_data_t2 = pd.concat([
    base_t2,
    pd.DataFrame(
        adata_t2[base_t2['cell_id'], curr_gene_list].X.toarray(),
        columns=gene_subset,
        index=base_t2.index
    )
], axis=1)

across_t_data = pd.concat([across_t_data_t1, across_t_data_t2])


print(f"Number of t1 twins: {int(t1_data.shape[0]/2)}")
print(f"Number of t2 twins: {int(t2_data.shape[0]/2)}")
print(f"Number of t3 twins: {int(t3_data.shape[0]/2)}")
print(f"Number of across t twins: {int(across_t_data_t1.shape[0])}")

# Drop column clone_id and rename pair_id to clone_id
t1_data.drop(columns=['clone_id'], inplace=True)
t1_data.rename(columns={'pair_id': 'clone_id'}, inplace=True)
t2_data.drop(columns=['clone_id'], inplace=True)
t2_data.rename(columns={'pair_id': 'clone_id'}, inplace=True)
t3_data.drop(columns=['clone_id'], inplace=True)
t3_data.rename(columns={'pair_id': 'clone_id'}, inplace=True)

across_t_data.drop(columns=['clone_id'], inplace=True)
across_t_data.rename(columns={'pair_id': 'clone_id'}, inplace=True)

t1_clones = t1_data.clone_id.values
t2_clones = t2_data.clone_id.values
t3_clones = t3_data.clone_id.values
across_t_clones = across_t_data.clone_id.values

# Subset directly
t1_twins = t1_data
t2_twins = t2_data
t3_twins = t3_data

# Across_t: pick exactly one random twin per clone_id
# One cell per clone at t1
across_t_twin1 = across_t_data[across_t_data.time_step == t1]
across_t_twin2 = across_t_data[across_t_data.time_step == t2]

# Reset index for cleanliness
t1_twins = t1_twins.reset_index(drop=True)
t2_twins = t2_twins.reset_index(drop=True)
t3_twins = t3_twins.reset_index(drop=True)
across_t_twin1 = across_t_twin1.reset_index(drop=True)
across_t_twin2 = across_t_twin2.reset_index(drop=True)

all_t1_measurements = (
    pd.concat([t1_twins, across_t_twin1], ignore_index=True)
      .drop_duplicates(subset="cell_id", keep="first")
)

all_t2_measurements = (
    pd.concat([t2_twins, across_t_twin2], ignore_index=True)
      .drop_duplicates(subset="cell_id", keep="first")
)

all_t3_measurements = t3_twins.drop_duplicates(subset="cell_id", keep="first")

# Sets of cell IDs
t1_twin_cells = set(t1_twins["cell_id"])
t1_across_cells = set(across_t_twin1["cell_id"])

t2_twin_cells = set(t2_twins["cell_id"])
t2_across_cells = set(across_t_twin2["cell_id"])

# Overlaps
overlap_t1 = t1_twin_cells & t1_across_cells
overlap_t2 = t2_twin_cells & t2_across_cells

print("T1 overlap (t1 ∩ across):", len(overlap_t1))
print("T2 overlap (twin ∩ across):", len(overlap_t2))

# Define input parameters
plot_correlation_matrices_as_heatmap = False
have_any_output = False
p_val_threshold_scrambled_gene_correlation = 0.02
show_scrambled_distribution_gene_correlation = False
z_score_threshold_two_states = 10
n_shuffles=10000
#
## # --- Step 1: Pairwise gene-gene correlations at t1: day 2 ---
#pairwise_gene_gene_correlation_matrix_t1 = calculate_pairwise_gene_gene_correlation_matrix(
#    all_t1_measurements, curr_gene_list
#)
#print(pairwise_gene_gene_correlation_matrix_t1)
#no_regulation_t1, potential_regulation_t1, threshold, _ = check_gene_gene_correlation_threshold(
#    all_t1_measurements, pairwise_gene_gene_correlation_matrix_t1, curr_gene_list, n_shuffles = n_shuffles, use_scramble = True, p_val_threshold = p_val_threshold_scrambled_gene_correlation, verbose = show_scrambled_distribution_gene_correlation, n_cores_to_use=50
#)
#
#pairwise_gene_gene_correlation_matrix_t1.to_csv("pairwise_gene_gene_correlation_matrix_t1.csv")
#
## --- Step 1: Pairwise gene-gene correlations at t2: day 4 ---
#pairwise_gene_gene_correlation_matrix_t2 = calculate_pairwise_gene_gene_correlation_matrix(
#    all_t2_measurements, curr_gene_list
#)
#no_regulation_t2, potential_regulation_t2, _, _= check_gene_gene_correlation_threshold(
#    all_t2_measurements, pairwise_gene_gene_correlation_matrix_t2, curr_gene_list, use_scramble = True, p_val_threshold = p_val_threshold_scrambled_gene_correlation, verbose = show_scrambled_distribution_gene_correlation,  n_cores_to_use=50
#)
#
#pairwise_gene_gene_correlation_matrix_t2.to_csv("pairwise_gene_gene_correlation_matrix_t2.csv")
#
#
## --- Step 1: Pairwise gene-gene correlations at
## 
##  t3: day 6 ---
#pairwise_gene_gene_correlation_matrix_t3 = calculate_pairwise_gene_gene_correlation_matrix(
#    t3_data_all_cells, curr_gene_list
#)
#no_regulation_t3, potential_regulation_t3, _, _ = check_gene_gene_correlation_threshold(
#    t3_data_all_cells, pairwise_gene_gene_correlation_matrix_t3, curr_gene_list, use_scramble = True, p_val_threshold = p_val_threshold_scrambled_gene_correlation, verbose = show_scrambled_distribution_gene_correlation,  n_cores_to_use=50
#)
#
#if plot_correlation_matrices_as_heatmap:
#    plot_matrix_as_heatmap(corr_matrix=pairwise_gene_gene_correlation_matrix_t3, gene_list=curr_gene_list, no_regulation=no_regulation_t3, potential_regulation=potential_regulation_t3,
#        title=f"Gene-gene correlations for {gene_set_name}", add_gene_labels=True, add_time=False, gray_out_no_reg=False, black_out_self = True
#    )
#
#pairwise_gene_gene_correlation_matrix_t3.to_csv("pairwise_gene_gene_correlation_matrix_t3.csv")
#
## === Combine and save all timepoint results ===
#rows = []
#
#for tp, (no_reg, pot_reg, corr_mat) in {
#    "t1": (no_regulation_t1, potential_regulation_t1, pairwise_gene_gene_correlation_matrix_t1),
#    "t2": (no_regulation_t2, potential_regulation_t2, pairwise_gene_gene_correlation_matrix_t2),
#    "t3": (no_regulation_t3, potential_regulation_t3, pairwise_gene_gene_correlation_matrix_t3),
#}.items():
#
#    all_pairs = set(tuple(sorted(p)) for p in no_reg + pot_reg)
#
#    for g1, g2 in all_pairs:
#        # lookup correlation (try both orders)
#        if g1 in corr_mat.index and g2 in corr_mat.columns:
#            corr_val = corr_mat.loc[g1, g2]
#        elif g2 in corr_mat.index and g1 in corr_mat.columns:
#            corr_val = corr_mat.loc[g2, g1]
#        else:
#            corr_val = None
#
#        pair_sorted = tuple(sorted((g1, g2)))
#        if pair_sorted in [tuple(sorted(p)) for p in pot_reg]:
#            category = "potential_regulation"
#        elif pair_sorted in [tuple(sorted(p)) for p in no_reg]:
#            category = "no_regulation"
#        else:
#            category = "uncategorized"
#
#        rows.append([g1, g2, corr_val, category, tp])
#
## Create DataFrame
#df = pd.DataFrame(rows, columns=["gene_1", "gene_2", "correlation", "category", "timepoint"])
#
## Define output filename with timestamp
#outfile = path_to_plot_data / f"gene_pair_results_{gene_set_name}.csv"
#
## Save file
#df.to_csv(outfile, index=False)
#
## Print confirmation with readable date/time
#print(f"Saved {len(df)} pairs to {outfile.name}")
#
## === Load saved CSV ===
df = pd.read_csv(path_to_plot_data / f"gene_pair_results_{gene_set_name}.csv")
#
## === Reconstruct lists and matrices ===
no_regulation = {}
potential_regulation = {}
pairwise_gene_gene_correlation_matrix = {}

# for tp, sub in df.groupby("timepoint"):
#     # Lists of tuples
#     no_regulation[tp] = list(zip(sub.loc[sub["category"] == "no_regulation", "gene_1"],
#                                  sub.loc[sub["category"] == "no_regulation", "gene_2"]))
#     potential_regulation[tp] = list(zip(sub.loc[sub["category"] == "potential_regulation", "gene_1"],
#                                         sub.loc[sub["category"] == "potential_regulation", "gene_2"]))
#     # Pivot to matrix
#     corr_mat = sub.pivot_table(index="gene_1", columns="gene_2", values="correlation")
#     # make symmetric since we only stored one order per pair
#     corr_mat = corr_mat.combine_first(corr_mat.T)
#     pairwise_gene_gene_correlation_matrix[tp] = corr_mat

for tp, sub in df.groupby("timepoint"):
    no_regulation[tp] = list(zip(sub.loc[sub["category"] == "no_regulation", "gene_1"],
                                 sub.loc[sub["category"] == "no_regulation", "gene_2"]))
    potential_regulation[tp] = list(zip(sub.loc[sub["category"] == "potential_regulation", "gene_1"],
                                        sub.loc[sub["category"] == "potential_regulation", "gene_2"]))

    # Build symmetric matrix explicitly
    all_genes = sorted(set(sub["gene_1"]).union(set(sub["gene_2"])))
    mat = pd.DataFrame(np.nan, index=all_genes, columns=all_genes)
    for _, row in sub.iterrows():
        g1, g2, val = row["gene_1"], row["gene_2"], row["correlation"]
        mat.loc[g1, g2] = val
        mat.loc[g2, g1] = val  # fill both directions explicitly
    pairwise_gene_gene_correlation_matrix[tp] = mat

## === Extract t1, t2, t3 structures ===
no_regulation_t1 = no_regulation["t1"]
no_regulation_t2 = no_regulation["t2"]
no_regulation_t3 = no_regulation["t3"]

potential_regulation_t1 = potential_regulation["t1"]
potential_regulation_t2 = potential_regulation["t2"]
potential_regulation_t3 = potential_regulation["t3"]

pairwise_gene_gene_correlation_matrix_t1 = pairwise_gene_gene_correlation_matrix["t1"]
pairwise_gene_gene_correlation_matrix_t2 = pairwise_gene_gene_correlation_matrix["t2"]
pairwise_gene_gene_correlation_matrix_t3 = pairwise_gene_gene_correlation_matrix["t3"]

# Optional sanity check
for tp in ["t1", "t2", "t3"]:
    print(f"{tp}: {len(no_regulation[tp])} no-reg pairs, "
          f"{len(potential_regulation[tp])} potential-reg pairs, "
          f"matrix {pairwise_gene_gene_correlation_matrix[tp].shape}")

# --- Step 2: Twin/random correlations at day 2 ---
#twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1 = calculate_twin_random_pair_correlations(
#    all_t1_measurements, t1_twins, curr_gene_list
#)
## print(twin_pair_correlation_matrix_t2)
#if plot_correlation_matrices_as_heatmap:
#    plot_matrix_as_heatmap(corr_matrix=twin_pair_correlation_matrix_t1, gene_list=curr_gene_list, no_regulation=no_regulation_t1, potential_regulation=potential_regulation_t1,
#        title=f"Twin pair correlations at time d{t1}", add_gene_labels=True, add_time=False, gray_out_no_reg=True, black_out_self=True
#    )
#
#    plot_matrix_as_heatmap(corr_matrix=random_pair_correlation_matrix_t1, gene_list=curr_gene_list, no_regulation=no_regulation_t1, potential_regulation=potential_regulation_t1,
#        title=f"Random pair correlations across both time points", add_gene_labels=True, add_time=False, time=[t1], gray_out_no_reg=True, black_out_self=True
#    )
#
## --- Step 3: Classify regulation type: single-state vs multiple-states ---
#multiple_states_gene_pairs_t1, single_state_regulation_t1 = differentiate_single_state_reg_and_multiple_states(
#    all_t1_measurements, potential_regulation_t1, twin_pair_correlation_matrix_t1, random_pair_correlation_matrix_t1, curr_gene_list, z_score_threshold=z_score_threshold_two_states
#)
#print(multiple_states_gene_pairs_t1, single_state_regulation_t1)
#
## --- Step 2: Twin/random correlations at day 4 ---
#twin_pair_correlation_matrix_t2, random_pair_correlation_matrix_t2 = calculate_twin_random_pair_correlations(
#    all_t2_measurements, t2_twins, curr_gene_list
#)
## print(twin_pair_correlation_matrix_t2)
#if plot_correlation_matrices_as_heatmap:
#    plot_matrix_as_heatmap( corr_matrix=twin_pair_correlation_matrix_t2, gene_list=curr_gene_list, no_regulation=no_regulation_t2, potential_regulation=potential_regulation_t2,
#        title=f"Twin pair correlations at time d{t2}", add_gene_labels=True, add_time=False, time=[t2], gray_out_no_reg=True, vmin = -0.4, vmax=0.4, black_out_self=True
#    )
#
#    plot_matrix_as_heatmap( corr_matrix=random_pair_correlation_matrix_t2, gene_list=curr_gene_list, no_regulation=no_regulation_t2, potential_regulation=potential_regulation_t2,
#        title=f"Random pair correlations across both time points", add_gene_labels=True, add_time=False, time=[t2], gray_out_no_reg=True, vmin = -0.4, vmax=0.4, black_out_self=True
#    )
#
## # --- Step 3: Classify regulation type: single-state vs multiple-states ---
#multiple_states_gene_pairs_t2, single_state_regulation_t2 = differentiate_single_state_reg_and_multiple_states(
#    all_t2_measurements, potential_regulation_t2, twin_pair_correlation_matrix_t2, random_pair_correlation_matrix_t2, curr_gene_list, z_score_threshold=z_score_threshold_two_states
#)
#print(multiple_states_gene_pairs_t2, single_state_regulation_t2)
#
#
#if len(multiple_states_gene_pairs_t1) > 0:
#
#    multiple_states_no_reg, multiple_states_and_reg = identify_reg_if_multiple_states(
#        twin_pair_correlation_matrix_t1,twin_pair_correlation_matrix_t2,random_pair_correlation_matrix_t1,
#        random_pair_correlation_matrix_t2,multiple_states_gene_pairs_t1,curr_gene_list
#        )
#else:
#    multiple_states_no_reg, multiple_states_and_reg = [], []
#
## ----------------------------------
## Collect all classified pairs
## ----------------------------------
#scenario_pair_lists_t2 = {
#    "single-state, no regulation": no_regulation_t2,
#    "single-state, regulation": single_state_regulation_t2,
#    "multiple states":multiple_states_gene_pairs_t2
#}
#
#records_t2 = []
#
#for scenario, pairs in scenario_pair_lists_t2.items():
#    for g1, g2 in pairs:
#        g1, g2 = sorted((g1, g2))   # normalize
#        records_t2.append({
#            "gene_1": g1,
#            "gene_2": g2,
#            "scenario": scenario,
#            "timepoint": "t2"        # optional but strongly recommended
#        })
#
#df_pair_classification_t2 = pd.DataFrame(records_t2)
#
## sanity check
#assert not df_pair_classification_t2.duplicated(
#    ["gene_1", "gene_2", "timepoint"]
#).any()
#
#df_pair_classification_t2.to_csv(f"{path_to_plot_data}/all_gene_pair_classification_{gene_set_name}_day_4.csv")
#
## ----------------------------------
## Collect all classified pairs
## ----------------------------------
#scenario_pair_lists_t1 = {
#    "single-state, no regulation": no_regulation_t1,
#    "single-state, regulation": single_state_regulation_t1,
#    "multiple states": multiple_states_gene_pairs_t1
#}
#
#records_t1 = []
#
#for scenario, pairs in scenario_pair_lists_t1.items():
#    for g1, g2 in pairs:
#        g1, g2 = sorted((g1, g2))   # normalize
#        records_t1.append({
#            "gene_1": g1,
#            "gene_2": g2,
#            "scenario": scenario,
#            "timepoint": "t1"        # optional but strongly recommended
#        })
#
#df_pair_classification_t1 = pd.DataFrame(records_t1)
#
## sanity check
#assert not df_pair_classification_t1.duplicated(
#    ["gene_1", "gene_2", "timepoint"]
#).any()
#
#df_pair_classification_t1.to_csv(f"{path_to_plot_data}/all_gene_pair_classification_{gene_set_name}.csv")
df_pair_classification_t1 = pd.read_csv(f"{path_to_plot_data}/all_gene_pair_classification_{gene_set_name}.csv")
df_pair_classification_t2 = pd.read_csv(f"{path_to_plot_data}/all_gene_pair_classification_{gene_set_name}_day_4.csv")
consistent_pairs = (
    set(potential_regulation_t1)
    & set(potential_regulation_t2)
    & set(potential_regulation_t3)
)

consistent_corr = []

for g1, g2 in consistent_pairs:
    # optional: enforce ordering to avoid (A,B) vs (B,A)
    g1, g2 = sorted((g1, g2))

    c1 = pairwise_gene_gene_correlation_matrix_t1.loc[g1, g2]
    c2 = pairwise_gene_gene_correlation_matrix_t2.loc[g1, g2]
    c3 = pairwise_gene_gene_correlation_matrix_t3.loc[g1, g2]

    # ignore zero / NaN correlations
    if any(np.isnan([c1, c2, c3])) or any(c == 0 for c in (c1, c2,c3)):
        continue

    if np.sign(c1) == np.sign(c2) == np.sign(c3):
        consistent_corr.append((g1, g2))
consistent_corr = sorted(consistent_corr)
print(len(consistent_corr))
consistent_corr

# --- Step 5: Infer directionality of single-state interactions ---
infer_direction_for_which_edges = "all-potential-regulation"
p_value_threshold_cross_correlation = 0.01
n_cores = 50


if infer_direction_for_which_edges == "single-state" :
    if len(single_state_regulation_t1) > 0:
        bidirectional_pairs = {(a, b) for (a, b) in single_state_regulation_t1} | \
                  {(b, a) for (a, b) in single_state_regulation_t1}
        # Add self-pairs
        genes = {g for pair in single_state_regulation_t1 for g in pair}
        self_pairs = {(g, g) for g in genes}
        # Final
        all_gene_pairs = bidirectional_pairs | self_pairs
        all_gene_pairs = list(all_gene_pairs)
        direction_matrix = get_cross_correlations(across_t_twin1, across_t_twin2, gene_pairs=all_gene_pairs)

        final_directed_edges = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True)

elif infer_direction_for_which_edges == "all-potential-regulation":
        if len(consistent_corr) > 0:
                combined_list = consistent_corr
                bidirectional_pairs = {(a, b) for (a, b) in combined_list} | \
                      {(b, a) for (a, b) in combined_list}
                genes = {g for pair in combined_list for g in pair}
                self_pairs = {(g, g) for g in genes}

                # Final
                all_gene_pairs_all_reg = bidirectional_pairs | self_pairs
                all_gene_pairs_all_reg = list(all_gene_pairs_all_reg)

                direction_matrix = get_cross_correlations(across_t_twin1, across_t_twin2, gene_pairs=all_gene_pairs_all_reg)
                final_directed_edges = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs_all_reg, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True)
        else:
                final_directed_edges = []
                direction_matrix = pd.DataFrame(
                    np.zeros((len(curr_gene_list), len(curr_gene_list))),
                    index=curr_gene_list,
                    columns=curr_gene_list
                )
else:
    print("running all pairs")
    direction_matrix = get_cross_correlations(across_t_twin1, across_t_twin2, gene_pairs=all_gene_pairs)
    final_directed_edges = identify_actual_directed_edges(across_t_twin1, across_t_twin2, direction_matrix, gene_pairs=all_gene_pairs, threshold = p_value_threshold_cross_correlation, n_cores_to_use = n_cores, verbose = True)
print(final_directed_edges)
# print(pre_threshold_direction_matrix)
direction_matrix = direction_matrix.reindex(
index=curr_gene_list,
columns=curr_gene_list,
fill_value=0
)
unfiltered_direction_matrix = direction_matrix
if final_directed_edges:
    for i in direction_matrix.index:
        for j in direction_matrix.columns:
            if i != j and (i, j) not in final_directed_edges:
                direction_matrix.loc[i,j] = 0

import json
multiple_states = multiple_states_gene_pairs_t1 + multiple_states_gene_pairs_t2

directional_gene_correlation_data = {
    "gene_list": list(curr_gene_list),
    "corr_matrix": direction_matrix.values.tolist(),
    "no_regulation_pairs": [list(p) for p in no_regulation_t1],
    "final_directed_edges": [list(p) for p in final_directed_edges],
    "multi_state_regulation_pairs": [list(p) for p in multiple_states]
}

with open(f"{path_to_plot_data}/directional_gene_correlation_data.json", "w") as f:
    json.dump(directional_gene_correlation_data, f, indent=2)
