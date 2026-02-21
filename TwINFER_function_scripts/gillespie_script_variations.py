#Updates
# # Optimized Gillespie-SSA Simulation Pipeline
# %% Input utilities
import os
import uuid
import json
from datetime import datetime
import re
import numpy as np
import pandas as pd
import numba
from numba import prange, set_num_threads, get_num_threads
from tqdm.auto import tqdm
import time
import concurrent.futures
import argparse
import gc 
from numba.typed import List
from joblib import Parallel, delayed
from tqdm import tqdm
import glob
import ast
# %% Input utilities

def read_input_matrix(path_to_matrix: str) -> (int, np.ndarray):
    """
    Reads an input matrix from a specified file path and returns its dimensions and content.

    Args:
        path_to_matrix (str): The file path to the matrix file. The file should contain
                              a comma-separated matrix of integers.

    Returns:
        tuple: A tuple containing:
            - int: The number of rows in the matrix.
            - np.ndarray: The matrix as a NumPy array. If the matrix is a single value,
                          it is reshaped into a 1x1 array.

    Raises:
        ValueError: If the file cannot be loaded.
    """
    try:
        matrix = np.loadtxt(path_to_matrix, dtype=int, delimiter=',')
        if matrix.ndim == 0:
            matrix = matrix.reshape((1,1))
        return matrix.shape[0], matrix
    except Exception as e:
        raise ValueError(f"Error loading matrix from {path_to_matrix}: {e}")


def assign_parameters_to_genes(csv_path, gene_list, rows=None):
    """
    Assigns parameters to a list of genes based on values from a CSV file.

    This function reads a CSV file containing parameter values, selects rows 
    either randomly or based on the provided indices, and assigns the parameters 
    to the specified genes. It calculates additional parameters such as 
    degradation rates for mRNA and protein based on their respective half-lives.

    Args:
        csv_path (str): Path to the CSV file containing parameter values. 
                        The file should have columns including 'mrna_half_life' 
                        and 'protein_half_life'.
        gene_list (list): List of gene names to which parameters will be assigned.
        rows (list, optional): List of row indices to select from the CSV file. 
                               If None, rows are randomly selected with replacement. 
                               Defaults to None.

    Returns:
            param_dict (dict): A dictionary mapping parameter names (formatted 
                                 as "{parameter_gene}") to their values.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        raise ValueError(f"Parameter csv file not found at path: {csv_path}")
    n = len(gene_list)

    param_dict = {}
    for i,row in enumerate(rows):
        if i >= n:
            print(f"Using the first {n} parameters out of {len(rows)} rows")
            break
        gene = gene_list[i]
        if int(row) in df.index:
            vals = df.loc[int(row)].copy()
        else:
            raise KeyError(f"Row index {int(row)} not found in the DataFrame.")
        if 'k_off' not in vals:
            if "pi_on" in vals:
                vals['k_off'] = vals['k_on']*(1/vals['pi_on'] - 1)
            else:
                raise ValueError("Either pi_on or k_off must be specified!")
        vals["k_deg_mRNA"] = np.log(2)/vals["mrna_half_life"]
        vals["k_deg_protein"] = np.log(2)/vals["protein_half_life"]
        vals.drop(["mrna_half_life","protein_half_life"],axis=0,inplace=True,errors="ignore")
        for k, v in vals.items():
            if "_to_" in k:
                param_dict[f"{{{k}}}"] = float(v)  # Interaction parameter: keep as is
            else:
                param_dict[f"{{{k}_{gene}}}"] = float(v)  # Gene-specific parameter
    return param_dict

def _add_additive_per_edge_reaction(reactions, connectivity_matrix, gene_list, j, curr_gene):
    """
    New function for opposing-sign 2-regulator or >2 regulator case.
    Uses per-edge k_add. Each regulator contributes independently.
    """
    regulators_index = np.where(connectivity_matrix[:, j] != 0)[0]
    for i in regulators_index:
        regulator = gene_list[i]
        sign = int(np.sign(connectivity_matrix[i, j]))
        edge = f"{regulator}_to_{curr_gene}"
        expr = (
            f"(({sign}*{{k_add_{edge}}})"
            f"*({regulator}_protein**{{n_{edge}}})"
            f"/({{K_{edge}}}**{{n_{edge}}}+{regulator}_protein**{{n_{edge}}}))"
            f"*{curr_gene}_I"
        )
        reactions.append({"species1": f"{curr_gene}_A", "change1": 1,
                          "species2": f"{curr_gene}_I", "change2": -1,
                          "propensity": expr, "time": "-"})

def generate_reaction_network_from_matrix(connectivity_matrix: np.ndarray,
                                          combinatorial_interaction_type:str = "additive"):
    """
    Generate a reaction network from a given connectivity matrix.

    This function constructs a reaction network based on gene interactions defined 
    in the input connectivity matrix. It generates reactions for gene activation/inactivation, 
    regulation, mRNA production/degradation, and protein production/degradation 
    for each gene in the network.

    Args:
        connectivity_matrix (np.ndarray): A square matrix representing gene interactions. 
            Each element connectivity_matrix[i, j] indicates the regulatory effect of gene i 
            on gene j. Positive values represent activation, negative values represent 
            repression, and zero indicates no interaction.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - reactions_df (pd.DataFrame): A DataFrame containing the reaction network. 
              Each row represents a reaction with the following columns:
                - 'species1': The species involved in the reaction.
                - 'change1': The change in the count of 'species1'.
                - 'species2': The second species involved in the reaction (if applicable).
                - 'change2': The change in the count of 'species2'.
                - 'time': Placeholder for reaction time (currently set to "-").
                - 'propensity': The propensity function for the reaction.
            - gene_list (List[str]): A list of gene names generated from the connectivity matrix.

    Notes:
        - The propensity functions for reactions are defined using a set of predefined templates.
        - Parameters for each reaction are dynamically generated based on the gene and interaction 
          matrix information.
        - The function aggregates reactions with identical species and changes into a single row 
          with combined propensity functions.
    """
    n_genes = connectivity_matrix.shape[0]
    gene_list = [f"gene_{i+1}" for i in range(n_genes)]
    prop = {
        "activation": "{k_on}*{curr_gene}_I",
        "inactivation": "{k_off}*{curr_gene}_A",
        "mRNA_prod": "{k_prod_mRNA}*{curr_gene}_A",
        "mRNA_deg": "{k_deg_mRNA}*{curr_gene}_mRNA",
        "protein_prod": "{k_prod_protein}*{curr_gene}_mRNA",
        "protein_deg": "{k_deg_protein}*{curr_gene}_protein"
        }
    reactions = []
    for j, curr_gene in enumerate(gene_list):
            param = lambda k: f"{{{k}_{curr_gene}}}"
            # baseline activation
            expr = prop["activation"].replace("{k_on}", param("k_on")).replace("{curr_gene}", curr_gene)
            reactions.append({"species1":f"{curr_gene}_A","change1":1,
                            "species2":f"{curr_gene}_I","change2":-1,
                            "propensity":expr,"time":"-"})

            # inactivation
            expr = prop["inactivation"].replace("{k_off}",param("k_off")).replace("{curr_gene}",curr_gene)
            reactions.append({"species1":f"{curr_gene}_I","change1":1,
                            "species2":f"{curr_gene}_A","change2":-1,
                            "propensity":expr,"time":"-"})
            
            # production/degradation
            for label,suffix,chg in [
                ("mRNA_prod","mRNA",1),("mRNA_deg","mRNA",-1),
                ("protein_prod","protein",1),("protein_deg","protein",-1)
            ]:
                expr = prop[label].replace("{curr_gene}",curr_gene)
                for k in ["k_prod_mRNA","k_deg_mRNA","k_prod_protein","k_deg_protein"]:
                    expr = expr.replace(f"{{{k}}}",param(k))
                reactions.append({"species1":f"{curr_gene}_{suffix}","change1":chg,
                                "species2":"-","change2":"-",
                                "propensity":expr,"time":"-"})
            
            # regulation logic:
                # If there is only one regulator, it should be basic sign*k_add*(TF^n/(K^n + TF^n))
                # If there are 2 regulators, if they are both positive or both negative, depending on the combinatorial interaction specificied - specify r1, r2 and r12 accordingly
                # If there are 2 regulators with opposing signs or more than 2 regulators, only additive is possible
                #k_add is per target, not per edge - so this needs to be updated accordingly - not a matrix, it is a list#TODO

            regulators_index = np.where(connectivity_matrix[:,j]!=0)[0]
            if len(regulators_index) == 0:
                pass
            elif len(regulators_index) == 1:
                #There is no combinatorial regulation
                prop["regulatory_single"] = "(({sign}*{k_add})*({tf}_protein**{n})/({K}**{n}+{tf}_protein**{n}))*{curr_gene}_I"
                i = regulators_index[0]
                regulator = gene_list[i]
                sign = int(np.sign(connectivity_matrix[i,j]))
                edge = f"{regulator}_to_{curr_gene}"
                expr = prop["regulatory_single"]\
                    .replace("{sign}",str(sign))\
                    .replace("{k_add}",f"{{k_add_{curr_gene}}}")\
                    .replace("{n}",f"{{n_{edge}}}")\
                    .replace("{K}",f"{{K_{edge}}}")\
                    .replace("{tf}",regulator)\
                    .replace("{curr_gene}",curr_gene)
                reactions.append({"species1":f"{curr_gene}_A","change1":1,
                                "species2":f"{curr_gene}_I","change2":-1,
                                "propensity":expr,"time":"-"})
            
            elif len(regulators_index) == 2:
                prop["regulatory_comb"] = """
                                        (({sign}*{k_add}*{pre_factor}) * (
                                            {r1} * ({tf1}_protein**{n1}) / ({K1}**{n1})
                                        + {r2} * ({tf2}_protein**{n2}) / ({K2}**{n2})
                                        + {r12} * (
                                                ({tf1}_protein**{n1}) * ({tf2}_protein**{n2})
                                            ) / (
                                                ({K1}**{n1}) * ({K2}**{n2})
                                            )
                                        )) / (
                                            1
                                        + ({tf1}_protein**{n1}) / ({K1}**{n1})
                                        + ({tf2}_protein**{n2}) / ({K2}**{n2})
                                        + (
                                                ({tf1}_protein**{n1}) * ({tf2}_protein**{n2})
                                            ) / (
                                                ({K1}**{n1}) * ({K2}**{n2})
                                            )
                                        ) * {curr_gene}_I
                                    """

                i1, i2 = regulators_index
                sign1 = int(np.sign(connectivity_matrix[i1, j]))
                sign2 = int(np.sign(connectivity_matrix[i2, j])) 

                if sign1 == sign2:
                    #Can have different combinatorial regulations
                    if combinatorial_interaction_type == "OR":
                        print(f"Using OR logic for gene {curr_gene}")
                        r1 = "1"
                        r2 = "1"
                        r12 = "1"
                        pre_factor = "(2.0/3.0)"

                    elif combinatorial_interaction_type == "AND":
                        print(f"Using AND logic for gene {curr_gene}")
                        r1 = "0"
                        r2 = "0"
                        r12 = "1"
                        pre_factor = "(2.0)"

                    elif combinatorial_interaction_type == "additive":
                        print(f"Using additive logic for gene {curr_gene}")
                        r1 = "1"
                        r2 = "1"
                        r12 = "2"
                        pre_factor = "(1.0)"

                    missing = [v for v in ("r1", "r2", "r12") if v not in locals()]

                    if missing:
                        raise NameError(
                            f"Missing combinatorial parameters for {curr_gene}: {missing}"
                        )

                    #TODO make r1, r2 and r12 more generalizable and input from user
                    reg1 = gene_list[i1]
                    reg2 = gene_list[i2]

                    edge1 = f"{reg1}_to_{curr_gene}"
                    edge2 = f"{reg2}_to_{curr_gene}"
                    expr = prop["regulatory_comb"]\
                            .replace("{sign}", str(sign1))\
                            .replace("{pre_factor}", str(pre_factor))\
                            .replace("{k_add}", f"{{k_add_{curr_gene}}}")\
                            .replace("{tf1}", reg1)\
                            .replace("{tf2}", reg2)\
                            .replace("{n1}", f"{{n_{edge1}}}")\
                            .replace("{n2}", f"{{n_{edge2}}}")\
                            .replace("{K1}", f"{{K_{edge1}}}")\
                            .replace("{K2}", f"{{K_{edge2}}}")\
                            .replace("{r1}", r1)\
                            .replace("{r2}", r2)\
                            .replace("{r12}", r12)\
                            .replace("{curr_gene}", curr_gene)
                    reactions.append({
                            "species1": f"{curr_gene}_A",
                            "change1": 1,
                            "species2": f"{curr_gene}_I",
                            "change2": -1,
                            "propensity": expr,
                            "time": "-"
                        })
                else:
                    print(f"Gene {curr_gene} has opposing-sign regulators — using per-edge additive logic.")
                    _add_additive_per_edge_reaction(reactions, connectivity_matrix, gene_list, j, curr_gene)
            else:
                print(f"Gene {curr_gene} has {len(regulators_index)} regulators — using per-edge additive logic.")
                _add_additive_per_edge_reaction(reactions, connectivity_matrix, gene_list, j, curr_gene)        
                    
    df = pd.DataFrame(reactions)
    df['propensity'] = df['propensity'].astype(str)
    reactions_df = (
        df.assign(propensity=df['propensity'].str.strip())
        .query("propensity != ''")
        .groupby(['species1','change1','species2','change2','time'])['propensity']
        .agg(lambda x: ' + '.join(x) if len(x) > 1 else x.iloc[0])
        .reset_index()
    )                                   
    return reactions_df, gene_list                                      

def generate_initial_state_from_genes(gene_list):
    """
    Generate the initial state for a list of genes.

    This function creates a DataFrame representing the initial state of species
    associated with each gene in the provided list. For each gene, the following
    species are initialized:
    - `<gene>_A`: Active state, initialized with a count of 0.
    - `<gene>_I`: Inactive state, initialized with a count of 1.
    - `<gene>_mRNA`: Messenger RNA, initialized with a count of 0.
    - `<gene>_protein`: Protein, initialized with a count of 0.

    Args:
        gene_list (list of str): A list of gene names for which the initial states
                                 are to be generated.

    Returns:
        pandas.DataFrame: A DataFrame containing the initial states of the species
                          for each gene. Each row represents a species with its
                          name (`species`) and initial count (`count`).
    """
    states = []
    for g in gene_list:
        states += [
            {"species":f"{g}_A","count":0},
            {"species":f"{g}_I","count":1},
            {"species":f"{g}_mRNA","count":0},
            {"species":f"{g}_protein","count":0},
        ]
    return pd.DataFrame(states)

def assign_k_values_matrix(param_dict, connectivity_matrix, gene_list, K_to_use):
    """
    Assign k-values for every regulatory interaction using a K matrix.

    Parameters
    ----------
    gene_list : list of gene names
    connectivity_matrix : numpy array (n_genes x n_genes)
        Nonzero entries indicate regulatory edges.
    param_dict : dict
        Dictionary where {k_src_to_tgt} keys will be replaced.
    K_to_use : numpy array (n_genes x n_genes)
        K-value matrix, where K_to_use[i, j] is K(src -> tgt).

    Returns
    -------
    param_dict : updated dict with new {k_src_to_tgt} values.
    """

    n_genes = len(gene_list)
    K_to_use = np.asarray(K_to_use, dtype=float)

    # -----------------------------------------------------------
    # 1) Validate matrix shape
    # -----------------------------------------------------------
    if K_to_use.shape != (n_genes, n_genes):
        raise ValueError(
            f"K_to_use must have shape ({n_genes},{n_genes}), "
            f"but got {K_to_use.shape}"
        )

    # -----------------------------------------------------------
    # 2) Validate that every TRUE edge has a specified K value
    # -----------------------------------------------------------
    for i in range(n_genes):
        for j in range(n_genes):
            if connectivity_matrix[i, j] != 0:
                if np.isnan(K_to_use[i, j]):
                    raise ValueError(
                        f"K-value missing (NaN) for interaction "
                        f"{gene_list[i]} -> {gene_list[j]} at index [{i},{j}]"
                    )

                if K_to_use[i, j] <= 0:
                    raise ValueError(
                        f"K-value for {gene_list[i]} -> {gene_list[j]} "
                        f"must be > 0, got {K_to_use[i, j]}"
                    )

    # -----------------------------------------------------------
    # 3) Assign K-values into param_dict
    # -----------------------------------------------------------
    protein_levels = []
    for i, src in enumerate(gene_list):
        for j, tgt in enumerate(gene_list):

            if connectivity_matrix[i, j] != 0:
                key = f"{{K_{src}_to_{tgt}}}"
                param_dict[key] = float(K_to_use[i, j])
                protein_levels.append(float(K_to_use[i, j]))
    print("Assigned K from input matrix.")
    print(param_dict)
    return np.array(protein_levels), param_dict

def generate_K_from_steady_state_calc(param_dict, connectivity_matrix, gene_list,
                                      target_hill=0.5, scale_K=None):
    """
    Calculate steady-state protein levels and assign K values for gene interactions.

    Args:
        param_dict (dict): Dictionary containing parameters for gene regulation.
        connectivity_matrix (numpy.ndarray): Matrix representing gene interactions.
        gene_list (list): List of gene names.
        target_hill (float, optional): Hill function value used to scale regulatory
                                       effects. Default is 0.5.
        scale_K (numpy.ndarray, optional): Scaling matrix for K values. If None,
                                           defaults to a matrix of ones.
    Returns:
        tuple:
            - protein_levels (numpy.ndarray): Steady-state protein levels per gene.
            - param_dict (dict): Updated dictionary with assigned K values.
    """
    n_genes = len(gene_list)
    if scale_K is None:
        scale_K = np.ones((n_genes, n_genes))

    protein_levels = np.zeros(n_genes)

    for i, gene in enumerate(gene_list):
        k_on        = param_dict[f'{{k_on_{gene}}}']
        k_off       = param_dict[f'{{k_off_{gene}}}']
        k_prod_mRNA = param_dict[f'{{k_prod_mRNA_{gene}}}']
        k_deg_mRNA  = param_dict[f'{{k_deg_mRNA_{gene}}}']
        k_prod_prot = param_dict[f'{{k_prod_protein_{gene}}}']
        k_deg_prot  = param_dict[f'{{k_deg_protein_{gene}}}']

        regulators = np.where(connectivity_matrix[:, i] != 0)[0]
        n_regs = len(regulators)

        if n_regs == 0:
            k_on_eff = k_on

        elif n_regs == 1:
            # original per-gene k_add path
            k_add    = param_dict.get(f"{{k_add_{gene}}}", 0.0)
            sign     = int(np.sign(connectivity_matrix[regulators[0], i]))
            k_on_eff = k_on + sign * k_add * target_hill

        elif n_regs == 2:
            sign1 = int(np.sign(connectivity_matrix[regulators[0], i]))
            sign2 = int(np.sign(connectivity_matrix[regulators[1], i]))

            if sign1 == sign2:
                # original per-gene k_add path
                k_add    = param_dict.get(f"{{k_add_{gene}}}", 0.0)
                k_on_eff = k_on + sign1 * k_add * target_hill
            else:
                # new per-edge path — opposing signs
                reg_eff = 0.0
                for r in regulators:
                    edge       = f"{gene_list[r]}_to_{gene}"
                    k_add_edge = param_dict.get(f"{{k_add_{edge}}}", 0.0)
                    sign_r     = int(np.sign(connectivity_matrix[r, i]))
                    reg_eff   += sign_r * k_add_edge * target_hill
                k_on_eff = k_on + reg_eff

        else:
            # new per-edge path — >2 regulators
            reg_eff = 0.0
            for r in regulators:
                edge       = f"{gene_list[r]}_to_{gene}"
                k_add_edge = param_dict.get(f"{{k_add_{edge}}}", 0.0)
                sign_r     = int(np.sign(connectivity_matrix[r, i]))
                reg_eff   += sign_r * k_add_edge * target_hill
            k_on_eff = k_on + reg_eff

        # clamp k_on_eff to avoid negative burst probability
        k_on_eff_clamped = max(k_on_eff, 0.0)
        denom             = k_on_eff_clamped + k_off
        burst_prob        = k_on_eff_clamped / denom if denom > 0 else 0.0
        m                 = k_prod_mRNA * burst_prob / max(k_deg_mRNA, 1e-12)
        protein_levels[i] = max(m * k_prod_prot / max(k_deg_prot, 1e-12), 0.1)

    # assign K values
    for i, src in enumerate(gene_list):
        for j, tgt in enumerate(gene_list):
            if connectivity_matrix[i, j] != 0:
                param_dict[f"{{K_{src}_to_{tgt}}}"] = protein_levels[i] * scale_K[i, j]

    return protein_levels, param_dict

def generate_k_from_max_expression(param_dict, connectivity_matrix, gene_list,
                                      target_hill=0.5, scale_K=None):
    """
    Calculate steady-state protein levels and assign rate constants (k values) 
    for gene interactions based on the provided parameters and interaction 

    Args:
        param_dict (dict): Dictionary containing parameters for gene regulation, 
            including burst probabilities, production rates, degradation rates, 
            and interaction strengths.
        connectivity_matrix (numpy.ndarray): Matrix representing gene interactions, 
            where non-zero values indicate regulatory relationships and their signs 
            (positive for activation, negative for repression).
        gene_list (list): List of gene names corresponding to the rows and columns 
            of the connectivity matrix.
        target_hill (float, optional): Hill function value used to scale regulatory 
            effects. Default is 0.5.
        scale_K (numpy.ndarray, optional): Scaling matrix for rate constants. If 
            None, defaults to a matrix of ones with the same dimensions as the 
            interaction 
    Returns:
        tuple: A tuple containing:
            - protein_levels (numpy.ndarray): Array of steady-state protein levels 
              for each gene.
            - param_dict (dict): Updated dictionary with assigned rate constants 
              (k values) for gene intera
    Notes:
        - The function calculates steady-state protein levels based on burst 
          probabilities and production/degradation rates.
        - Regulatory effects are computed using the connectivity matrix and scaled 
          by the target Hill function value  (default is 0.5).
        - Rate constants (k values) are assigned based on steady-state protein 
          levels and multiplied by the scaling matrix.
    """
    n_genes = len(gene_list)
    if scale_K is None:
        scale_K = np.ones((n_genes, n_genes))
    protein_levels = np.zeros(n_genes)
    for i,gene in enumerate(gene_list):
        k_on = param_dict[f'{{k_on_{gene}}}']
        k_off = param_dict[f'{{k_off_{gene}}}']
        k_prod_mRNA = param_dict[f'{{k_prod_mRNA_{gene}}}']
        k_deg_mRNA  = param_dict[f'{{k_deg_mRNA_{gene}}}']
        k_prod_prot = param_dict[f'{{k_prod_protein_{gene}}}']
        k_deg_prot  = param_dict[f'{{k_deg_protein_{gene}}}']
        regs = np.where(connectivity_matrix[:,i]!=0)[0]

        reg_eff = 0.0
        for r in regs:
            edge = f"{gene_list[r]}_to_{gene}"
            k_add = param_dict.get(f"{{k_add_{edge}}}", 0.0)
            sign = connectivity_matrix[r,i]
            reg_eff += target_hill * k_add * sign
            # print(f"  {edge} — sign: {sign}, k_add: {k_add}")
        
        k_on_eff = k_on + reg_eff  # or replace k_on completely if no basal allowed
        # print(gene, k_on, reg_eff)
        burst_prob = k_on_eff/(k_on_eff+k_off)
        m = k_prod_mRNA * burst_prob / k_deg_mRNA
        protein_levels[i] = max(m * k_prod_prot / k_deg_prot, 0.1)
    
    # assign k values
    for i, src in enumerate(gene_list):
        for j, tgt in enumerate(gene_list):
            if connectivity_matrix[i,j]!=0:
                key = f"{{K_{src}_to_{tgt}}}"
                param_dict[key] = protein_levels[i]*scale_K[i,j]
    return protein_levels, param_dict

def add_interaction_terms(param_dict, connectivity_matrix, gene_list,
                          n_matrix=None, k_add_list=None, scale_K=None,
                          combinatorial_interaction_type="additive",
                          use_given_K=None, K_to_use=None):
    """
    Adds interaction terms to the parameter dictionary based on the connectivity matrix
    and gene list, and calculates steady-state parameters.

    Parameters:
        param_dict (dict): Dictionary to store the interaction parameters.
        connectivity_matrix (numpy.ndarray): Matrix representing interactions between genes.
                                            Non-zero values indicate an interaction.
        gene_list (list): List of gene names corresponding to the rows and columns of
                          the connectivity matrix.
        n_matrix (numpy.ndarray, optional): Matrix specifying the 'n' parameter for each
                                            interaction. Defaults to a matrix filled with 2.0.
        k_add_list (numpy.ndarray, optional): Per-gene k_add values. Used for single-regulator
                                              and same-sign 2-regulator cases. For opposing-sign
                                              or >2 regulator cases, per-edge defaults are used.
        scale_K (numpy.ndarray, optional): Scaling matrix for K values.
        combinatorial_interaction_type (str): One of 'additive', 'AND', 'OR'.
        use_given_K (bool, optional): If True, use K_to_use matrix instead of steady-state calc.
        K_to_use (numpy.ndarray, optional): Matrix of K values to use directly.

    Returns:
        tuple: (protein_levels, updated param_dict)
    """
    n = len(gene_list)

    if n_matrix is None:
        n_matrix = np.full((n, n), 2.0)

    if k_add_list is None:
        k_add_list = np.zeros(n, dtype=float)
        for j in range(n):
            regulators = np.where(connectivity_matrix[:, j] != 0)[0]
            if len(regulators) == 0:
                continue
            # default based on first regulator sign — used only for per-gene cases
            sign = int(np.sign(connectivity_matrix[regulators[0], j]))
            k_add_list[j] = 6.0 if sign > 0 else 0.8

    for j in range(n):
        curr_gene = gene_list[j]
        regulators = np.where(connectivity_matrix[:, j] != 0)[0]
        n_regs = len(regulators)

        if n_regs == 0:
            continue

        elif n_regs == 1:
            # original path — per-gene k_add
            param_dict[f"{{k_add_{curr_gene}}}"] = float(k_add_list[j])

        elif n_regs == 2:
            sign1 = int(np.sign(connectivity_matrix[regulators[0], j]))
            sign2 = int(np.sign(connectivity_matrix[regulators[1], j]))

            if sign1 == sign2:
                # original path — per-gene k_add
                param_dict[f"{{k_add_{curr_gene}}}"] = float(k_add_list[j])
            else:
                # new path — opposing-sign, per-edge k_add
                for i in regulators:
                    edge = f"{gene_list[i]}_to_{curr_gene}"
                    key = f"{{k_add_{edge}}}"
                    if key not in param_dict:
                        edge_sign = int(np.sign(connectivity_matrix[i, j]))
                        param_dict[key] = 6.0 if edge_sign > 0 else 0.8

        else:
            # new path — >2 regulators, per-edge k_add
            for i in regulators:
                edge = f"{gene_list[i]}_to_{curr_gene}"
                key = f"{{k_add_{edge}}}"
                if key not in param_dict:
                    edge_sign = int(np.sign(connectivity_matrix[i, j]))
                    param_dict[key] = 6.0 if edge_sign > 0 else 0.8

        # set n per edge — all cases
        for i in regulators:
            edge = f"{gene_list[i]}_to_{curr_gene}"
            param_dict[f"{{n_{edge}}}"] = float(n_matrix[i, j])

    if use_given_K and K_to_use is not None:
        return assign_k_values_matrix(param_dict, connectivity_matrix, gene_list, K_to_use)
    else:
        return generate_K_from_steady_state_calc(param_dict, connectivity_matrix, gene_list, scale_K=scale_K)

def setup_gillespie_params_from_reactions(init_states: pd.DataFrame,
                                          reactions: pd.DataFrame,
                                          param_dictionary: dict):
    """
    Sets up the parameters required for Gillespie simulation based on initial states, reaction definitions, 
    and a parameter dictionary. This function generates the initial population, update matrix, 
    and a compiled function for updating propensities
    Args:
        init_states (pd.DataFrame): A DataFrame containing the initial states of species. 
                                    Must include columns 'species' and 'count'.
        reactions (pd.DataFrame): A DataFrame defining the reactions. 
                                  Must include columns 'species1', 'species2', 'change1', 'change2', and 'propensity'.
        param_dictionary (dict): A dictionary mapping parameter names to their values, 
                                 used for substituting placeholders in propensity f
    Returns:
        tuple: A tuple containing:
            - pop0 (np.ndarray): Initial population counts as a NumPy array of integers.
            - update_matrix (np.ndarray): A matrix defining the changes in species counts for each reaction.
            - update_propensities (function): A compiled function for updating propensities using numba.
            - species_index (dict): A dictionary mapping species names to their indices
    Raises:
        ValueError: If any placeholders in the propensity formulas are missing from the parameter dic
    Notes:
        - The function dynamically generates and compiles a propensity update function using numba for performance.
        - Species names and parameters in the propensity formulas are replaced with their respective indices and values.
    """
    species_index = {s:i for i,s in enumerate(init_states['species'])}
    pop0 = init_states['count'].values.astype(np.int64)
    update_matrix = []
    prop_formulas = []
    missing = []
    for i,row in reactions.iterrows():
        delta = [0]*len(species_index)
        a1,a2 = row['species1'], row['species2']
        delta[species_index[a1]] = int(row['change1'])
        if a2!='-':
            delta[species_index[a2]] = int(row['change2'])
        update_matrix.append(delta)
        expr = row['propensity']
        # inject species
        for s,idx in species_index.items():
            expr = expr.replace(s, f"pop[idx_{s}]")
        # inject params
        placeholders = set(re.findall(r"{[^}]+}", expr))
        miss = placeholders - set(param_dictionary.keys())
        if miss:
            missing.append((i, miss))
            continue
        for k,v in param_dictionary.items():
            expr = expr.replace(k, str(v))
        line = f"prop[{i}] = {expr}"
        prop_formulas.append(line)
    if missing:
        raise ValueError(f"Missing params in propensities: {missing}")
    # build update function
    src = ["@numba.njit(fastmath=True)",
           "def update_propensities(prop, pop, t):"]
    for s,i in species_index.items():
        src.append(f"    idx_{s} = {i}")
    for L in prop_formulas:
        src.append("    " + L)
    ns = "\n".join(src)
    loc = {}
    exec(ns, {'numba':numba}, loc)
    return pop0, np.array(update_matrix, dtype=np.int64), loc['update_propensities'], species_index

# %% Vectorized extraction

def convert_samples_to_df(samples: np.ndarray, species_index: dict,
                              types=('mRNA','protein', "_A", "_I")) -> pd.DataFrame:
    """
    Extracts mRNA and protein data from simulation samples and organizes it into a pandas DataF
    Parameters:
        samples (np.ndarray): A 3D numpy array of shape (n_cells, n_time, n_species) containing simulation data.
                              Each entry represents the count of a species at a given cell and time step.
        species_index (dict): A dictionary mapping species names to their respective indices in the samples array.
        types (tuple, optional): A tuple of strings specifying the types of species to extract (e.g., 'mRNA', 'protein').
                                 Defaults to ('mRNA', 'protein')
    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted data. The DataFrame includes the following columns:
                      - 'cell_id': The ID of the cell (integer).
                      - 'time_step': The time step (integer).
                      - Columns for each extracted species, named according to the species_index keys.
    """
    n_cells, n_time, _ = samples.shape
    sel = [(name,idx) for name,idx in species_index.items()
           if any(name.endswith(t) for t in types)]
    names, idxs = zip(*sel)
    data = samples[:,:,idxs].reshape(n_cells*n_time, len(idxs))
    cell_ids   = np.repeat(np.arange(n_cells), n_time)
    time_steps = np.tile(np.arange(n_time), n_cells)
    df = pd.DataFrame(data, columns=names)
    df.insert(0,'time_step',time_steps)
    df.insert(0,'cell_id',cell_ids)
    return df

@numba.njit(parallel=True, fastmath=True)
def gillespie_simulation_all_cells(update_propensities, update_matrix,
                                   pop0_mat, time_points, verbose_flags):
    
    n_species, n_cells = pop0_mat.shape
    n_time = time_points.shape[0]
    n_rxns = update_matrix.shape[0]

    samples = np.empty((n_cells, n_time, n_species), dtype=np.int64)

    for cell in prange(n_cells):

        pop = pop0_mat[:, cell].copy()
        t = time_points[0]
        i_time = 0
        stuck_counter = 0
        max_attempts = 10000
        prop = np.zeros(n_rxns, dtype=np.float64)
        next_tp = time_points[0]

        while i_time < n_time:

            update_propensities(prop, pop, t)
            total = prop.sum()

            if total <= 0:
                stuck_counter += 1
                if stuck_counter > max_attempts:
                    verbose_flags[cell] = 1
                    samples[cell, i_time:, :] = pop
                    print(pop)
                    break
                samples[cell, i_time, :] = pop
                continue

            stuck_counter = 0

            reaction_time = np.random.exponential(1.0 / total)
            t += reaction_time

            # Sampling at timepoints
            while i_time < n_time and t >= next_tp:
                samples[cell, i_time, :] = pop
                i_time += 1
                if i_time < n_time:
                    next_tp = time_points[i_time]

            # Reaction selection
            cum_props = np.cumsum(prop)
            r = np.searchsorted(cum_props, np.random.rand() * total)
            pop += update_matrix[r]

    return samples

@numba.njit(parallel=False, fastmath=True)
def gillespie_simulation_all_cells_with_event_log(update_propensities, update_matrix,
                                   pop0_mat, time_points, verbose_flags,
                                   promoter_indices,
                                   event_times_all, event_genes_all, event_states_all,
                                   log_pi_on):
    n_species, n_cells = pop0_mat.shape
    n_time = time_points.shape[0]
    n_rxns = update_matrix.shape[0]
    n_genes = promoter_indices.shape[0]

    samples = np.empty((n_cells, n_time, n_species), dtype=np.int64)

    for cell in range(n_cells):

        pop = pop0_mat[:, cell].copy()
        t = time_points[0]
        i_time = 0
        stuck_counter = 0
        max_attempts = 10000
        prop = np.zeros(n_rxns, dtype=np.float64)
        next_tp = time_points[0]

        # references if we will log
        if log_pi_on:
            event_times  = event_times_all[cell]
            event_genes  = event_genes_all[cell]
            event_states = event_states_all[cell]

        while i_time < n_time:

            update_propensities(prop, pop, t)
            total = prop.sum()

            if total <= 0:
                stuck_counter += 1
                if stuck_counter > max_attempts:
                    verbose_flags[cell] = 1
                    samples[cell, i_time:, :] = pop
                    break
                samples[cell, i_time, :] = pop
                continue

            stuck_counter = 0

            old_pop = pop.copy()
            reaction_time = np.random.exponential(1.0 / total)
            t += reaction_time

            # Sampling at timepoints
            while i_time < n_time and t >= next_tp:
                samples[cell, i_time, :] = pop
                i_time += 1
                if i_time < n_time:
                    next_tp = time_points[i_time]

            # Reaction selection
            cum_props = np.cumsum(prop)
            r = np.searchsorted(cum_props, np.random.rand() * total)
            pop += update_matrix[r]

            # === Log promoter ON/OFF if flag is set ===
            if log_pi_on:
                for g in range(n_genes):
                    A = promoter_indices[g]
                    if old_pop[A] != pop[A]:
                        event_times.append(t)
                        event_genes.append(g)
                        event_states.append(pop[A])   # 0 or 1
                        break

    # ============================
    # RETURN LOGIC
    # ============================
    return samples, event_times_all, event_genes_all, event_states_all



# %%
# Check for steady state
# def is_steady_state(samples, time_points, mean_tol=0.05, std_tol=0.05,
#                     slope_tol=0.05, window_frac=0.1, verbose=False):
#     """
#     Check if the simulation has reached steady state.

#     Args:
#         samples (np.ndarray): Array of shape (n_cells, n_time, n_species)
#         time_points (np.ndarray): Array of time values
#         mean_tol (float): Max relative change in mean allowed
#         std_tol (float): Max relative change in std allowed
#         slope_tol (float): Max absolute slope allowed
#         window_frac (float): Fraction of final time used to assess steady state
#         verbose (bool): Whether to print detailed output

#     Returns:
#         bool: True if steady state is reached
#     """
#     n_cells, n_time, n_species = samples.shape
#     window = int(n_time * window_frac)
#     if window < 2:
#         raise ValueError("Window too small for steady state check.")

#     data = samples[:, -window:, :]  # shape: (n_cells, window, n_species)
#     mean_traj = data.mean(axis=0)   # shape: (window, n_species)
#     std_traj  = data.std(axis=0)    # shape: (window, n_species)

#     # Mean & std relative change over last window
#     rel_mean_change = np.abs(mean_traj[-1] - mean_traj[0]) / (mean_traj[0] + 1e-6)
#     rel_std_change  = np.abs(std_traj[-1] - std_traj[0]) / (std_traj[0] + 1e-6)

#     max_mean_change = rel_mean_change.max()
#     max_std_change  = rel_std_change.max()

#     steady_mean_std = max_mean_change < mean_tol and max_std_change < std_tol

#     # Slope check
#     times = time_points[-window:]
#     slopes = np.zeros(n_species)
#     for g in range(n_species):
#         y = mean_traj[:, g]
#         x = times
#         A = np.vstack([x, np.ones_like(x)]).T
#         m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
#         slopes[g] = m

#     max_abs_slope = np.abs(slopes).max()
#     steady_slope = max_abs_slope < slope_tol

#     is_steady = steady_mean_std or steady_slope

#     print(f"🧪 Steady-state check:")
#     print(f"  ➤ Max relative mean change: {max_mean_change:.4e}")
#     print(f"  ➤ Max relative std  change: {max_std_change:.4e}")
#     print(f"  ➤ Max abs slope:             {max_abs_slope:.4e}")
#     print(f"  ➤ Steady by mean/std:        {steady_mean_std}")
#     print(f"  ➤ Steady by slope:           {steady_slope}")
#     print(f"  ➤ Final decision:            {is_steady}")

#     return is_steady

def hill_fn(x, n, k):
        x = np.asarray(x)
        return x ** n / (x ** n + k ** n)

def is_steady_state(samples, time_points, mean_tol=0.05, std_tol=0.05,
                    window_frac=0.1, param_dict=None, interaction_matrix=None,
                    gene_list=None, verbose=True, combinatorial_interaction_type="additive"):
    """
    Check if simulation has reached steady state and matches expected protein levels.

    Args:
        samples (np.ndarray): Shape (n_cells, n_time, n_species)
        time_points (np.ndarray): Time values
        mean_tol (float): Tolerance for relative mean change over last window
        std_tol (float): Tolerance for relative std change over last window
        window_frac (float): Fraction of final time used to assess steady state
        param_dict (dict): All kinetic + interaction parameters
        interaction_matrix (np.ndarray): (n_genes, n_genes) connectivity matrix
        gene_list (list): Ordered list of gene names
        verbose (bool): Whether to print diagnostics
        combinatorial_interaction_type (str): One of 'additive', 'AND', 'OR'
    
    Returns:
        bool: True if steady state is reached
    """
    n_cells, n_time, n_species = samples.shape
    window = int(n_time * window_frac)
    if window < 2:
        raise ValueError("Window too small for steady state check.")

    protein_species_idx = np.arange(3, n_species, 4)
    n_genes = len(gene_list)
    comb_type = combinatorial_interaction_type.lower()

    # --- Step 1: mean/std stability check ---
    data = samples[:, -window:, :]
    mean_traj = data.mean(axis=0)
    std_traj  = data.std(axis=0)
    rel_mean_change = np.abs(mean_traj[-1] - mean_traj[0]) / (mean_traj[0] + 1e-6)
    rel_std_change  = np.abs(std_traj[-1] - std_traj[0])  / (std_traj[0]  + 1e-6)
    steady_mean_std = (rel_mean_change.max() < mean_tol) and (rel_std_change.max() < std_tol)

    # --- Step 2: compare expected vs simulated proteins ---
    last_n = min(100, n_time)
    rel_error_tp = []

    for t_idx in range(n_time - last_n, n_time):

        proteins_at_t    = samples[:, t_idx, protein_species_idx]  # (n_cells, n_genes)
        mean_at_t_prot   = proteins_at_t.mean(axis=0)              # (n_genes,)
        protein_expected = np.zeros(n_genes)

        for i, gene in enumerate(gene_list):
            k_on        = param_dict[f'{{k_on_{gene}}}']
            k_off       = param_dict[f'{{k_off_{gene}}}']
            k_prod_mRNA = param_dict[f'{{k_prod_mRNA_{gene}}}']
            k_deg_mRNA  = param_dict[f'{{k_deg_mRNA_{gene}}}']
            k_prod_prot = param_dict[f'{{k_prod_protein_{gene}}}']
            k_deg_prot  = param_dict[f'{{k_deg_protein_{gene}}}']

            regulators = np.where(interaction_matrix[:, i] != 0)[0]
            n_regs     = len(regulators)

            if n_regs == 0:
                p_on_eff = k_on

            elif n_regs == 1:
                # original — per-gene k_add, matches simulation exactly
                r      = regulators[0]
                edge   = f"{gene_list[r]}_to_{gene}"
                k_add  = param_dict.get(f"{{k_add_{gene}}}", 0.0)
                n_val  = param_dict.get(f"{{n_{edge}}}", 2.0)
                K_val  = param_dict.get(f"{{K_{edge}}}", 1.0)
                sign   = int(np.sign(interaction_matrix[r, i]))
                TF     = proteins_at_t[:, r]
                hill   = TF**n_val / (K_val**n_val + TF**n_val)
                p_on_eff = k_on + sign * k_add * hill  # (n_cells,)

            elif n_regs == 2:
                sign1 = int(np.sign(interaction_matrix[regulators[0], i]))
                sign2 = int(np.sign(interaction_matrix[regulators[1], i]))

                if sign1 == sign2:
                    # original — per-gene k_add, normalised combinatorial formula
                    r1, r2 = regulators
                    edge1  = f"{gene_list[r1]}_to_{gene}"
                    edge2  = f"{gene_list[r2]}_to_{gene}"
                    k_add  = param_dict.get(f"{{k_add_{gene}}}", 0.0)
                    n1     = param_dict.get(f"{{n_{edge1}}}", 2.0)
                    n2     = param_dict.get(f"{{n_{edge2}}}", 2.0)
                    K1     = param_dict.get(f"{{K_{edge1}}}", 1.0)
                    K2     = param_dict.get(f"{{K_{edge2}}}", 1.0)
                    sign   = sign1
                    TF1    = proteins_at_t[:, r1]
                    TF2    = proteins_at_t[:, r2]
                    u1     = TF1**n1 / K1**n1
                    u2     = TF2**n2 / K2**n2
                    denom  = 1 + u1 + u2 + u1*u2

                    if comb_type == "additive":
                        numerator  = u1 + u2 + 2*u1*u2
                        pre_factor = 1.0
                    elif comb_type == "or":
                        numerator  = u1 + u2 + u1*u2
                        pre_factor = 2.0/3.0
                    elif comb_type == "and":
                        numerator  = u1*u2
                        pre_factor = 2.0
                    else:
                        raise ValueError(f"Unknown combinatorial_interaction_type: {combinatorial_interaction_type}")

                    reg_eff  = sign * k_add * pre_factor * numerator / denom
                    p_on_eff = k_on + reg_eff  # (n_cells,)

                else:
                    # new — opposing-sign, per-edge k_add, simple additive
                    reg_eff = np.zeros(n_cells)
                    for r in regulators:
                        edge       = f"{gene_list[r]}_to_{gene}"
                        k_add_edge = param_dict.get(f"{{k_add_{edge}}}", 0.0)
                        n_val      = param_dict.get(f"{{n_{edge}}}", 2.0)
                        K_val      = param_dict.get(f"{{K_{edge}}}", 1.0)
                        sign_r     = int(np.sign(interaction_matrix[r, i]))
                        TF         = proteins_at_t[:, r]
                        hill       = TF**n_val / (K_val**n_val + TF**n_val)
                        reg_eff   += sign_r * k_add_edge * hill
                    p_on_eff = k_on + reg_eff  # (n_cells,)

            else:
                # new — >2 regulators, per-edge k_add, simple additive
                reg_eff = np.zeros(n_cells)
                for r in regulators:
                    edge       = f"{gene_list[r]}_to_{gene}"
                    k_add_edge = param_dict.get(f"{{k_add_{edge}}}", 0.0)
                    n_val      = param_dict.get(f"{{n_{edge}}}", 2.0)
                    K_val      = param_dict.get(f"{{K_{edge}}}", 1.0)
                    sign_r     = int(np.sign(interaction_matrix[r, i]))
                    TF         = proteins_at_t[:, r]
                    hill       = TF**n_val / (K_val**n_val + TF**n_val)
                    reg_eff   += sign_r * k_add_edge * hill
                p_on_eff = k_on + reg_eff  # (n_cells,)

            # clamp and compute expected protein
            p_on_eff_clamped = np.maximum(p_on_eff, 0.0)
            denom_burst      = p_on_eff_clamped + k_off
            denom_burst      = np.where(denom_burst <= 0, 1e-12, denom_burst)
            burst_prob       = float(np.mean(p_on_eff_clamped / denom_burst))
            m                = k_prod_mRNA * burst_prob / max(k_deg_mRNA, 1e-12)
            protein_expected[i] = max(m * k_prod_prot / max(k_deg_prot, 1e-12), 0.1)

        rel_err = np.abs(mean_at_t_prot - protein_expected) / (protein_expected + 1e-12)
        rel_error_tp.append(rel_err)

    rel_error_tp = np.vstack(rel_error_tp)  # (last_n, n_genes)

    # --- Step 3: per-gene success fraction ---
    frac_within_tol       = np.mean(rel_error_tp < 0.01, axis=0)
    steady_match_per_gene = frac_within_tol >= 0.8
    steady_match          = bool(np.all(steady_match_per_gene))

    # --- Verbose output ---
    if verbose:
        print("\n Steady-state check:")
        print(f"  Max rel mean change over last {window} steps: {rel_mean_change.max():.4e}")
        print(f"  Max rel std  change over last {window} steps: {rel_std_change.max():.4e}")
        print(f"  Steady by mean/std stability:                 {steady_mean_std}")
        print(f"  Steady by param-based protein match:          {steady_match}")
        print(f"  Per-gene fraction of time points within 1% of expected protein:")
        for gene, frac, passed in zip(gene_list, frac_within_tol, steady_match_per_gene):
            status = "pass" if passed else "fail"
            print(f"     {gene:>15}: {frac*100:6.2f}%  {status}")

    return steady_match

# %% Wrapping functions 
def run_simulation(update_propensities, update_matrix, pop0, time_points, n_cells=1000, promoter_indices = None):
    """
    Simulates the dynamics of a population of cells using the Gillespie algorithm.

    Parameters:
        update_propensities (callable): A function to compute the propensities for reactions.
        update_matrix (numpy.ndarray): The stoichiometry matrix defining the system's reactions.
        pop0 (numpy.ndarray): Initial population vector for all species (shape: [n_species]).
        time_points (numpy.ndarray): Array of time points at which to sample the population.
        n_cells (int, optional): Number of cells to simulate. Defaults to 1000.
        

    Returns:
        numpy.ndarray: A 3D array containing the simulated population data. 
                       Shape: [n_species, len(time_points), n_cells].

    Notes:
        - The function uses a JIT-compiled helper function `gillespie_simulation_all_cells` for efficient simulation.
        - Warnings are printed for cells that encounter issues during simulation:
            - Cell stuck due to zero propensities for too long.
    """
    n_species = pop0.shape[0]
    pop0_mat = np.tile(pop0[:, None], (1, n_cells))
    pop0_mat = pop0_mat.copy()
    verbose_flags = np.zeros(n_cells, dtype=np.int64)

    samples = gillespie_simulation_all_cells(update_propensities, update_matrix,
                                   pop0_mat, time_points, verbose_flags)
                                   
    for cell in range(n_cells):
        if verbose_flags[cell] == 1:
            print(f"⚠️ WARNING: Cell {cell} got stuck (zero propensities).")
    return samples

def get_promoter_indices(species_index, gene_list):
    promoter_indices = []
    for g in gene_list:
        key = f"{g}_A"   # the active/ON promoter state
        promoter_indices.append(species_index[key])
    return np.array(promoter_indices, dtype=np.int64)

def save_promoter_events(event_times, event_genes, event_states):
    """
    Convert Numba event logs into a tidy DataFrame and return the DataFrame.

    Parameters
    ----------
    event_times : List[List[float]]
        Per-cell list of times when promoter switched.
    event_genes : List[List[int]]
        Per-cell list of gene indices corresponding to each event.
    event_states : List[List[int]]
        Per-cell list of new promoter states (0 or 1).
    """
    
    rows = []
    n_cells = len(event_times)

    for cell in range(n_cells):
        times  = event_times[cell]
        genes  = event_genes[cell]
        states = event_states[cell]

        for k in range(len(times)):
            rows.append({
                "cell": cell,
                "gene": genes[k],
                "time": times[k],
                "new_state": states[k],   # 0 or 1
            })

    df = pd.DataFrame(rows)
    return df



def allocate_event_logs(n_cells):
    event_times_all  = List()
    event_genes_all  = List()
    event_states_all = List()

    for _ in range(n_cells):
        event_times_all.append(List.empty_list(numba.float64))
        event_genes_all.append(List.empty_list(numba.int64))
        event_states_all.append(List.empty_list(numba.int64))

    return event_times_all, event_genes_all, event_states_all

def validate_regulatory_configuration(connectivity_matrix):
    """
    Validate that the regulatory configuration is compatible with the current
    simulation assumptions. Raises an error for unsupported configurations,
    and warns for configurations that will use simple additive per-edge logic.
    """
    n = connectivity_matrix.shape[0]
    gene_list = [f"gene_{i+1}" for i in range(n)]

    for j in range(n):
        regulators = np.where(connectivity_matrix[:, j] != 0)[0]
        n_regs = len(regulators)

        if n_regs > 2:
            reg_names = [gene_list[i] for i in regulators]
            print(
                f"Gene '{gene_list[j]}' has {n_regs} regulators {reg_names}. "
                f"Using simple additive per-edge logic."
            )

        if n_regs == 2:
            signs = np.sign(connectivity_matrix[regulators, j]).astype(int)
            if signs[0] != signs[1]:
                reg_names = [gene_list[i] for i in regulators]
                print(
                    f"Gene '{gene_list[j]}' has opposing-sign regulators {reg_names} "
                    f"(signs {signs.tolist()}). "
                    f"Using simple additive per-edge logic."
                )

def divide_mother_cell_content(
    mother_states,
    species_index,
    seed=None,
    partition_suffixes=("_mRNA", "_protein"),
    p_major=0.5,          # 0.6 means 60/40, 0.7 means 70/30
    randomize_polarity=True,
):
    rng = np.random.default_rng(seed)

    mother_states = mother_states.astype(np.int64)
    n_species, n_cells = mother_states.shape

    twin_1 = np.empty_like(mother_states)
    twin_2 = np.empty_like(mother_states)

    # partition only mRNA + protein explicitly
    partition_indices = np.array(
        [idx for name, idx in species_index.items()
         if name.endswith(partition_suffixes)],
        dtype=np.int64
    )
    copy_indices = np.setdiff1d(np.arange(n_species), partition_indices)

    # copy everything else (promoters, etc.)
    twin_1[copy_indices] = mother_states[copy_indices]
    twin_2[copy_indices] = mother_states[copy_indices]

    # total molecules available to split (your "keep mean = M" rule)
    doubled = 2 * mother_states[partition_indices]  # shape: (n_part, n_cells)

    # per-cell probability for twin_1
    if randomize_polarity:
        # mask[j] = True => twin_1 is the major daughter for cell j
        mask = rng.random(n_cells) < 0.5
        p_cell = np.where(mask, p_major, 1.0 - p_major)  # shape: (n_cells,)
    else:
        p_cell = np.full(n_cells, p_major)

    # broadcast p_cell across partitioned species rows
    draw = rng.binomial(doubled, p_cell[None, :])

    twin_1[partition_indices] = draw
    twin_2[partition_indices] = doubled - draw

    return twin_1, twin_2



# --- Worker for a single parameter set ---
def process_param_set(rows, label, base_config):
    """
    Processes a set of parameters for a Gillespie simulation, running the simulation for a specified number of cells and handling the results.
    Parameters:
        rows (list): A list of parameter rows to be processed.
        label (str): A label for identifying the simulation run.
        base_config (dict): A dictionary containing common parameters such as paths, connectivity matrix, and simulation settings.
    Returns:
        str: The file path of the saved DataFrame containing the results of the simulation.
    Raises:
        AssertionError: If the number of parameter rows is less than the number of genes.
    """
    # base_config contains common parameters: paths, k_add_list, n_matrix, time_points
    # Unpack base_config
    path_to_connectivity_matrix = base_config['path_to_connectivity_matrix']
    param_csv      = base_config['param_csv']
    k_add_list   = base_config.get("k_add_list", None)
    n_matrix   = base_config.get("n_matrix", None)
    time_points    = np.arange(0, base_config['simulation_time_before_division'], 1)
    sample_twins_time_points = np.arange(0, base_config['twin_simulation_time_after_division'] + base_config['twin_measurement_resolution'], base_config['twin_measurement_resolution']) 
    n_cells = base_config['n_cells']
    scale_K = base_config.get("scale_K", None)
    log_pi_on = base_config.get("log_pi_on", False)
    use_given_K = base_config.get("use_given_K", False)
    divide_binomial = base_config.get("divide_binomial", False)
    p_major = base_config.get("p_major", 0.5)
    K_to_use = base_config.get("K_to_use", None)
    if use_given_K and K_to_use is not None:
        print("Using given hill Constants.")
    combinatorial_interaction_type = base_config.get("combinatorial_interaction_type", "additive")
    if combinatorial_interaction_type not in ['additive', 'AND', 'OR']:
        raise ValueError("The three options for combinatorial interaction type are: 'additive', 'AND', 'OR' ")
    print(f"Log pi on is set to {log_pi_on}")
    print(f"Combinatorial interaction type: {combinatorial_interaction_type}")
    # Build reactions and parameters for this row set
    n_genes, connectivity_matrix = read_input_matrix(path_to_connectivity_matrix)
    assert len(rows) >= n_genes, "The number of parameter rows entered is less than the number of genes"
    validate_regulatory_configuration(connectivity_matrix)
    reactions_df, gene_list = generate_reaction_network_from_matrix(connectivity_matrix, combinatorial_interaction_type=combinatorial_interaction_type)
    
    init_states = generate_initial_state_from_genes(gene_list)
    param_dict = assign_parameters_to_genes(param_csv, gene_list, rows)

    if n_matrix is None:
        n_matrix = np.zeros((n_genes, n_genes))
    
    if k_add_list is None:
        k_add_per_gene = np.zeros(n_genes, dtype=float)

        for j in range(n_genes):
            regulators = np.where(connectivity_matrix[:, j] != 0)[0]

            if len(regulators) == 0:
                continue

            target_gene = gene_list[j]

            # check for per-edge k_add values in param_dict from CSV
            k_add_values = [
                v for key, v in param_dict.items()
                if key.startswith("{k_add_") and key.endswith(f"_to_{target_gene}}}")
            ]

            if k_add_values:
                # average of CSV-provided per-edge values — used for per-gene cases
                # for opposing-sign/>2 regulator cases this is ignored in add_interaction_terms
                k_add_per_gene[j] = np.mean(k_add_values)
                if len(k_add_values) > 1:
                    print(f"{target_gene}: averaging {len(k_add_values)} k_add values from CSV → {k_add_per_gene[j]:.4f}")
            else:
                # sign-based default — only meaningful for per-gene cases
                sign = int(np.sign(connectivity_matrix[regulators[0], j]))
                k_add_per_gene[j] = 6.0 if sign > 0 else 0.8

    else:
        if len(k_add_list) < n_genes:
            raise ValueError(
                f"k_add_list must have length {n_genes}, got {len(k_add_list)}"
            )
        k_add_per_gene = np.array(k_add_list, dtype=float)[:n_genes]
        print(f"Using provided k_add_list: {k_add_per_gene} out of {len(k_add_list)} provided values")

    for j in range(n_genes):
        regulators = np.where(connectivity_matrix[:, j] != 0)[0]

        if len(regulators) == 0:
            continue  # no regulation → leave k_add = 0

        # fill n_matrix per edge as before
        for i in regulators:
            edge = f"{gene_list[i]}_to_{gene_list[j]}"
            n_matrix[i, j] = param_dict.get(f"{{n_{edge}}}", 2.0)

    print("Done until addition of interaction terms")
    steady_state, full_param_dict = add_interaction_terms(param_dict=param_dict, connectivity_matrix=connectivity_matrix, gene_list=gene_list,
                                                          n_matrix=n_matrix,
                                                          k_add_list=k_add_per_gene, scale_K=scale_K,
                                                          combinatorial_interaction_type=combinatorial_interaction_type,  use_given_K = use_given_K, K_to_use = K_to_use)
    print(full_param_dict)

    pop0, update_matrix, update_prop, species_index = setup_gillespie_params_from_reactions(
        init_states, reactions_df, full_param_dict)
    promoter_indices = get_promoter_indices(species_index, gene_list)
    print("Starting base simulation")
    # 1) Run base simulation
    base_samples = run_simulation(update_prop, update_matrix, pop0, time_points, n_cells, promoter_indices= promoter_indices)
    flag = 0
    if not is_steady_state(samples = base_samples, time_points =  time_points, param_dict = full_param_dict, interaction_matrix = connectivity_matrix, gene_list = gene_list):
        print(f"⚠️ Base simulation (basal) for {label} may not be steady. Please manually verify and increase pre-division time if it has not reached steady state.")
        # Log the issue in a separate file
        error_record = {
            "id": uuid.uuid4().hex[:8],
            "rows": rows,
            "timestamp": datetime.now().strftime("%d%m%Y_%H%M%S"),
            "issue": "Base simulation not steady",
            "label": label
        }
        flag = 1
        log_folder = os.path.join(os.path.dirname(base_config['log_file']))
        os.makedirs(log_folder, exist_ok=True)
        log_file_path = os.path.join(log_folder, f"error_log.jsonl")
        with open(log_file_path, "a") as log_file:
            log_file.write(json.dumps(error_record) + "\n")
        
    df_base = convert_samples_to_df(base_samples, species_index)
    
    # 2) Replicate into two to create daughter cells
    final_states = base_samples[:, -1, :]
    del base_samples
    gc.collect()
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    id = uuid.uuid4().hex[:8]
    prefix = f"{label}_{timestamp}_ncells_{n_cells}_{base_config['type']}_{id}"
    df_base.to_csv(f"{base_config['output_folder']}/simulation_before_division_df_{prefix}.csv", index=False)
    rep_time = sample_twins_time_points
    if divide_binomial:
        twin_1, twin_2 = divide_mother_cell_content(final_states.T, species_index=species_index,seed=101010, p_major=p_major)
        pop0_rep = np.concatenate([twin_1, twin_2], axis=1)
    else:
        pop0_rep = np.concatenate([final_states.T, final_states.T], axis=1)
    verbose_flags = np.zeros(2*n_cells, dtype=np.int64)

    if log_pi_on:
        # ---- allocate logging lists outside numba ----
        event_times_all, event_genes_all, event_states_all = allocate_event_logs(2 * n_cells)

        # ---- run simulation with logging ----
        rep_samples, event_times_all, event_genes_all, event_states_all = gillespie_simulation_all_cells_with_event_log(
            update_prop, update_matrix, pop0_rep, rep_time,
            verbose_flags, promoter_indices,
            event_times_all, event_genes_all, event_states_all,
            log_pi_on=True
        )

        # ---- flatten events and save csv ----
        event_log = save_promoter_events(event_times_all, event_genes_all, event_states_all)

        event_log_file_name = (
            f"event_log_{label}_{timestamp}_ncells_{n_cells}_{base_config['type']}_{id}"
        )
        event_log.to_csv(f"{base_config['output_folder']}/{event_log_file_name}.csv", index=False)

    else:
        # ---- still allocate empty lists to satisfy Numba return type ----

        rep_samples = gillespie_simulation_all_cells(
            update_prop, update_matrix, pop0_rep, rep_time, verbose_flags
        )

    # 3) Extract from simulation and label
    df_rep = convert_samples_to_df(rep_samples, species_index)
    n_total = 2 * n_cells
    replicate_ids = np.repeat([1, 2], n_cells)
    clone_ids = np.tile(np.arange(n_cells), 2)

    df_rep['replicate'] = replicate_ids[df_rep['cell_id']]
    df_rep['clone_id'] = clone_ids[df_rep['cell_id']]
    df_rep['cell_id'] = df_rep.index // len(rep_time) 
    
    # 4) Save
    # timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    # id = uuid.uuid4().hex[:8]
    df_rep.to_csv(f"{base_config['output_folder']}/df_{prefix}.csv", index=False)
    # np.savetxt(f"{base_config['output_folder']}/samples_{prefix}.csv", rep_samples.reshape(2*n_cells, -1), delimiter=",")
    # if flag:

    record = {
        "id": id,
        "rows": rows,
        "n_cells": n_cells,
        "type": base_config['type'],
        "timestamp": timestamp,
        "param_dict": full_param_dict,
        "steady_state": steady_state.tolist() if hasattr(steady_state, 'tolist') else list(steady_state)

    }
    os.makedirs(os.path.dirname(base_config['log_file']), exist_ok=True)
    with open(base_config['log_file'],"a") as f:
        f.write(json.dumps(record) + "\n")
    return f"{base_config['output_folder']}/df_{prefix}.csv"

#%%
def check_if_file_exists(rows, output_folder, type_name):
    """
    Check if a file for the given parameter set already exists
    """
    # Create the pattern that matches your existing filename format
    row_pattern = "_".join(map(str, rows))
    pattern = f"{output_folder}/df_row_{row_pattern}_*_{type_name}_*.csv"
    
    # Check if any files match this pattern
    existing_files = glob.glob(pattern)
    
    if existing_files:
        return True
    return False
#%%
# --- Main execution with parallel parameter sets ---
if __name__ == "__main__":
    root = "/projects/b1042/GoyalLab/Keerthana/"
    # Base configuration - the commented out lines can be used instead of providing arguments to the file (e.g. if using it as ipynb notebook)
    base_config = {
        'time_points':    np.arange(0, 2500, 1), #Time to reach steady state
        'n_cells':        10000, #Before division
        # "path_to_matrix":  "/path/to/interaction/matrix.txt",
        # "param_csv":      "/path/to/parameters.csv",
        # "row_to_start":      0,
        # "output_folder":      "/path/to/output/folder/",
        # "log_file":      "/path/to/log.jsonl",
        # "type":      "A_to_B",
        
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Gillespie simulation with specified inputs.")
    parser.add_argument("--path_to_connectivity_matrix", type=str, required=True, help="Path to the connectivity matrix file specifying the GRN to simulate.")
    parser.add_argument("--param_csv", type=str, required=True, help="Path to the parameters for all genes and interaction terms.")
    parser.add_argument("--row_to_start", type=int, required=True, help="Row of parameter file to start for this batch of simulations.")
    parser.add_argument("--row_to_end", type=int, required=False, default = None, help="Row of parameter file to end for this batch of simulations.")
    parser.add_argument("--output_folder", type=str , required=True, help="Path to output folder to store simulation.")
    parser.add_argument("--log_file", type=str , required=True, help="Json file to save log.")
    parser.add_argument("--type", type=str , required=True, help="Name of the network used -- will be in the filename.")
    parser.add_argument("--number_parallel_processes", type=int, default=1, required=False, help="Number of parallel parameter sets to be run at once (default: 1).")
    parser.add_argument("--number_of_cores_per_parameter", type=int, default=4, required=False, help="Number of cores to be used per parameter (default: 4).")
    parser.add_argument("--n_genes", type=int, default=2, required=False, help="Number of genes in the system (default: 2).")
    parser.add_argument("--n_cells", type=int, default=5000, required=False, help="Number of cells in the system (default: 5000).")
    parser.add_argument("--simulation_time_before_division", type=int, default=2500, required=False, help="Number of hours to run to achieve steady state (default: 2500h).")
    parser.add_argument("--twin_simulation_time_after_division", type=int, default=48, required=False, help="Number of hours to run after cell division for collecting twin data (default: 48h).")
    parser.add_argument("--twin_measurement_resolution", type=int, default=1, required=False, help="The time duration between every twin measurement (default: 1). For example, if it is 1h, then, data is stored eevry hour.")
    parser.add_argument("--scale_K", type=str, default=None, required=False, help="The matrix of values to scale Hill constant K for each interaction. "
                        "Provide as string representation of nested list, e.g., '[[0,1],[0,0]]'. "
                        "Default is matrix of 1s for all interactions.")    
    parser.add_argument("--log_pi_on", action="store_true", default=False, help="If set, the activity of promoter (A/I) and their timing will be recorded and saved for every genes.")   
    parser.add_argument("--k_add_list", type=str, default=None,
                    help="Comma-separated k_add per gene, e.g. '6.0,0.8'")
    parser.add_argument("--n_matrix", type=str, default=None,
                        help="Nested list string for n_matrix, e.g. '[[2,2],[2,2]]'")
    parser.add_argument("--use_given_K", action="store_true", default=False,
                        help="Use provided K_to_use matrix instead of steady-state calc.")
    parser.add_argument("--K_to_use", type=str, default=None,
                        help="Nested list string for K matrix, e.g. '[[0,100],[50,0]]'")
    parser.add_argument("--divide_binomial", action="store_true", default=False,
                        help="Use binomial partitioning at cell division.")
    parser.add_argument("--p_major", type=float, default=0.5,
                        help="Major daughter fraction for binomial division (default: 0.5).")
    parser.add_argument("--combinatorial_interaction_type", type=str, default="additive",
                        choices=["additive", "AND", "OR"],
                        help="Combinatorial logic for two-regulator genes (default: additive).")
    args = parser.parse_args()

    # # Update base configuration with parsed arguments
    base_config["path_to_connectivity_matrix"] = args.path_to_connectivity_matrix
    base_config["param_csv"] = args.param_csv
    base_config["row_to_start"] = int(args.row_to_start)
    
    base_config["output_folder"] = args.output_folder
    base_config["log_file"] = args.log_file
    base_config["type"] = args.type
    base_config["number_parallel_processes"] = args.number_parallel_processes
    base_config["n_genes"] = args.n_genes
    base_config["n_cells"] = args.n_cells
    base_config["number_of_cores_per_parameter"] = args.number_of_cores_per_parameter
    base_config["simulation_time_before_division"] = args.simulation_time_before_division
    base_config["twin_simulation_time_after_division"] = args.twin_simulation_time_after_division
    base_config["twin_measurement_resolution"] = args.twin_measurement_resolution
    base_config['scale_K'] = None
    base_config['log_pi_on']= args.log_pi_on

    if args.k_add_list is not None:
        raw = args.k_add_list.strip()
        if raw.startswith("["):
            # handles [6.0,0.8,6.0]
            base_config["k_add_list"] = [float(x) for x in raw.strip("[]").split(",")]
        else:
            # handles 6.0,0.8,6.0
            base_config["k_add_list"] = [float(x) for x in raw.split(",")]
    else:
        base_config["k_add_list"] = None

    base_config["use_given_K"] = args.use_given_K
    base_config["divide_binomial"] = args.divide_binomial
    base_config["p_major"] = args.p_major
    base_config["combinatorial_interaction_type"] = args.combinatorial_interaction_type

    for key, arg_val in [("n_matrix", args.n_matrix), ("K_to_use", args.K_to_use)]:
        if arg_val is not None:
            try:
                base_config[key] = np.array(ast.literal_eval(arg_val))
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing {key}: {e}")
                raise
        else:
            base_config[key] = None

    os.makedirs(base_config["output_folder"], exist_ok = True)
    try:
        df = pd.read_csv(base_config['param_csv'])
    except FileNotFoundError:
        print(f"Error: The file {base_config['param_csv']} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    if args.row_to_end is not None:
        base_config["row_to_end"] = int(args.row_to_end)
    else:
        base_config["row_to_end"] = df["row_to_start"] + 1
        print(f"row_to_end not specified, defaulting to last pair: {base_config['row_to_end']}")
    # Read the connectivity matrix before using it
    path_to_connectivity_matrix = base_config["path_to_connectivity_matrix"]
    n_genes, mat = read_input_matrix(path_to_connectivity_matrix)  # Ensure mat is defined
    
    if args.scale_K is not None:
        try:
            # Parse the string representation into nested list, then convert to numpy array
            parsed_matrix = ast.literal_eval(args.scale_K)
            base_config["scale_K"] = np.array(parsed_matrix)
            print(f"Using provided scale_K matrix: {base_config['scale_K']}")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing scale_K matrix: {e}")
            print("Expected format: '[[0,1],[0,0]]' (include the quotes)")
            raise
        except Exception as e:
            print(f"Error converting to numpy array: {e}")
            raise
        expected_shape = (base_config["n_genes"], base_config["n_genes"])
        
        if base_config["scale_K"].shape != expected_shape:
            print(f"Warning: scale_K matrix shape {base_config['scale_K'].shape} doesn't match expected {expected_shape}")
    
    
    start_pair = base_config["row_to_start"]  # row_to_start now refers to pair_id
    end_pair = base_config["row_to_end"]
    print(f"start_pair: {start_pair}, end_pair: {end_pair}")
    row_list = []
    labels = []

    for pair in range(start_pair, end_pair + 1):

        subset = df[df["pair_id"] == pair].sort_values("gene_id")
        rows = subset.index.tolist()
    
        # Ensure only complete groups are taken
        rows_to_use = rows[:n_genes]
        if len(rows) >= n_genes:
            if len(rows) > n_genes:
                print(f"pair {pair}: {len(rows)} rows found, using first {n_genes}: {rows_to_use}")
            if not check_if_file_exists(rows_to_use, base_config["output_folder"], base_config["type"]):
                row_list.append(rows_to_use)
                labels.append(f"row_{'_'.join(map(str, rows_to_use))}")
        else:
            print(f"pair {pair}: only {len(rows)} rows found, need {n_genes}. Skipping.")
            break
        
    param_sets = list(zip(row_list, labels))
    print(base_config["output_folder"])
    print("length of param_sets", len(param_sets))
    # Modified function wrapper for joblib that sets numba threads internally
    def process_param_set_with_numba_config(rows, label, config):
        """
        Wrapper function that configures numba threads and then processes the parameter set
        """
        set_num_threads(base_config['number_of_cores_per_parameter'])
        print(f"Worker process - Numba threads set to: {get_num_threads()}")
        
        # Call your original processing function
        return process_param_set(rows, label, config)

    # Use joblib instead of concurrent.futures
    

    print(f"Starting joblib with {base_config['number_parallel_processes']} parallel processes")

    # Run parallel processing with joblib
    results = Parallel(
        n_jobs=base_config['number_parallel_processes'], 
        backend='multiprocessing',
        verbose=0  # Set to 10 for more verbose output
    )(
        delayed(process_param_set_with_numba_config)(rows, label, base_config) 
        for rows, label in tqdm(param_sets, desc="Param sets")
    )

    # Process results
    for result in results:
        if result:
            print(f"Completed simulation: {result}")
