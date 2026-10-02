import os
from pathlib import Path

import pandas as pd
import numpy as np

from BLRun.runner import Runner

# JUMP3 runs natively via its precompiled MATLAB-Compiler binary rather than
# through Docker, which isn't available on this HPC cluster. Quest has no
# MATLAB Runtime R2019a (v9.6) module (the closest installed MATLAB is
# R2018a/R2018b, then a gap straight to R2020b), so the exact MCR v9.6 was
# self-installed into mcr_installs/v96/ from the same public, unauthenticated
# MathWorks URL the Dockerfile uses. The wrapper script (run_runJump3.sh) is
# already vendored in the repo and takes the MCR root as its first argument.
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_JUMP3_DIR  = _REPO_ROOT / "Algorithms" / "JUMP3" / "J3p"
_MCR_ROOT   = _REPO_ROOT / "mcr_installs" / "v96"


class JUMP3Runner(Runner):
    """Concrete runner for the JUMP3 GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for JUMP3.

        Splits into one input file per pseudotime trajectory (column in
        PseudoTime.csv), matching the pattern used by the other multi-
        trajectory runners (GRISLI, SCODE, etc.). The original version
        assumed a single column literally named 'PseudoTime', which does
        not exist for multi-trajectory datasets like the GSD example
        (columns are 'PseudoTime1'/'PseudoTime2') and raised a KeyError.

        Cells are written in ascending pseudotime order (matching the
        sort_values pattern already used by LEAP/GRNVBEM) because JUMP3
        requires the first row of each time series to be at time 0 — it
        does not re-sort by the Time column itself.
        '''

        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            # Sort ascending by pseudotime so the first written row is the
            # earliest time point (see docstring above).
            index = PTData.loc[index, colName].sort_values().index

            newExpressionData = ExpressionData.loc[:,index].T.copy()
            newExpressionData.index = newExpressionData.index.map(str)

            subPT = PTData.loc[index, colName]
            subPT.index = subPT.index.map(str)

            # Acc. to JUMP3:
            # In input argument Time, the first time point of each time series must be 0.
            # Also has to be an integer! (JUMP3's MATLAB code uses Time as an
            # array-size argument, so continuous pseudotime must be rounded.)
            newExpressionData['Time'] = np.round(subPT - subPT.min()).astype(int)
            newExpressionData['Experiment'] = 1

            newExpressionData.to_csv(self.working_dir / f"ExpressionData{idx}.csv",
                                 sep = ',', header  = True, index = False)

    def run(self):
        '''
        Function to run JUMP3 algorithm
        '''

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns

        for idx in range(len(colNames)):
            cmdToRun = ' '.join(["/usr/bin/time -v -o", f"{self.working_dir}/time{idx}.txt",
                                str(_JUMP3_DIR / "run_runJump3.sh"), str(_MCR_ROOT),
                                f"{self.working_dir}/ExpressionData{idx}.csv",
                                f"{self.working_dir}/outFile{idx}.txt"])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from JUMP3.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns
        OutSubDF = [0]*len(colNames)

        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)
        GeneList = list(ExpressionData.index)

        for indx in range(len(colNames)):
            outFile = workDir / f'outFile{indx}.txt'
            if not outFile.exists():
                print(str(outFile) + ' does not exist, skipping...')
                return

            # Read output
            OutDF = pd.read_csv(outFile, sep = ',')

            # Sort values in a matrix using code from:
            # https://stackoverflow.com/questions/21922806/sort-values-of-matrix-in-python
            OutMatrix = np.abs(OutDF.values)
            idx = np.argsort(OutMatrix, axis = None)[::-1]
            rows, cols = np.unravel_index(idx, OutDF.shape)
            DFSorted = OutMatrix[rows, cols]

            OutSubDF[indx] = pd.DataFrame({
                'Gene1':      [GeneList[r] for r in rows],
                'Gene2':      [GeneList[c] for c in cols],
                'EdgeWeight': DFSorted,
            })

        outDF = pd.concat(OutSubDF)
        # megre the dataframe by taking the maximum value from each DF
        res = outDF[outDF['EdgeWeight'] == outDF.groupby(['Gene1','Gene2'])['EdgeWeight'].transform('max')]
        finalDF = res.sort_values('EdgeWeight', ascending=False)

        self._write_ranked_edges(finalDF[['Gene1', 'Gene2', 'EdgeWeight']])