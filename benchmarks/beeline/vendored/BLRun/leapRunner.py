import os
from pathlib import Path

import pandas as pd
import numpy as np

from BLRun.runner import Runner

# LEAP runs natively via R (module load R/4.2.3 on Quest) rather than through
# Docker, which isn't available on this HPC cluster. The LEAP CRAN package is
# installed into a project-local R library (r_lib/), mirroring what the
# grnbeeline/leap:base image installs from the same CRAN Archive URL.
_REPO_ROOT    = Path(__file__).resolve().parent.parent
_LEAP_SCRIPT  = _REPO_ROOT / "Algorithms" / "LEAP" / "runLeap.R"
_R_LIBS_USER  = _REPO_ROOT / "r_lib"


class LEAPRunner(Runner):
    """Concrete runner for the LEAP GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for LEAP.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            exprName = "ExpressionData"+str(idx)+".csv"

            subPT = PTData.loc[index,:]
            subExpr = ExpressionData[index]
            # Order columns by PseudoTime
            newExpressionData = subExpr[subPT.sort_values([colName]).index.astype(str)]

            newExpressionData.insert(loc = 0, column = 'GENES', \
                                                         value = newExpressionData.index)

            # Write .csv file
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def run(self):
        '''
        Function to run LEAP algorithm

        Requires the maxLag parameter
        '''

        maxLag = str(self.params['maxLag'])

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            inner = ' '.join([f"R_LIBS_USER={_R_LIBS_USER}",
                              "/usr/bin/time -v -o", f"{self.working_dir}/time{idx}.txt",
                              "Rscript", str(_LEAP_SCRIPT),
                              f"{self.working_dir}/ExpressionData{idx}.csv",
                              maxLag,
                              f"{self.working_dir}/outFile{idx}.txt"])
            # bash -lc ensures Lmod's `module` function is available and that
            # module-provided R takes priority over any R bundled in a conda
            # env BLRunner.py might itself be running under (BEELINE's
            # environment.yml pins an old r=3.5.0, which otherwise silently
            # shadows the module's Rscript on PATH).
            cmdToRun = f'bash -lc "module load R/4.2.3 && {inner}"'

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from LEAP.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        OutSubDF = [0]*len(colNames)

        for indx in range(len(colNames)):
            outFileName = 'outFile'+str(indx)+'.txt'
            # Quit if output file does not exist
            if not (workDir / outFileName).exists():
                print(str(workDir / outFileName) + ' does not exist, skipping...')
                return

            # Read output
            OutSubDF[indx] = pd.read_csv(workDir / outFileName, sep = '\t', header = 0)
            OutSubDF[indx].Score = np.abs(OutSubDF[indx].Score)
        outDF = pd.concat(OutSubDF)
        FinalDF = outDF[outDF['Score'] == outDF.groupby(['Gene1','Gene2'])['Score'].transform('max')]

        self._write_ranked_edges(FinalDF.sort_values('Score', ascending=False).rename(
            columns={'Score': 'EdgeWeight'}
        )[['Gene1', 'Gene2', 'EdgeWeight']])
