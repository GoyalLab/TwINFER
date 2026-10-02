import os
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner

# SCRIBE runs natively via a dedicated "scribe" conda env (Bioconductor R 4.1
# + monocle 2.22.0 + custom-built RANNinf/Scribe packages) rather than through
# Docker, which isn't available on this HPC cluster.
_REPO_ROOT     = Path(__file__).resolve().parent.parent
_SCRIBE_SCRIPT = _REPO_ROOT / "Algorithms" / "SCRIBE" / "runScribe.R"
# Invoked by absolute path rather than `conda run -p ...` -- this host has
# more than one conda installation, and the `conda` command itself can behave
# unreliably depending on which install's binary is first on PATH for a
# given shell/node. Conda R builds embed their own lib dir via RPATH, so no
# extra env vars are needed to find shared libraries at runtime.
_CONDA_PREFIX  = _REPO_ROOT / "conda_envs" / "scribe"
_ENV_RSCRIPT   = _CONDA_PREFIX / "bin" / "Rscript"


class SCRIBERunner(Runner):
    """Concrete runner for the SCRIBE GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SCRIBE.
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
            ExpressionData.loc[:,index].to_csv(self.working_dir / exprName,
                                     sep = ',', header  = True, index = True)
            cellName = "pseudoTimeData"+str(idx)+".csv"
            ptDF = PTData.loc[index,[colName]]
            # Scribe expects a column labeled Time.
            ptDF.rename(columns = {colName:'Time'}, inplace = True)

            ptDF.to_csv(self.working_dir / cellName,
                                     sep = ',', header  = True, index = True)

        SCRIBE_GENE_FILE = self.working_dir / "GeneData.csv"
        if not SCRIBE_GENE_FILE.exists():
            # required column!!
            geneDict = {}
            geneDict['gene_short_name'] = [gene.replace('x_', '') for gene in ExpressionData.index]

            geneDF = pd.DataFrame(geneDict, index = ExpressionData.index)
            geneDF.to_csv(SCRIBE_GENE_FILE,
                          sep = ',', header = True)

    def run(self):
        '''
        Function to run SCRIBE algorithm.
        To see all the inputs runScribe.R script takes, run:
        docker run scribe:base /bin/sh -c "Rscript runScribe.R -h"
        '''

        # required inputs
        delay = str(self.params['delay'])
        method = str(self.params['method'])
        low = str(self.params['lowerDetectionLimit'])
        fam = str(self.params['expressionFamily'])

        # Build the command to run Scribe
        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns

        for idx in range(len(colNames)):
            # Specify file names for inputs and outputs
            exprName = "ExpressionData"+str(idx)+".csv"
            cellName = "pseudoTimeData"+str(idx)+".csv"
            outFile = "outFile"+str(idx)+".csv"
            timeFile = 'time'+str(idx)+".txt"

            cmdToRun = ' '.join(["/usr/bin/time -v -o", f"{self.working_dir}/{timeFile}",
                           str(_ENV_RSCRIPT), str(_SCRIBE_SCRIPT),
                           '-e', f"{self.working_dir}/{exprName}", '-c', f"{self.working_dir}/{cellName}",
                           '-g', f"{self.working_dir}/GeneData.csv", '-o', f"{self.working_dir}/", '-d', delay, '-l', low,
                           '-m', method, '-x', fam, '--outFile', outFile])

            if str(self.params['log']) == 'True':
                cmdToRun += ' --log'
            if str(self.params['ignorePT']) == 'True':
                cmdToRun += ' -i'

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from SCRIBE.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns
        OutSubDF = [0]*len(colNames)
        for idx in range(len(colNames)):
            # Read output
            outFile = 'outFile'+str(idx)+'.csv'
            if not (workDir / outFile).exists():
                # Quit if output file does not exist
                print(str(workDir / outFile) + ' does not exist, skipping...')
                return
            OutSubDF[idx] = pd.read_csv(workDir / outFile, sep = ' ', header = None)

        # megre the dataframe by taking the maximum value from each DF
        # From here: https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
        outDF = pd.concat(OutSubDF)
        outDF.columns= ['Gene1','Gene2','EdgeWeight']
        # Group by rows code is from here:
        # https://stackoverflow.com/questions/53114609/pandas-how-to-remove-duplicate-rows-but-keep-all-rows-with-max-value
        res = outDF[outDF['EdgeWeight'] == outDF.groupby(['Gene1','Gene2'])['EdgeWeight'].transform('max')]
        # Sort values in the dataframe
        finalDF = res.sort_values('EdgeWeight',ascending=False)

        self._write_ranked_edges(finalDF[['Gene1', 'Gene2', 'EdgeWeight']])
