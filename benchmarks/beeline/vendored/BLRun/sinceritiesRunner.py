import os
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner

# SINCERITIES runs natively via R (module load R/4.2.3 on Quest) rather than
# through Docker, which isn't available on this HPC cluster. Its R packages
# are installed into the shared project-local r_lib/. SINCERITIES' own source
# is vendored in the repo as SINCERITIES.zip (COPY'd by the Dockerfile, not
# fetched externally), unzipped here into SINCERITIES_src/. MAIN.R loads
# "SINCERITIES functions/*.R" via relative paths, so it must be run with that
# directory as cwd (mirroring the Dockerfile's WORKDIR /SINCERITIES/).
_REPO_ROOT       = Path(__file__).resolve().parent.parent
_SINCERITIES_SRC = _REPO_ROOT / "Algorithms" / "SINCERITIES" / "SINCERITIES_src"
_R_LIBS_USER     = _REPO_ROOT / "r_lib"


class SINCERITIESRunner(Runner):
    """Concrete runner for the SINCERITIES GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SINCERITIES.
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
            newExpressionData = ExpressionData.loc[:,index].T
            # Perform quantile binning as recommeded in the paper
            # http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html#pandas.qcut
            nBins = int(self.params['nBins'])
            tQuantiles = pd.qcut(PTData.loc[index,colName], q = nBins, duplicates ='drop')
            mid = [(a.left + a.right)/2 for a in tQuantiles]

            newExpressionData['Time'] = mid
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def run(self):
        '''
        Function to run SINCERITIES algorithm
        '''

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            inner = ' '.join([f"cd {_SINCERITIES_SRC} &&",
                              f"R_LIBS_USER={_R_LIBS_USER}",
                              "/usr/bin/time -v -o", f"{self.working_dir}/time{idx}.txt",
                              "Rscript MAIN.R",
                              f"{self.working_dir}/ExpressionData{idx}.csv",
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
        Function to parse outputs from SINCERITIES.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)
        colNames = PTData.columns
        OutSubDF = [0]*len(colNames)
        for idx in range(len(colNames)):
            # Read output
            outFile = 'outFile'+str(idx)+'.txt'
            if not (workDir / outFile).exists():
                # Quit if output file does not exist
                print(str(workDir / outFile) + ' does not exist, skipping...')
                return
            OutSubDF[idx] = pd.read_csv(workDir / outFile, sep = ',', header = 0)

        # megre the dataframe by taking the maximum value from each DF
        # From here: https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
        outDF = pd.concat(OutSubDF)
        # Group by rows code is from here:
        # https://stackoverflow.com/questions/53114609/pandas-how-to-remove-duplicate-rows-but-keep-all-rows-with-max-value
        res = outDF[outDF['Interaction'] == outDF.groupby(['SourceGENES','TargetGENES'])['Interaction'].transform('max')]
        # Sort values in the dataframe
        finalDF = res.sort_values('Interaction',ascending=False)
        finalDF.drop(labels = 'Edges',axis = 'columns', inplace = True)
        # SINCERITIES output is incorrectly orderd
        finalDF.columns = ['Gene2','Gene1','EdgeWeight']
        self._write_ranked_edges(finalDF[['Gene1', 'Gene2', 'EdgeWeight']])
