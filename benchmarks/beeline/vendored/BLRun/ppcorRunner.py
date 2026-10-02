import os
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner

# PPCOR runs natively via R (module load R/4.2.3 on Quest) rather than through
# Docker, which isn't available on this HPC cluster. The ppcor CRAN package is
# installed into a project-local R library (r_lib/), shared with LEAP.
_REPO_ROOT     = Path(__file__).resolve().parent.parent
_PPCOR_SCRIPT  = _REPO_ROOT / "Algorithms" / "PPCOR" / "runPPCOR.R"
_R_LIBS_USER   = _REPO_ROOT / "r_lib"


class PPCORRunner(Runner):
    """Concrete runner for the PPCOR GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for PPCOR.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        PPCOR_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not PPCOR_EXPRESSION_FILE.exists():
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            newExpressionData = ExpressionData.copy()

            # Write .csv file
            newExpressionData.to_csv(PPCOR_EXPRESSION_FILE,
                                 sep = ',', header  = True, index = True)

    def run(self):
        '''
        Function to run PPCOR algorithm
        '''

        inner = ' '.join([f"R_LIBS_USER={_R_LIBS_USER}",
                          "/usr/bin/time -v -o", f"{self.working_dir}/time.txt",
                          "Rscript", str(_PPCOR_SCRIPT),
                          f"{self.working_dir}/ExpressionData.csv", f"{self.working_dir}/outFile.txt"])
        # bash -lc ensures Lmod's `module` function is available and that
        # module-provided R takes priority over any R bundled in a conda env
        # BLRunner.py might itself be running under (BEELINE's environment.yml
        # pins an old r=3.5.0, which otherwise silently shadows the module's
        # Rscript on PATH).
        cmdToRun = f'bash -lc "module load R/4.2.3 && {inner}"'

        # Run command
        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from PPCOR.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        # Read output. EdgeWeight is the raw corVal for every pair, unthresholded --
        # same convention as every other algorithm's rankedEdges.csv. The p-value-based
        # significance decision belongs to scoring-time consumers (e.g. the natural-
        # threshold metric in benchmark_network_sweep.ipynb), which read `pValue`
        # straight from this same outFile.txt rather than from a pre-filtered EdgeWeight.
        OutDF = pd.read_csv(outFile, sep = '\t', header = 0)
        OutDF = OutDF.assign(absCorVal = OutDF['corVal'].abs())
        OutDF = OutDF.sort_values('absCorVal', ascending=False)

        self._write_ranked_edges(
            OutDF[['Gene1', 'Gene2']].assign(EdgeWeight=OutDF['corVal'])[['Gene1', 'Gene2', 'EdgeWeight']]
        )
