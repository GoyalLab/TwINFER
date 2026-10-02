import os
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner

# GRNBoost2 runs natively via the arboreto Python package rather than through
# Docker, which isn't available on this HPC cluster. See genie3Runner.py for
# details — both algorithms share the same runArboreto.py entry point and
# conda env.
_REPO_ROOT       = Path(__file__).resolve().parent.parent
_ARBORETO_SCRIPT = _REPO_ROOT / "Algorithms" / "ARBORETO" / "runArboreto.py"
# Invoked by absolute path rather than `conda run -n grnboost310` -- see
# genie3Runner.py for why (multiple conda installations on this host make
# name-based env resolution unreliable depending on shell/node).
_ENV_PYTHON      = Path.home() / ".conda" / "envs" / "grnboost310" / "bin" / "python"


class GRNBoost2Runner(Runner):
    """Concrete runner for the GRNBoost2 GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GRNBoost2.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        GRNBOOST2_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not GRNBOOST2_EXPRESSION_FILE.exists():
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            # Write .csv file
            ExpressionData.T.to_csv(GRNBOOST2_EXPRESSION_FILE,
                                 sep = '\t', header  = True, index = True)

    def run(self):
        '''
        Function to run GRNBOOST2 algorithm
        '''

        cmdToRun = ' '.join(["/usr/bin/time -v -o", f"{self.working_dir}/time.txt",
                            str(_ENV_PYTHON), str(_ARBORETO_SCRIPT), "--algo=GRNBoost2",
                            f"--inFile={self.working_dir}/ExpressionData.csv",
                            f"--outFile={self.working_dir}/outFile.txt"])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from GRNBOOST2.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        # Read output
        OutDF = pd.read_csv(outFile, sep = '\t', header = 0)

        self._write_ranked_edges(OutDF.rename(columns={
            'TF': 'Gene1', 'target': 'Gene2', 'importance': 'EdgeWeight'
        })[['Gene1', 'Gene2', 'EdgeWeight']])
