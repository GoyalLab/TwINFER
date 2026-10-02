import os
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner

# GENIE3 runs natively via the arboreto Python package rather than through
# Docker, which isn't available on this HPC cluster. arboreto/pandas/distributed
# are installed in the "grnboost310" conda env, mirroring what the
# grnbeeline/arboreto:base image installs via `conda install -c bioconda arboreto pandas`.
_REPO_ROOT       = Path(__file__).resolve().parent.parent
_ARBORETO_SCRIPT = _REPO_ROOT / "Algorithms" / "ARBORETO" / "runArboreto.py"
# Invoked by absolute path rather than `conda run -n grnboost310`: this host
# has more than one conda installation, and name-based env resolution can
# silently pick up the wrong one (or fail to find pandas at all) depending on
# which install's `conda` binary is first on PATH for a given shell/node.
_ENV_PYTHON      = Path.home() / ".conda" / "envs" / "grnboost310" / "bin" / "python"


class GENIE3Runner(Runner):
    """Concrete runner for the GENIE3 GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GENIE3.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        GENIE3_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not GENIE3_EXPRESSION_FILE.exists():
            # input data
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            # Write .csv file — arboreto expects cells as rows, genes as columns
            ExpressionData.T.to_csv(GENIE3_EXPRESSION_FILE,
                                 sep = '\t', header  = True, index = True)

    def run(self):
        '''
        Function to run GENIE3 algorithm
        '''

        cmdToRun = ' '.join(["/usr/bin/time -v -o", f"{self.working_dir}/time.txt",
                            str(_ENV_PYTHON), str(_ARBORETO_SCRIPT), "--algo=GENIE3",
                            f"--inFile={self.working_dir}/ExpressionData.csv",
                            f"--outFile={self.working_dir}/outFile.txt"])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from GENIE3.
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
