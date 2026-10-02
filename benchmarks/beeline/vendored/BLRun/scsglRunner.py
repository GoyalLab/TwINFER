import os
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner

# scSGL runs natively via a dedicated "scsgl" conda env (Python 3.8 + R 4.1 +
# r-pcaPP together, so rpy2 can find R at runtime) rather than through Docker,
# which isn't available on this HPC cluster.
_REPO_ROOT      = Path(__file__).resolve().parent.parent
_SCSGL_DIR      = _REPO_ROOT / "Algorithms" / "SCSGL"
_SCSGL_SCRIPT   = _SCSGL_DIR / "run_scSGL.py"
# Invoked by absolute path rather than `conda run -n scsgl` -- this host has
# more than one conda installation, and name-based env resolution can
# silently pick up the wrong one depending on which install's `conda` binary
# is first on PATH for a given shell/node. R_HOME and PATH are set explicitly
# (rather than relying on conda's activate.d hooks, which only run through
# `conda activate`/`conda run`) so rpy2 can still find this env's bundled R.
_ENV_PREFIX     = Path.home() / ".conda" / "envs" / "scsgl"
_ENV_PYTHON     = _ENV_PREFIX / "bin" / "python"
_ENV_R_HOME     = _ENV_PREFIX / "lib" / "R"


class SCSGLRunner(Runner):
    """Concrete runner for the scSGL GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for scSGL.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        SCSGL_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not SCSGL_EXPRESSION_FILE.exists():
            # input data
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            # Write gene expression data in SCSGL folder
            ExpressionData.to_csv(SCSGL_EXPRESSION_FILE,
                                 sep = ',', header  = True)

        SCSGL_GROUND_TRUTH_FILE = self.working_dir / "GroundTruthNetwork.csv"
        if not SCSGL_GROUND_TRUTH_FILE.exists():
            groundTruthNetworkData = pd.read_csv(self.ground_truth_file,
                                         header = 0, index_col = 0)

            # Write reference network data in SCSGL folder
            groundTruthNetworkData.to_csv(SCSGL_GROUND_TRUTH_FILE,
                                 sep = ',', header  = True)

    def run(self):
        '''
        Function to run SCSGL algorithm
        '''

        pos_density = str(self.params['pos_density'])
        neg_density = str(self.params['neg_density'])
        assoc = str(self.params['assoc'])

        cmdToRun = ' '.join([f"cd {_SCSGL_DIR} &&",
                            f"R_HOME={_ENV_R_HOME}",
                            f"PATH={_ENV_PREFIX}/bin:$PATH",
                            "/usr/bin/time -v -o", f"{self.working_dir}/time.txt",
                            str(_ENV_PYTHON), str(_SCSGL_SCRIPT),
                            f"--expression_file={self.working_dir}/ExpressionData.csv",
                            f"--ground_truth_net_file={self.working_dir}/GroundTruthNetwork.csv",
                            f"--out_file={self.working_dir}/outFile.txt",
                            '--pos_density='+pos_density, '--neg_density='+neg_density, '--assoc='+assoc])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from SCSGL.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        # Read output file
        OutDF = pd.read_csv(outFile, sep = '\t', header = 0)

        OutDF.sort_values(by="EdgeWeight", ascending=False, inplace=True)

        self._write_ranked_edges(OutDF[['Gene1', 'Gene2', 'EdgeWeight']])
