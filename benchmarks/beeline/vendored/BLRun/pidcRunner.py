import os
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner

# PIDC runs natively via Julia (module load julia/1.10.2 on Quest) rather than
# through Docker, which isn't available on this HPC cluster. NetworkInference
# and LightGraphs are installed into JULIA_DEPOT below, mirroring what the
# grnbeeline/pidc:base image installs via Algorithms/PIDC/installPackages.jl.
_REPO_ROOT   = Path(__file__).resolve().parent.parent
_PIDC_SCRIPT = _REPO_ROOT / "Algorithms" / "PIDC" / "runPIDC.jl"
_JULIA_DEPOT = _REPO_ROOT / "julia_depot"


class PIDCRunner(Runner):
    """Concrete runner for the PIDC GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for PIDC.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        PIDC_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not PIDC_EXPRESSION_FILE.exists():
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            # A zero-variance (constant across every cell) gene breaks
            # InformationMeasures.jl's bin-width calculation in get_bin_ids!
            # (ArgumentError: indexed assignment with a single value to
            # possibly many locations is not supported) -- a degenerate range
            # for a discretizer that assumes some spread. Drop such genes
            # before handing PIDC the expression matrix; they carry no mutual
            # information with anything else anyway, so this only removes
            # genes PIDC could never have scored an edge for regardless.
            constant_genes = ExpressionData.index[ExpressionData.nunique(axis=1) <= 1]
            if len(constant_genes) > 0:
                print(f"PIDC: dropping {len(constant_genes)} zero-variance gene(s): "
                      f"{list(constant_genes)}")
                ExpressionData = ExpressionData.drop(index=constant_genes)

            ExpressionData.to_csv(PIDC_EXPRESSION_FILE,
                                 sep = '\t', header  = True, index = True)

    def run(self):
        '''
        Function to run PIDC algorithm
        '''

        cmdToRun = ' '.join([f"JULIA_DEPOT_PATH={_JULIA_DEPOT}",
                            "time -v -o", f"{self.working_dir}/time.txt",
                            "julia", str(_PIDC_SCRIPT),
                            f"{self.working_dir}/ExpressionData.csv", f"{self.working_dir}/outFile.txt"])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from PIDC.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        # Read output (headerless: col 0 = Gene1, col 1 = Gene2, col 2 = EdgeWeight)
        OutDF = pd.read_csv(outFile, sep = '\t', header = None)

        self._write_ranked_edges(pd.DataFrame({
            'Gene1':      OutDF[0],
            'Gene2':      OutDF[1],
            'EdgeWeight': OutDF[2],
        }))
