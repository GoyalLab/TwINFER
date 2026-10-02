import hashlib
import os
from pathlib import Path

import pandas as pd

from BLRun.runner import Runner

# SINGE runs natively via its precompiled MATLAB-Compiler binaries rather
# than through Docker, which isn't available on this HPC cluster. Quest's
# `module load matlab/r2018a` is an exact version match (v9.4) for the
# binaries (fetched from the same public gitter-lab/SINGE GitHub release URL
# the Dockerfile uses), so no self-installed MCR is needed here (contrast
# GRISLI/GRNVBEM/JUMP3, which need MCR v9.6 — a version Quest has no
# installed MATLAB for at all). The CSV->mat conversion step (originally
# `octave` inside the container) runs via a conda-forge octave build, since
# Quest's own octave/8.3.0 module is missing libcholmod.so.2 (SuiteSparse)
# and fails to start.
_REPO_ROOT     = Path(__file__).resolve().parent.parent
_SINGE_DIR     = _REPO_ROOT / "Algorithms" / "SINGE"
_MATLAB_ROOT   = Path("/software/matlab/R2018a")
# Invoked by absolute path rather than `conda run -n octave_env` -- this host
# has more than one conda installation, and name-based env resolution can
# silently pick up the wrong one depending on which install's `conda` binary
# is first on PATH for a given shell/node. OCTAVE_HOME is set explicitly
# (rather than relying on conda's activate.d hook, which only runs through
# `conda activate`/`conda run`) to match what that hook itself does.
_OCTAVE_PREFIX = Path.home() / ".conda" / "envs" / "octave_env"
_OCTAVE_BIN    = _OCTAVE_PREFIX / "bin" / "octave"


class SINGERunner(Runner):
    """Concrete runner for the SINGE GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for SINGE.
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
            newExpressionData['PseudoTime'] = PTData.loc[index,colName]
            newExpressionData.to_csv(self.working_dir / exprName,
                                 sep = ',', header  = True, index = False)

    def run(self):
        '''
        Function to run SINGE algorithm
        '''

        # if the parameters aren't specified, then use default parameters
        # TODO allow passing in multiple sets of hyperparameters
        # these must be in the right order!
        params_order = [
            'lambda', 'dT', 'num_lags', 'kernel_width',
            'prob_zero_removal', 'prob_remove_samples',
            'family'
        ]
        default_params = {
            'lambda': '0.01',
            'dT': '10',
            'num_lags': '5',
            'kernel_width': '4',
            'prob_zero_removal': '0',
            'prob_remove_samples': '0.2',
            'family': 'gaussian',
            'num_replicates': '2',
        }
        params = self.params
        for param, val in default_params.items():
            if param not in params:
                params[param] = val

        num_replicates = int(params['num_replicates'])
        replicates = []
        for replicate in range(num_replicates):
           replicates.append(' '.join('--' + p.replace('_', '-') + ' ' + str(params[p]) for p in params_order) + ' '.join(['', '--replicate', str(replicate), '--ID', str(replicate)]))
        params_str = '\n'.join(replicates)

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        for idx in range(len(colNames)):
            os.makedirs(str(self.working_dir / str(idx)), exist_ok = True)

            inputFile = str(self.working_dir / f"ExpressionData{idx}.csv")
            inputMat = str(self.working_dir / f"ExpressionData{idx}.mat")
            geneListMat = str(self.working_dir / f"GeneList{idx}.mat")
            paramsFile = self.working_dir / "hyperparameters.txt"
            paramsFile.write_text(params_str + '\n')

            '''
            This is a workaround for https://github.com/gitter-lab/SINGE/blob/master/code/parseParams.m#L39
            not allowing '/' characters in the outDir parameter. The symlink is created inside
            _SINGE_DIR (alongside SINGE.sh) so a bare relative name can be passed as outdir.

            The symlink name is derived from a hash of self.working_dir rather than just
            `idx` so that concurrent runs (e.g. a Slurm array job running many
            dataset/algorithm combinations at once) don't clobber each other's symlink --
            every runner shares this one _SINGE_DIR, but idx alone (0, 1, 2, ...) repeats
            across every dataset.
            '''
            run_hash = hashlib.md5(str(self.working_dir).encode()).hexdigest()[:10]
            outFileSymlink = f"out_{run_hash}_{idx}"
            symlink_path = _SINGE_DIR / outFileSymlink
            if symlink_path.is_symlink() or symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(self.working_dir / str(idx))

            '''
            See https://github.com/gitter-lab/SINGE/blob/master/README.md.  SINGE expects a data matfile with variables "X" and "ptime",
            and a gene_list matfile with the variable "gene_list".

            Saving fullKp is a very hacky workaround for https://github.com/gitter-lab/SINGE/blob/master/code/iLasso_for_SINGE.m#L56,
            that assumes this input was saved in matfile v7.3 which octave does not support.
            '''
            convert_input_to_matfile = f'OCTAVE_HOME={_OCTAVE_PREFIX} {_OCTAVE_BIN} -q --eval "CSV = csvread(\'' + inputFile + '\'); ' + \
                                 'X = sparse(CSV(2:end,1:end-1).\'); ptime = CSV(2:end,end).\'; ' + \
                                 'Kp2.Kp = single(ptime); Kp2.sumKp = single(ptime*X.\'); fullKp(1, ' + \
                                 str(int(params['dT'])*int(params['num_lags'])) + ') = Kp2; ' + \
                                 'save(\'-v7\',\'' + inputMat + '\', \'X\', \'ptime\', \'fullKp\'); ' + \
                                 'f = fopen(\'' + inputFile + '\'); gene_list = strsplit(fgetl(f), \',\')(1:end-1).\'; fclose(f); ' + \
                                 'save(\'-v7\',\'' + geneListMat + '\', \'gene_list\')"'

            cmdToRun = ' '.join([convert_input_to_matfile,
                                '&&', f"cd {_SINGE_DIR} &&",
                                '/usr/bin/time -v -o', str(self.working_dir / f"time{idx}.txt"),
                                './SINGE.sh', str(_MATLAB_ROOT), 'standalone',
                                inputMat, geneListMat, outFileSymlink, str(paramsFile)])

            self._run_docker(cmdToRun, append=(idx > 0))

    def parseOutput(self):
        '''
        Function to parse outputs from SINGE.
        '''
        workDir = self.working_dir

        PTData = pd.read_csv(self.input_dir / self.pseudoTimeData,
                             header = 0, index_col = 0)

        colNames = PTData.columns
        OutSubDF = [0]*len(colNames)

        for idx in range(len(colNames)):

            # Quit if output directory does not exist
            if not (workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt').exists():
                print(str(workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt') + ' does not exist, skipping...')
                return

            # Read output
            OutSubDF[idx] = pd.read_csv(workDir / str(idx) / 'SINGE_Ranked_Edge_List.txt',
                                sep = '\t', header = 0)
        # megre the dataframe by taking the maximum value from each DF
        # Code from here:
        # https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
        outDF = pd.concat(OutSubDF)
        outDF.columns= ['Gene1','Gene2','EdgeWeight']
        # Group by rows code is from here:
        # https://stackoverflow.com/questions/53114609/pandas-how-to-remove-duplicate-rows-but-keep-all-rows-with-max-value
        res = outDF[outDF['EdgeWeight'] == outDF.groupby(['Gene1','Gene2'])['EdgeWeight'].transform('max')]
        # Sort values in the dataframe
        finalDF = res.sort_values('EdgeWeight', ascending=False)
        self._write_ranked_edges(finalDF[['Gene1', 'Gene2', 'EdgeWeight']])
