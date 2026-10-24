#!/bin/bash
# waits for chunk downloads, appends parts to the partial files in order, verifies size + md5 vs Zenodo
ROOT=${TWINFER_PROJECT_ROOT:-/gpfs/projects/b1255/hzhang/TwINFER_KA}
cd "${PETRACE_DIR:-$ROOT/final_figure_data/real_data_analysis_and_melanoma/raw_data/PeTrace_zenodo_23088914}"      # folder that holds the .h5td files, parts/, sizes.txt
while ! grep -q ALLDONE chunk_dl.log; do sleep 20; done
declare -A loc=( [E9.5-R1.h5td]=E9.5-R1.h5td [E9.5-R3.h5td]=E9.5-R3.h5td [E9.5-R2.h5td]=E9.5-R2_redownload/E9.5-R2.h5td )
declare -A md5=( [E9.5-R1.h5td]=77643a7ca61b4c37606f1df870584245 [E9.5-R2.h5td]=ce22bea6e4dd211e7e03f0cefbcb8a69 [E9.5-R3.h5td]=f36209ed4584fe613251cd22e595a988 )
for n in E9.5-R1.h5td E9.5-R3.h5td E9.5-R2.h5td; do
  want=$(awk -v n=$n '$1==n{print $2}' sizes.txt)
  cat parts/$n.* >> ${loc[$n]}
  got=$(stat -c%s ${loc[$n]})
  sum=$(md5sum ${loc[$n]} | cut -d' ' -f1)
  if [ "$got" = "$want" ] && [ "$sum" = "${md5[$n]}" ]; then echo "$n OK size=$got md5=$sum"; else echo "$n FAIL size=$got (want $want) md5=$sum (want ${md5[$n]})"; fi
done > verify_result.txt 2>&1
echo VERIFYDONE >> verify_result.txt
