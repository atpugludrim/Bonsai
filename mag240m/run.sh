#!/bin/bash
# run on linux with `savelog` installed or change the logfile name manually
gpu=$1
rotate_logs(){
  local folder="$1"
  local file="$2"
  N=5
  if [ -z "$folder" ]; then
    echo "Error: No folder specified"
    return 1
  fi
  if [ ! -d "$folder" ]; then
    echo "Error: The specified folder doesn't exist"
    return 1
  fi
  savelog -n -c "${N}" "${folder}/${file}"
  # assumes that you have savelog
}
datasets=("mag240m")
mag240m_root="/DATATWO/datasets/mag240m"
sizes=(0.005 0.01 0.03)
for dataset in ${datasets[@]}
do
  dsdir="logs/${dataset}"
  mkdir -p "${dsdir}"
  for size in ${sizes[@]}
  do
    logdir="${dsdir}/frac-${size}"
    mkdir -p "${logdir}"
    log="log"
    rotate_logs "${logdir}" "${log}"
    logfile="${logdir}/${log}"
    echo "Writing logs to ${logfile}"
    printf "${dataset}-${size}\n" | tee "${logfile}"
    t1="$(date +'%s.%N')"
    # Run the Python script with mprof
    # mprof run --output memory_profile.dat python3 -m pdb -c cont main.py \
    # use main.py for BONSAI condensation and training
    CUDA_VISIBLE_DEVICES="${gpu}" python3 main.py \
        --target_size_frac "${size}" \
        --dataset "${dataset}" \
        --nepochs 500 \
        --mag240m_root "${mag240m_root}" \
        2> >(while read line; do echo "err: $line"; done >&1) > >(while read line; do echo "$line"; done >&1) | tee -a "${logfile}" # for logging. optional.
        # ^^ This syntax diverts both stdout and stderr to "stdout" while appending "err: " at the beginning of stderr lines
        #    and then `| tee -a "${logfile}"` copies the stream into the logfile as well.
        #    This syntax works in bash.
    t2="$(date +'%s.%N')"
    dur=$(echo "${t2}-${t1}" | bc)
    printf "It took %ss\n" "${dur}" | tee -a "${logfile}"
  done
done
