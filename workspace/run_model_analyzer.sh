#/bin/bash
seperator=---------------------------------------------------------------
seperator=$seperator$seperator
pattern="%-24s| %-24s| %-7s| %-7s| %-7s| %-7s|\n"
TableWidth=87

export CPU_COUNT=$(nproc)
export STEP_CONCURRENCY=${CPU_COUNT}
export CONCURRENCY_RUNS=16
export MIN_CONCURRENCY=${CPU_COUNT}
export MAX_CONCURRENCY=$((${CPU_COUNT}*${CONCURRENCY_RUNS}))
export MODEL_MANIFEST="fpenet facenet_ensemble"


function traverse_input(){
for row in $(echo $@ | jq -r '.input[] | @base64'); do
    ARG=""
    _jq() {
        echo ${row} | base64 --decode | jq -r ${1} 
    }
    shape=$(_jq '.dims' | jq -r 'join(",")')
    if [[ $shape -ge 0 ]]; then
        ARG=${ARG:+$ARG }" --shape $(_jq '.name'):$shape"
    else
        ARG=${ARG:+$ARG }" --shape $(_jq '.name'):16"
    fi
    echo $ARG
done
}

function traverse_output(){
for row in $(echo $@ | jq -r '.output[] | @base64'); do
    unset ARG
    _jq() {
        echo ${row} | base64 --decode | jq -r ${1} 
    }
    shape=$(_jq '.dims' | jq -r 'join(",")')
    if [[ $shape -ge 0 ]]; then
        ARG=${ARG:+$ARG }" --shape $(_jq '.name'):$shape"
    else
        ARG=${ARG:+$ARG }" --shape $(_jq '.name'):16"
    fi
    echo $ARG
done
}

clear

printf "$pattern" Name Platform Inputs Outputs Batch Status
printf "%.${TableWidth}s\n" "$seperator"
for MODEL in $MODEL_MANIFEST
do
    config=$(curl -s $TRITON_SERVER_IP:8000/v2/models/$MODEL/config)
    name=$(echo "${config}" | jq -r '.name')
    platform=$(echo "${config}" | jq -r '.platform')
    batchsize=$(echo "${config}" | jq -r '.max_batch_size')
    inputs=$(echo "${config}" | jq -r '.input | length')
    outputs=$(echo "${config}" | jq -r '.output | length')
    seq_check=$(echo ${config} | jq '.sequence_batching | length')
    
    if [[ $seq_check -gt 0 ]]; then
        unset batchsize
        batchsize=1
    elif [[ $batchsize -le 0 ]]; then 
        unset batchsize
        batchsize=1
    fi
    code_status=$(curl -m 1 -L -s -o /dev/null -w %{http_code} $TRITON_SERVER_IP:8000/v2/models/$MODEL/versions/1/ready)
    status=$([ "$code_status" == 200 ] && echo OK || echo $code_status)
    printf "$pattern" $name $platform "${inputs}" "${outputs}" $batchsize $status

    extra_args=$(traverse_input $config)
    perf_analyzer \
        -m $MODEL \
        -a \
        -i grpc \
        -u $TRITON_SERVER_IP:8001 \
        --max-threads $(($(nproc)*4)) \
        --request-distribution constant \
        --measurement-interval 30000 \
        --concurrency-range $MIN_CONCURRENCY:$MAX_CONCURRENCY:$STEP_CONCURRENCY \
        -b $batchsize $extra_args > /dev/null 2>&1 &
done
printf "%.${TableWidth}s\n" "$seperator"