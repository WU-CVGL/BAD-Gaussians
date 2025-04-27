SCENE_DIR="/datasets/tencent/data-real"
RESULT_DIR="results/benchmark_mcmc_500k/dbnerf_real_cubic4"
SCENE_LIST="blurball blurbasket blurbuick blurcoffee blurdecoration blurgirl blurheron blurparterre blurpuppet blurstair"

CAP_MAX=500000

for SCENE in $SCENE_LIST;
do
    GPUS="0"
    DATA_FACTOR=4
    SCALE_FACTOR=0.25
    TRAJ_TYPE="cubic"

    echo "Running $SCENE"

    # train
    CUDA_VISIBLE_DEVICES=$GPUS python simple_trainer_deblur.py mcmc \
        --disable_viewer \
        --data_factor $DATA_FACTOR \
        --scale_factor $SCALE_FACTOR \
        --camera-optimizer.mode $TRAJ_TYPE \
        --strategy.cap-max $CAP_MAX \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done


for SCENE in $SCENE_LIST;
do
    echo "=== NVS Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/nvs*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done
