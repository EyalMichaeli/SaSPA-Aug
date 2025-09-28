#!/bin/bash
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

############################################################################################################
# Define the hyperparameter values

# dataset="cars" # options: planes, cars, cub, dtd, compcars-parts
net="resnet50"
gpu_id="0"
run_name="saspa"
# iterate over
seeds=("1" "2" "3")
train_sample_ratios=("1.0")
stop_aug_after_epoch=("1000")

# this script runs the best baseline on all datasets

for dataset in "planes" "cars" "compcars-parts" "dtd" "cub" 
    do
    echo "Running for dataset: $dataset"
    if [ "$dataset" == "planes" ]; then
        special_augs=("classic")
        aug_sample_ratios=("0.4")
    elif [ "$dataset" == "cars" ]; then
        special_augs=("randaug-cutmix")  # similar results to classic-cutmix
        aug_sample_ratios=("0.4")
    elif [ "$dataset" == "compcars-parts" ]; then
        special_augs=("randaug-cutmix")
        aug_sample_ratios=("0.4")
    elif [ "$dataset" == "cub" ]; then
        special_augs=("classic")
        aug_sample_ratios=("0.1")
    elif [ "$dataset" == "dtd" ]; then
        special_augs=("classic") # similar results to classic-cutmix
        aug_sample_ratios=("0.4")
    elif [ "$dataset" == "planes_biased" ]; then
        special_augs=("classic")
        aug_sample_ratios=("0.4")
    else
        echo "Dataset not recognized"
        exit 1
    fi

    # use aug ratio = 0 if u want to use the original dataset
    # aug_sample_ratios=("0.0")

    limit_aug_per_image_list=("2")
    run_name_to_use="$run_name-$net"

    ############################################################################################################

    # add to run name the net
    echo "Running with aug_json: $aug_json and run_name: $run_name"

    # set the gpu environment variable
    export CUDA_VISIBLE_DEVICES=$gpu_id

    # Run the training 
    for train_sample_ratio in "${train_sample_ratios[@]}"
    do
        echo "Running with train_sample_ratio: $train_sample_ratio"
        for special_aug in "${special_augs[@]}"
        do
            echo "Running with special_aug: $special_aug"
            for aug_sample_ratio in "${aug_sample_ratios[@]}"
            do
                echo "Running with aug_sample_ratio: $aug_sample_ratio"
                for stop_aug_after_epoch in "${stop_aug_after_epoch[@]}"
                do
                    echo "Running with stop_aug_after_epoch: $stop_aug_after_epoch"
                    for limit_aug_per_image in "${limit_aug_per_image_list[@]}"
                    do
                        echo "Running with limit_aug_per_image: $limit_aug_per_image"
                        for seed in "${seeds[@]}"
                        do
                            echo "Running with seed: $seed"
                            run_name_to_use="$run_name-$net-train_$train_sample_ratio-aug_ratio_$aug_sample_ratio-$special_aug"
                            echo "Running with seed: $seed and train_sample_ratio: $train_sample_ratio and special_aug: $special_aug and aug_sample_ratio: $aug_sample_ratio"
                            python fgvc/train.py \
                                --seed $seed \
                                --train_sample_ratio $train_sample_ratio \
                                --logdir logs/$dataset/$run_name_to_use \
                                --special_aug $special_aug \
                                --dataset $dataset \
                                --stop_aug_after_epoch $stop_aug_after_epoch \
                                --limit_aug_per_image $limit_aug_per_image \
                                --net $net
                            wait # Wait for the previous training process to finish before starting the next one
                            echo "Finished run with seed: $seed"
                        done
                    done
                done
            done
        done
    done
done
    ###############################################################################################################################################################





echo "Finished running all the trainings"

# run with 
# bash fgvc/trainings_scripts/consecutive_runs_best_classic_aug.sh
# Or with nohup:
# chmod +x fgvc/trainings_scripts/consecutive_runs_best_classic_aug.sh
# nohup fgvc/trainings_scripts/consecutive_runs_best_classic_aug.sh > fgvc/trainings_scripts/consecutive_runs_best_classic_aug.log 2>&1 &
