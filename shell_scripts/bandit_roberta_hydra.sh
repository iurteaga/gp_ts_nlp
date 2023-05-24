#!/bin/sh

#################### PROCESS SCRIPT ARGUMENTS  ########################
# Helper for script
helpFunction()
{
    echo ""
    echo "Usage: $0 -b PATH_TO_BANDIT_CONFIG -i MAX_INTERACTIONS -p PATH_TO_PRETRAIN_CONFIG -d PATH_TO_PRETRAIN_DATA -f PATH_TO_FINETUNE_CONFIG_DIR -e EVALUATE -s SEED -m MACHINE"
    echo "      -b Path to bandit configuration file to use"
    echo "      -i Max number of pre-training interactions to run"
    echo "      -u Number of max updates per-interaction to run"
    echo "      -r Number of updates to start from (when restarting)"
    echo "      -p Path to fairseq pre-training configuration file to use"
    echo "      -d Path to pre-training dataset to use"
    echo "      -f Path to fairseq fine-tuning directory with configuration files to use"
    echo "      -e Whether to evaluate after finetuning"
    echo "      -s Seed to use"
    echo "      -m Machine where to run scripts"
    echo "      -l Whether to load an already pre-trained model to start from"
    exit 1 # Exit script after printing help
}

# Process script options
while getopts b:i:u:p:d:f:e:s:m:l: arg
do
    case "$arg" in
        b ) PATH_TO_BANDIT_CONFIG="$OPTARG" ;;
        i ) MAX_INTERACTIONS="$OPTARG" ;;
        u ) MAX_UPDATES="$OPTARG" ;;
        r ) RESTART_UPDATES="$OPTARG" ;;
        p ) PATH_TO_PRETRAIN_CONFIG="$OPTARG" ;;
        d ) PATH_TO_PRETRAIN_DATA="$OPTARG" ;;
        f ) PATH_TO_FINETUNE_CONFIG_DIR="$OPTARG" ;;
        e ) EVALUATE="$OPTARG" ;;
        s ) SEED="$OPTARG" ;;
        m ) MACHINE="$OPTARG" ;;
        l ) LOAD_PRETRAINED_MODEL="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case argument is non-existent
    esac
done

# If max interactions is not provided, just run for 10
if [ -z "$MAX_INTERACTIONS" ]
then
    MAX_INTERACTIONS=10
fi

# If max updates is not provided
if [ -z "$MAX_UPDATES" ]
then
    #set max update counter at high number
    MAX_UPDATE_COUNT=1000000
else
    # If restart updates is not provided
    if [ -z "$RESTART_UPDATES" ]
    then
        # Init at 0
        MAX_UPDATE_COUNT=0
    else
        # Init at provided restart updates
        MAX_UPDATE_COUNT=`echo "$RESTART_UPDATES" | xargs`
    fi
fi

# Print helpFunction in case config arguments are empty
if [ -z "$PATH_TO_BANDIT_CONFIG" ] || [ -z "$PATH_TO_PRETRAIN_CONFIG" ] || [ -z "$PATH_TO_PRETRAIN_DATA" ]
then
    echo "Some or all of the arguments are empty";
    helpFunction
fi

# If EVALUATE is not provided, just assume false
if [ -z "$EVALUATE" ]
then
    EVALUATE='false'
fi

# If SEED is not provided, just assume 1
if [ -z "$SEED" ]
then
    SEED=1
fi

# If MACHINE is not provided, just assume ginsburg
if [ -z "$MACHINE" ]
then
    MACHINE='ginsburg'
fi

# If LOAD_PRETRAINED_MODEL does not exist
if [ -z "$LOAD_PRETRAINED_MODEL" ]
then
    LOAD_PRETRAINED_MODEL='null'
fi

#################### SET UP DIRs, PYTHON and PATHs ########################
# TODO: The following need to be personalized to each user's server configuration
if [ $MACHINE = 'habanero' ]
then
    # Main nlp_bandit_dir in habanero
    export NLP_BANDIT_DIR=/rigel/dsi/projects/nlp_bandit
    
    # Update pythonpath for singularity image
    export PYTHONPATH=$PYTHONPATH:$NLP_BANDIT_DIR/nlp_bandit_singularity/lib/python3.8/site-packages
    # Update path for fairseq
    export PATH=$PATH:$NLP_BANDIT_DIR/nlp_bandit_singularity/bin
    
    python --version
elif [ $MACHINE = 'ginsburg' ]
then
    # Main nlp_bandit_dir in habanero
    export NLP_BANDIT_DIR=/burg/dsi/users/iu2153/gp_ts_nlp
    
    python --version
# google-colab set-up here
elif [ $MACHINE = 'colab' ]
then
    # Main nlp_bandit_dir in colab
    export NLP_BANDIT_DIR=/content/nlp_bandit

    python --version
else
    # Assuming local
    export NLP_BANDIT_DIR=..
fi

# Execution path is nlp_bandit_dir
cd $NLP_BANDIT_DIR

# To help debug Fairseq
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

#################### SCRIPT CONFIG ########################
# Cleaning vars
PATH_TO_BANDIT_CONFIG=`echo "$PATH_TO_BANDIT_CONFIG" | xargs`
MAX_INTERACTIONS=`echo "$MAX_INTERACTIONS" | xargs`
MAX_UPDATE_COUNT=`echo "$MAX_UPDATE_COUNT" | xargs`
PATH_TO_PRETRAIN_CONFIG=`echo "$PATH_TO_PRETRAIN_CONFIG" | xargs`
PATH_TO_PRETRAIN_DATA=`echo "$PATH_TO_PRETRAIN_DATA" | xargs`
PATH_TO_FINETUNE_CONFIG_DIR=`echo "$PATH_TO_FINETUNE_CONFIG_DIR" | xargs`
EVALUATE=`echo "$EVALUATE" | xargs`
SEED=`echo "$SEED" | xargs`
MACHINE=`echo "$MACHINE" | xargs`
LOAD_PRETRAINED_MODEL=`echo "$LOAD_PRETRAINED_MODEL" | xargs`

# Output dir is based on pretrain, finetune and bandit configs (plus seed)
# Pretrain
PRETRAIN_CONFIG=${PATH_TO_PRETRAIN_CONFIG#*"fairseq_hydra/$MACHINE/"}
# Config name
if [ -z "$MAX_UPDATES" ]
then
    PRETRAIN_CONFIG=${PRETRAIN_CONFIG%".yaml"}
else
    MAX_UPDATES=`echo "$MAX_UPDATES" | xargs`
    PRETRAIN_CONFIG="${PRETRAIN_CONFIG%".yaml"}"_u"$MAX_UPDATES"
fi
PRETRAIN_CONFIG_FILENAME=`basename "$PATH_TO_PRETRAIN_CONFIG"`
PRETRAIN_CONFIG_NAME=`basename "$PATH_TO_PRETRAIN_CONFIG" .yaml`
#IF Fine-tuning
if [ ! -z "$PATH_TO_FINETUNE_CONFIG_DIR" ]
then
    FINETUNE_CONFIG_DIR=${PATH_TO_FINETUNE_CONFIG_DIR#*"fairseq_hydra/$MACHINE/"}
    FINETUNE_CONFIG_DIR=${FINETUNE_CONFIG_DIR%".yaml"}
    FINETUNE_CONFIG_NAME=`basename "$FINETUNE_CONFIG_DIR"`
    FAIRSEQ_EXPERIMENT_NAME="$PRETRAIN_CONFIG"_"$FINETUNE_CONFIG_NAME"
else
    FAIRSEQ_EXPERIMENT_NAME="$PRETRAIN_CONFIG"
fi

# Bandit config
BANDIT_CONFIG=${PATH_TO_BANDIT_CONFIG#*"bandit_config/"}

# Main experiment name
EXPERIMENT_NAME="$FAIRSEQ_EXPERIMENT_NAME/$BANDIT_CONFIG"

# Script name
SCRIPT_NAME=`basename "$0"`
SCRIPT_NAME=${SCRIPT_NAME%".sh"}
# Data
DATA_NAME=${PATH_TO_PRETRAIN_DATA#*"data-bin/"}

# Whether we are loading a model
LOAD_MODEL_NAME='new'
if [ $LOAD_PRETRAINED_MODEL != 'null' ]
then
    LOAD_MODEL_NAME=`basename "$LOAD_PRETRAINED_MODEL"`
    LOAD_MODEL_NAME=${LOAD_MODEL_NAME%".pt"}
fi

# Output dir
OUTPUT_DIR="$NLP_BANDIT_DIR/nlp_bandit_experiments/$DATA_NAME/$SCRIPT_NAME/$EXPERIMENT_NAME/$LOAD_MODEL_NAME/seed_$SEED"
if [ -d $OUTPUT_DIR ]
then
    echo "OUTPUT_DIR=$OUTPUT_DIR already exists!"
else
    echo "New OUTPUT_DIR=$OUTPUT_DIR"
    
    # Make sure it's new and emptied
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR
fi

# Job output file
OUTFILE=$OUTPUT_DIR/job_output.out
# STDOUT and STDERR redirection
set -o errexit
if [ -f $OUTFILE ]
then
    echo "Job output file $OUTFILE exists."
    
    # Append
    exec 1>>$OUTFILE
    exec 2>&1
else
    # Redirect
    exec 1>$OUTFILE
    exec 2>&1
fi

# Fairseq config
PATH_TO_BANDIT_PRETRAIN_CONFIG="$OUTPUT_DIR/$PRETRAIN_CONFIG_FILENAME"
# Replicate initial fairseq config within output dir
# NOTE: if this new config content differs from previous, we will simply continue with new 
cp $PATH_TO_PRETRAIN_CONFIG $PATH_TO_BANDIT_PRETRAIN_CONFIG

# If we want to initialize from model
if [ $LOAD_PRETRAINED_MODEL != 'null' ]
then
    # Make sure checkpoint dir is created
    mkdir -p $OUTPUT_DIR/checkpoints
    
    # Copy initial model to directory within output dir
    cp $NLP_BANDIT_DIR/$LOAD_PRETRAINED_MODEL $OUTPUT_DIR/checkpoints/checkpoint_last.pt
fi

#IF Fine-tuning: prepare all finetune configs
if [ ! -z "$PATH_TO_FINETUNE_CONFIG_DIR" ]
then
    FINETUNE_DIR="$OUTPUT_DIR/$FINETUNE_CONFIG_DIR"
    for PATH_TO_FINETUNE_CONFIG_FILE in "$PATH_TO_FINETUNE_CONFIG_DIR"/*
    do
        FINETUNE_CONFIG=${PATH_TO_FINETUNE_CONFIG_FILE#*"fairseq_hydra/$MACHINE/"}
        FINETUNE_CONFIG=${FINETUNE_CONFIG%".yaml"}
        FINETUNE_OUTPUT_DIR="$OUTPUT_DIR/$FINETUNE_CONFIG"
        FINETUNE_CONFIG_FILENAME=`basename "$PATH_TO_FINETUNE_CONFIG_FILE"`
        PATH_TO_BANDIT_FINETUNE_CONFIG="$FINETUNE_OUTPUT_DIR/$FINETUNE_CONFIG_FILENAME"
        # Finetune specific output dir 
        mkdir -p $FINETUNE_OUTPUT_DIR
        
        # Replicate initial config within fine-tuning output dir
        # so that bandit can edit without overwriting
        cp $PATH_TO_FINETUNE_CONFIG_FILE $PATH_TO_BANDIT_FINETUNE_CONFIG
    done
else
    FINETUNE_DIR="None"
fi

#################### START SCRIPT ########################
# Executed script info
echo "Starting script $0 at $(date) with"
echo "      -path_to_bandit_config $PATH_TO_BANDIT_CONFIG"
echo "      -max_interactions $MAX_INTERACTIONS"
echo "      -path_to_pretrain_config $PATH_TO_PRETRAIN_CONFIG"
echo "      -path_to_pretrain_data $PATH_TO_PRETRAIN_DATA"
echo "      -path_to_finetune_config_dir $PATH_TO_FINETUNE_CONFIG_DIR"
echo "      -evaluate $EVALUATE"
echo "      -seed $SEED"
echo "      -machine $MACHINE"
echo "      -load $LOAD_PRETRAINED_MODEL"

# Initialize bandit object
python3 ./nlp_bandit_scripts/init_gp_bandit.py -bandit_config $PATH_TO_BANDIT_CONFIG -output_dir $OUTPUT_DIR

# Start
echo "Starting Bandit Fairseq RoBERTa script for $MAX_INTERACTIONS interactions at $(date)"

for INTERACTION in $(seq 1 $MAX_INTERACTIONS);
do
    # Start interaction
    # Figure out interaction updates
    if [ -z "$MAX_UPDATES" ]
    then
        #set max update counter at high number
        MAX_UPDATE_COUNT=1000000
    else
        # Add max updates to count so far
        MAX_UPDATE_COUNT=$(($MAX_UPDATE_COUNT + $MAX_UPDATES))
    fi
    
    echo "Starting bandit interaction $INTERACTION/$MAX_INTERACTIONS up to $MAX_UPDATE_COUNT updates at $(date)"
    
    ### Pre-training
    echo "Starting pre-training";
    # Run pre-training, with specified seed
    fairseq-hydra-train -m --config-dir $OUTPUT_DIR \
        --config-name $PRETRAIN_CONFIG_NAME \
        task.data=$NLP_BANDIT_DIR/$PATH_TO_PRETRAIN_DATA \
        checkpoint.restore_file=$OUTPUT_DIR/checkpoints/checkpoint_last.pt \
        checkpoint.save_dir=$OUTPUT_DIR/checkpoints \
        common.seed=$(($SEED + $INTERACTION)) \
        optimization.max_update=$MAX_UPDATE_COUNT
    echo "Finished pre-training";

    ### Fine-tuning
    if [ ! -z "$PATH_TO_FINETUNE_CONFIG_DIR" ]
    then
        for PATH_TO_FINETUNE_CONFIG_FILE in "$PATH_TO_FINETUNE_CONFIG_DIR"/*
        do
            FINETUNE_CONFIG=${PATH_TO_FINETUNE_CONFIG_FILE#*"fairseq_hydra/$MACHINE/"}
            FINETUNE_CONFIG=${FINETUNE_CONFIG%".yaml"}
            FINETUNE_OUTPUT_DIR="$OUTPUT_DIR/$FINETUNE_CONFIG"
            FINETUNE_CONFIG_FILENAME=`basename "$PATH_TO_FINETUNE_CONFIG_FILE" .yaml`
            
            echo "Starting fine-tuning $FINETUNE_OUTPUT_DIR";
            # Run fine-tuning, with specified seed
            fairseq-hydra-train -m --config-dir $FINETUNE_OUTPUT_DIR \
                    --config-name $FINETUNE_CONFIG_FILENAME \
                    checkpoint.restore_file=$OUTPUT_DIR/checkpoints/checkpoint_last.pt \
                    checkpoint.save_dir=$FINETUNE_OUTPUT_DIR/checkpoints \
                    common.seed=$SEED
            echo "Finished fine-tuning $FINETUNE_OUTPUT_DIR";
            
            # Evaluate, if requested
            if [ $EVALUATE = 'true' ]
            then
                ### Evaluation in glue-task
                echo "Need to revisit GLUE evaluation: what data to use?!"
            fi
        done
    fi
    
    ### Bandit code
    # Use bandit to update fairseq config, based on last checkpoint
    python3 ./nlp_bandit_scripts/bandit_update_fairseq_hydra.py \
                    -path_to_bandit_config $PATH_TO_BANDIT_CONFIG \
                    -bandit_interaction $INTERACTION \
                    -path_to_output_dir $OUTPUT_DIR \
                    -path_to_pretrain_config $PATH_TO_BANDIT_PRETRAIN_CONFIG \
                    -path_to_finetune_config_dir $FINETUNE_DIR
    
    echo "Finished bandit interaction $INTERACTION/$MAX_INTERACTIONS at $(date)"
    
    ### Cleaning up finetuned checkpoints, to save on space
    if [ ! -z "$PATH_TO_FINETUNE_CONFIG_DIR" ]
    then
        for PATH_TO_FINETUNE_CONFIG_FILE in "$PATH_TO_FINETUNE_CONFIG_DIR"/*
        do
            FINETUNE_CONFIG=${PATH_TO_FINETUNE_CONFIG_FILE#*"fairseq_hydra/$MACHINE/"}
            FINETUNE_CONFIG=${FINETUNE_CONFIG%".yaml"}
            FINETUNE_OUTPUT_DIR="$OUTPUT_DIR/$FINETUNE_CONFIG"
            rm $FINETUNE_OUTPUT_DIR/checkpoints/checkpoint*.pt
        done
    fi
done

# Finished
echo "Done Bandit Fairseq RoBERTa script for $MAX_INTERACTIONS interactions at $(date)"

# Finished
echo "Done script $0 at $(date)"

# END
