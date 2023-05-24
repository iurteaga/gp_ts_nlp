#!/bin/bash

#################### PROCESS SCRIPT ARGUMENTS  ########################
# Helper for script
helpFunction()
{
    echo ""
    echo "Usage: $0 -p PATH_TO_PRETRAINED_CKPT -f PATH_TO_FINETUNE_CONFIG_DIR -m MACHINE"
    echo "      -p Path to fairseq pre-trained model to load: note that this should be a full path!"
    echo "      -f Path to fairseq fine-tuning directory with configuration files to use"
    echo "      -e Whether to evaluate after finetuning"
    echo "      -s Seed to use"
    echo "      -m Machine where to run scripts"
    exit 1 # Exit script after printing help
}

# Process script options
while getopts p:f:e:s:m: arg
do
    case "$arg" in
        p ) PATH_TO_PRETRAINED_CKPT="$OPTARG" ;;
        f ) PATH_TO_FINETUNE_CONFIG_DIR="$OPTARG" ;;
        e ) EVALUATE="$OPTARG" ;;
        s ) SEED="$OPTARG" ;;
        m ) MACHINE="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case argument is non-existent
    esac
done

# Print helpFunction in case config arguments are empty
if [ -z "$PATH_TO_PRETRAINED_CKPT" ]
then
    echo "Some or all of the arguments are empty";
    helpFunction
fi

# If EVALUATE is not pass, just assume false
if [ -z "$EVALUATE" ]
then
    EVALUATE='false'
fi

# If SEED is not pass, just assume 1
if [ -z "$SEED" ]
then
    SEED=1
fi

# If MACHINE is not pass, just assume ginsburg
if [ -z "$MACHINE" ]
then
    MACHINE='ginsburg'
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
PATH_TO_PRETRAINED_CKPT=`echo "$PATH_TO_PRETRAINED_CKPT" | xargs`
PATH_TO_FINETUNE_CONFIG_DIR=`echo "$PATH_TO_FINETUNE_CONFIG_DIR" | xargs`
EVALUATE=`echo "$EVALUATE" | xargs`
SEED=`echo "$SEED" | xargs`
MACHINE=`echo "$MACHINE" | xargs`

# Output dir is based on pretrained model name and finetune configs (plus seed)
PRETRAINED_MODEL_NAME=`echo "${PATH_TO_PRETRAINED_CKPT#*"/nlp_bandit_experiments/"}" | sed 's/\//_/g' | sed 's/\.pt//g'`

#IF Fine-tuning
if [ ! -z "$PATH_TO_FINETUNE_CONFIG_DIR" ]
then
    FINETUNE_CONFIG_DIR=${PATH_TO_FINETUNE_CONFIG_DIR#*"fairseq_hydra/$MACHINE/"}
    FINETUNE_CONFIG_DIR=${FINETUNE_CONFIG_DIR%".yaml"}
    FINETUNE_CONFIG_NAME=`basename "$FINETUNE_CONFIG_DIR"`
fi

# Output dir
SCRIPT_NAME=`basename "$0"`
SCRIPT_NAME=${SCRIPT_NAME%".sh"}
OUTPUT_DIR="$NLP_BANDIT_DIR/nlp_bandit_experiments/$SCRIPT_NAME/$PRETRAINED_MODEL_NAME/$FINETUNE_CONFIG_NAME/seed_$SEED"
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

#IF Fine-tuning: prepare all finetune configs
if [ ! -z "$PATH_TO_FINETUNE_CONFIG_DIR" ]
then
    for PATH_TO_FINETUNE_CONFIG_FILE in "$PATH_TO_FINETUNE_CONFIG_DIR"/*
    do
        FINETUNE_CONFIG=${PATH_TO_FINETUNE_CONFIG_FILE#*"fairseq_hydra/$MACHINE/"}
        FINETUNE_CONFIG=${FINETUNE_CONFIG%".yaml"}
        FINETUNE_OUTPUT_DIR="$OUTPUT_DIR/$FINETUNE_CONFIG"
        FINETUNE_CONFIG_FILENAME=`basename "$PATH_TO_FINETUNE_CONFIG_FILE"`
        PATH_TO_FINETUNE_CONFIG="$FINETUNE_OUTPUT_DIR/$FINETUNE_CONFIG_FILENAME"
        # Finetune specific output dir 
        mkdir -p $FINETUNE_OUTPUT_DIR
        
        # Replicate initial config within fine-tuning output dir
        cp $PATH_TO_FINETUNE_CONFIG_FILE $PATH_TO_FINETUNE_CONFIG
    done
fi

#################### START SCRIPT ########################
# Executed script info
echo "Starting script $0 at $(date) with"
echo "      -path_to_pretrained_ckpt $PATH_TO_PRETRAINED_CKPT"
echo "      -path_to_finetune_config_dir $PATH_TO_FINETUNE_CONFIG_DIR"
echo "      -evaluate $EVALUATE"
echo "      -seed $SEED"
echo "      -machine $MACHINE"

# Start
echo "Starting Load Fairseq RoBERTa script at $(date)"

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
                checkpoint.restore_file=$PATH_TO_PRETRAINED_CKPT \
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

# Finished
echo "Done Load Fairseq RoBERTa script at $(date)"

# Finished
echo "Done script $0 at $(date)"

# END
