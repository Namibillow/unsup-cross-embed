#!/bin/bash

# OUTPATH=low_resource/en
# PREFIX=en_50K_bpe
# DATA=en_50K_processed.txt

OUTPATH=data/processed_data/${1}  # path where processed files will be stored
PREFIX=${2} # Output file prefix
DATA=$OUTPATH/${3} # path to the text
FASTBPE=$HOME/fastBPE/fast  # path to the fastBPE tool
NUMV=16000

# create output dir if not exist
mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
$FASTBPE learnbpe $NUMV $DATA > "$OUTPATH/${PREFIX}_codes"

$FASTBPE applybpe "$OUTPATH/${PREFIX}_${NUMV}.txt" $DATA "$OUTPATH/${PREFIX}_codes"

# get train vocabulary 
cat "$OUTPATH/${PREFIX}_${NUMV}.txt" | $FASTBPE getvocab - > "${OUTPATH}/${PREFIX}_${NUMV}.vocab"

# truncate the frequency info
cat "${OUTPATH}/${PREFIX}_${NUMV}.vocab" | cut -f1 -d ' ' | sponge "${OUTPATH}/${PREFIX}_${NUMV}.vocab"