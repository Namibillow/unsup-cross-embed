#!/bin/bash

# OUTPATH=low_resource/jaen

# PREFIX=en_50K_bpe
# or
# PREFIX=ja_50K_en_50K_bpe

# SRC_DATA=en/en_50K_processed.txt
# TGT_DATA=ja/ja_50K_processed.txt

OUTPATH="data/processed_data/${1}"  # path where processed files will be stored
echo "Writing to ${OUTPATH}. To change this, set the OUTPUT_DIR environment variable."

PREFIX=${OUTPATH}/${2} # Output file prefix

SRC_DATA=${3} # relative path to the text

if [ ! -z "$4" ]
  then
    echo "Will train BPE jointly"
    TGT_DATA=${4}
fi

FASTBPE=$HOME/fastBPE/fast  # path to the fastBPE tool
NUMV=32000 # number of merges

# create output dir if not exist
mkdir -p $OUTPATH

if [ -z $TGT_DATA ]
then # Separate training 

    # learn bpe codes on the training set (or only use a subset of it)
    $FASTBPE learnbpe $NUMV ${SRC_DATA} > "${PREFIX}_codes"

    $FASTBPE applybpe "${PREFIX}_${NUMV}_processed.txt" ${SRC_DATA} "${PREFIX}_codes"

    # get train vocabulary 
    cat "${PREFIX}_${NUMV}_processed.txt" | $FASTBPE getvocab - > "${PREFIX}_${NUMV}.vocab_dict"

    NUMOFLINES=$(wc -l < "${PREFIX}_${NUMV}.vocab_dict")

else # joint training 

    $FASTBPE learnbpe $NUMV ${SRC_DATA} ${TGT_DATA} > "${PREFIX}_codes"
    
    # Apply learn bpe to both text 
    $FASTBPE applybpe "${PREFIX}_${NUMV}_src_processed.txt" ${SRC_DATA} "${PREFIX}_codes"
    $FASTBPE applybpe "${PREFIX}_${NUMV}_tgt_processed.txt" ${TGT_DATA} "${PREFIX}_codes"

    cat "${PREFIX}_${NUMV}_src_processed.txt" "${PREFIX}_${NUMV}_tgt_processed.txt" | $FASTBPE getvocab - > "${PREFIX}_${NUMV}.vocab_dict"
    NUMOFLINES=$(wc -l < "${PREFIX}_${NUMV}.vocab_dict")
fi 

# truncate the frequency info
cat "${PREFIX}_${NUMV}.vocab_dict" | cut -f1 -d ' ' | sponge "${PREFIX}_${NUMV}.vocab_dict"

echo "Language: ${1}
original file path: ${3}
Number of merges performed: ${NUMV}
Total vocaburaly: ${NUMOFLINES}
" >> "${PREFIX}_${NUMV}_info.txt"

echo "done"
