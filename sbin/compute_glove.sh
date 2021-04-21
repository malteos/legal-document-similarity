#!/usr/bin/env bash

export CORPUS=./data/ocb_and_wikisource.w2v_tokens.txt
export VOCAB_FILE=ocb_and_wikisource.vocab.txt
export COOCCURRENCE_FILE=ocb_and_wikisource.cooccurrence.bin
export COOCCURRENCE_SHUF_FILE=ocb_and_wikisource.cooccurrence.shuf.bin
export BUILDDIR=/Volumes/data/repo/glove/build
export SAVE_FILE=ocb_and_wikisource
export VERBOSE=2
export MEMORY=4.0
export VOCAB_MIN_COUNT=2
export VECTOR_SIZE=300
export MAX_ITER=15
export WINDOW_SIZE=15
export BINARY=2
export NUM_THREADS=8
export X_MAX=10

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE \
    -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE \
    -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

echo -n "$(wc -l < $SAVE_FILE.txt | xargs) $VECTOR_SIZE\n" > $SAVE_FILE.w2v.txt
cat $SAVE_FILE.txt >> $SAVE_FILE.w2v.txt
