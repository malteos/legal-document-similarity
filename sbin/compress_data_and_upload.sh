#!/usr/bin/env bash

# Compress data for upload

# Word vectors
gzip ./data/ocb_and_wikisource.fasttext.w2v.txt
gzip ./data/ocb_and_wikisource.glove.w2v.txt

# Datasets
cd ./data/wikisource
tar -cvzf wikisource.tar.gz docs_without_text.json doc_id2idx.json idx2doc_id.json texts.json.gz gold.csv meta.csv

cd ./data/ocb
tar -cvzf ocb.tar.gz cits.json doc_id2idx.json idx2doc_id.json texts.json.gz gold.csv meta.csv

# Models
tar -cvzf wikisource_models.tar.gz ./models/wikisource/*
tar -cvzf ocb_models.tar.gz ./models/ocb/*

# Results
tar -cvzf results.tar.gz ./data/results_ocb.csv ./data/results_wikisource.csv


### Upload to GitHub release (with https://github.com/github-release/github-release)
export GITHUB_TOKEN=
export GITHUB_USER=
export GITHUB_REPO=

~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name wikisource.tar.gz --file wikisource.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name ocb.tar.gz --file ocb.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name wikisource_models.tar.gz --file wikisource_models.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name ocb_models.tar.gz --file ocb_models.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name results.tar.gz  --file results.tar.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name ocb_and_wikisource.fasttext.w2v.txt.gz  --file ocb_and_wikisource.fasttext.w2v.txt.gz
~/github-release upload --user $GITHUB_USER --repo $GITHUB_REPO --tag 1.0 --name ocb_and_wikisource.glove.w2v.txt.gz  --file ocb_and_wikisource.glove.w2v.txt.gz

