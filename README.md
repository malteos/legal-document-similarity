# Legal Document Similarity

Implementation, trained models and result data for the paper *Evaluating Document Representations for Content-based Legal Literature Recommendations* ([PDF on Arxiv](#TODO)). 
The supplemental material is available for download under [GitHub Releases](https://github.com/malteos/legal-document-similarity/releases).

The qualitative analysis is available as PDF in `/appendix`.

## Requirements

- Python 3.7
- CUDA GPU (for Transformers)
- Case Law Access Project API key (only for dataset construction)
- Legal-BERT must be downloaded from [here](https://archive.data.jhu.edu/dataset.xhtml?persistentId=doi:10.7281/T1/N1X6I4).

## Installation

Create a new virtual environment for Python 3.7 with Conda:
 
 ```bash
conda create -n paper python=3.7
conda activate paper
```

Clone repository and install dependencies:
```bash
git clone https://github.com/malteos/legal-document-similarity.git repo
cd repo
pip install -r requirements.txt
```

## Experiments

To reproduce our experiments, follow these steps:


### Download datasets

We construct two silver standards from [Open Case Book](https://opencasebook.org/) 
and [WikiSource](https://en.wikisource.org/wiki/Category:United_States_Supreme_Court_decisions_by_topic).
The underlying full-text and citation data is taken from 
the [Case Law Access Project](https://case.law/) 
and [CourtListener](https://courtlisten.com/).
Scripts for data preprocessing are in `./datasets`.

```bash
mkdir -p ./data/ocb ./data/wikisource

# Open Case Book
wget https://github.com/malteos/legal-document-similarity/releases/download/1.0/ocb.tar.gz
tar -xvzf ocb.tar.gz -C ./data/ocb

# WikiSource
wget https://github.com/malteos/legal-document-similarity/releases/download/1.0/wikisource.tar.gz
tar -xvzf wikisource.tar.gz -C ./data/wikisource
```

### Prepare word vectors

With the following commands, fastText and GloVe vectors can be trained or downloaded.

```bash
# Extract plain-text corpora
python cli.py extract_text --data_dir=./data

# fastText vectors
python cli.py train_fasttext --data_dir=./data

# GloVe vectors
sh ./sbin/compute_glove.sh

# Download pretrained word vectors
wget -O ./data/ https://github.com/malteos/legal-document-similarity/releases/download/1.0/ocb_and_wikisource.fasttext.w2v.txt.gz
wget -O ./data/ https://github.com/malteos/legal-document-similarity/releases/download/1.0/ocb_and_wikisource.glove.w2v.txt.gz
```

### Generate Document Representations

The following commands create or download vectors for all document in the two datasets. 

```bash
# Generate (using GPU 0)
python cli.py compute_doc_vecs wikisource --override=1 --gpu 0 --data_dir=./data
python cli.py compute_doc_vecs ocb --override=1 --gpu 0 --data_dir=./data

# Download pretrained document vectors
wget https://github.com/malteos/legal-document-similarity/releases/download/1.0/ocb_models.tar.gz
wget https://github.com/malteos/legal-document-similarity/releases/download/1.0/wikisource_models.tar.gz
tar -xvzf ocb_models.tar.gz
tar -xvzf wikisource_models.tar.gz
```

### Evaluation

After generating the document representations for Open Case Book and WikiSource, 
the results can be computed and viewed with a Jupyter notebook. 
Figures and tables from the paper are part of the notebook.

```bash
jupyter notebook evaluation.ipynb
```

Due to the space constraints some results could not be included in the paper.
The full results for all methods are available as 
[CSV file](https://github.com/malteos/legal-document-similarity/releases/download/1.0/results.tar.gz)
(or via the Jupyter notebook).

## How to cite

If you are using our code, please cite [our paper](#TODO):

```bibtex
@InProceedings{Ostendorff2021,

}
```

## License

MIT
