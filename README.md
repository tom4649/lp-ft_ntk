# Understanding Linear Probing then Fine-tuning Language Models from NTK Perspective

This repository contains the code for our paper:

> [Understanding Linear Probing then Fine-tuning Language Models from NTK Perspective](https://arxiv.org/abs/2405.16747). Akiyoshi Tomihari and Issei Sato.

## Dependencies
The main dependencies are:
```plaintext
Python 3.9 or higher
adapter-transformers = 2.2.0
torch = 1.12.1
```

Please refer to the `pyproject.toml` file for more details.

## Setup
To set up and run the project, follow these steps:
```bash
# Install torch and torchvision
pushd wheels
bash download.sh
popd

# Configure the project to create virtual environments within the project directory
poetry config virtualenvs.in-project true

# Set the local python version using pyenv
pyenv local 3.9.18

# Install dependencies and activate the virtual environment
poetry install
poetry shell
```
## Data files
The data files for GLUE and SuperGLUE will be automatically downloaded.

To conduct the experiments of OOD and PubMed, you need to download the following data files:
#### OOD Datasets
- **Amazon**:
  - `train.tsv`
  - `test.tsv`
  - Location: `tasks/OOD_NLP/datasets/process/SentimentAnalysis/amazon`
- **Dynasent, SemEval, SST-5**:
  - `test.tsv` for each dataset
  - Locations:
    - `tasks/OOD_NLP/datasets/process/SentimentAnalysis/dynasent`
    - `tasks/OOD_NLP/datasets/process/SentimentAnalysis/semeval`
    - `tasks/OOD_NLP/datasets/process/SentimentAnalysis/sst5`

#### PubMed 20k Dataset
- Files:
  - `train.txt`
  - `dev.txt`
  - `test.txt`
- Location: `tasks/pubmed/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign`

Please refer to the data sources and our codes for the details.

## Acknowledgments
We use the following resources and libraries:
- Base code structure: [PETuning](https://github.com/guanzhchen/PETuning)

- Computing the NTK matrix and linear probing: [LM-Kernel-FT](https://github.com/princeton-nlp/LM-Kernel-FT)

- LoRA method: [LoRA](https://github.com/microsoft/LoRA)

- Libraries for implementation: [Hugging Face Transformers](https://github.com/huggingface/transformers), [Adapter-Hub](https://github.com/Adapter-Hub/adapter-transformers)

- Datasets: [OOD_NLP](https://github.com/lifan-yuan/OOD_NLP), [PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)

## Citation
```bibtex

@misc{tomihari2024understanding,
      title={Understanding Linear Probing then Fine-tuning Language Models from NTK Perspective},
      author={Akiyoshi Tomihari and Issei Sato},
      year={2024},
      eprint={2405.16747},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
