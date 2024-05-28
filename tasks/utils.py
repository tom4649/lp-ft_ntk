from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks

GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
OOD_DATASETS = ["amazon", "dynasent", "semeval", "sst5"]
PUBMED_DATASETS = ["pubmed"]


TASKS = ["glue", "superglue", "ood", "pubmed"]

DATASETS = GLUE_DATASETS + SUPERGLUE_DATASETS + OOD_DATASETS + PUBMED_DATASETS

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
}
