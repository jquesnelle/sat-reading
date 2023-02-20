This repository contains the code used to produce the models and data in the blog post [Language Models vs. The SAT Reading Test](https://jeffq.com/blog/language-models-vs-the-sat-reading-test).

Dataset: [emozilla/sat-reading](https://huggingface.co/datasets/emozilla/sat-reading)
Models: [XXL (11B)](https://huggingface.co/emozilla/flan-t5-xxl-sat-reading) [XL (3B)](https://huggingface.co/emozilla/flan-t5-xl-sat-reading) [Large (780M)](https://huggingface.co/emozilla/flan-t5-large-sat-reading) [Base (350M)](https://huggingface.co/emozilla/flan-t5-base-sat-reading)

| File | Description |
| ---- | ----------- |
| [combine-raw-data.py](combine-raw-data.py) | Combine data in `raw-data` folder into a single JSON |
| [create-dataset.py](create-dataset.py) | Create [datasets](https://github.com/huggingface/datasets)-compatible datasets from combined JSON |
| [process-dataset-for-training.py](process-dataset-for-training.py) | Create a tokenized version of an existing dataset for training |
| [prompt-loop.py](promot-loop.py) | Playground for loading and prompting models |
| [take-tests.py](take-tests.py) | Evaluate models against a dataset |
| [train.py](train.py) | Finetune a FLAN-T5 model |

To check the generalization of finetuned models, install [lm-evaluation-harness](https://github.com/bigscience-workshop/lm-evaluation-harness) and run it on the `SuperGLUE` metrics: `cb`, `copa`, `superglue_rte`, `wic`, and `wsc` (and any other metrics you'd like, of course).