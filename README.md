# RPG-MoGe: Source Code Documentation

## Overview
This repository contains the implementation of RPG-MoGe, a novel approach for speech relation extraction. The following guide provides step-by-step instructions for setting up and running the system.


## Datasets

This repository contains and references several datasets for Speech Relation Extraction (SpeechRE).

### 1. TTS-based SpeechRE Datasets (provided in this repo)

Under the [`Datasets/`](./Datasets) folder, we provide the **text parts** of two TTS-synthesized SpeechRE datasets:

* `speech_conll04`: TTS version of the widely used CoNLL04 dataset.
* `speech_ReTracred`: TTS version of the Retacred dataset.

These datasets contain the **textual annotations** aligned with synthetic speech (not included here).

---

### 2. CommonVoice-SpeechRE (proposed in this work)

To advance SpeechRE with **real human speech**, we propose the **CommonVoice-SpeechRE** dataset, derived from [Common Voice 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0).

We release it in two parts on Hugging Face:

* **[CommonVoice-SpeechRE-audio](https://huggingface.co/datasets/kobe8-24/CommonVoice-SpeechRE-audio)**

  * Contains **19,583 real speech samples**, downsampled to 16kHz.
  * Audio files are named with unique speech IDs.

* **[CommonVoice-SpeechRE-text](https://huggingface.co/datasets/kobe8-24/CommonVoice-SpeechRE-text)**

  * Provides transcripts, entity annotations, and relation triplets aligned with audio IDs.
  * Entity and relation annotations are **manually labeled** by our team.

When using these datasets, please cite both **Common Voice** and our **CommonVoice-SpeechRE** work.

## Installation and Setup

### 1. Environment Configuration
Install all required dependencies by executing:
```bash
pip install -r requirements.txt
```

### 2. Model Download
Download the Whisper model by running:
```bash
python download_hfmodel.py
```

## Training Procedures
Execute the following scripts to train the models on different datasets:

1. CONLL04 Dataset:
```bash
python train_conll04.py
```

2. CV17 Dataset:
```bash
python train_cv17.py
```

3. Retraced Dataset:
```bash
python train_retraced.py
```

## Evaluation
To evaluate the model performance on test sets, run the corresponding evaluation scripts:

1. CONLL04 Evaluation:
```bash
python test_conll04.py
```

2. Retacred Evaluation:
```bash
python test_retacred.py
```

3. CV17 Evaluation:
```bash
python test_cv17.py
```

