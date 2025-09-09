# RPG-MoGe: Source Code Documentation

## Overview
This repository contains the implementation of RPG-MoGe, a novel approach for speech relation extraction. The following guide provides step-by-step instructions for setting up and running the system.

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

## Dataset Information
During the current submission and review phase, our annotated CommonVoice-SpeechRE dataset is limited to 500 sample instances. This subset is provided for preliminary evaluation and demonstration purposes.
