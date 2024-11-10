# Towards a Unified Framework for AI Medical Report Generation and Medical Visual Question Answering with Protective Guardrails

Code release for ECE 570 Term Project

## Overview of this repository
This repository contains all the code utilized in this project. The code is divided into 3 sections, corresponding to the 3 sections of the Methodology in the paper:
* Stage 1: AI Medical Report Generation
* Stage 2a: Small Language Model Guardrails
* Stage 2b: Medical Visual Question Answering with a Vision Language Model

The following sections of this README describe how to run the code for each Stage. Due to computational constraints and ease of implementation, the code for the three stages is split over Google Colab and a local Python environment. Please follow the instructions carefully as some steps require running code locally, while others require creating a Google Colab Notebook due to the availability of high-end GPUs.

## Stage 1: AI Medical Report Generation
Below are the instructions for setup/installation and achieving the three goals comprising AI Medical Report Generation.

### Environment Setup and Dependency Installation
Please note that these setup/installation instructions follow the same steps as those in the original [PromptMRG](https://github.com/jhb86253817/PromptMRG) framework. If you encounter any issues, please refer to the PromptMRG repository for clarification.
1. Clone this repository.
```Shell
git clone https://github.com/REPONAME.git
```
2. Create a new conda environment.
```Shell
conda create -n promptmrg python=3.10
conda activate promptmrg
```
3. Install the dependencies found in REPONAME/requirements.txt.
```Shell
pip install -r requirements.txt
```

### Acquiring Datasets
The datasets and corresponding annotation files are quite large, so they are not located in this repository by default. Please follow the instructions listed below to retrieve them. To avoid unnecessary confusion and errors, these instructions are directly copied from the original [PromptMRG](https://github.com/jhb86253817/PromptMRG) repository.

* **MIMIC-CXR**: The images can be downloaded from either [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/) or [R2Gen](https://github.com/zhjohnchan/R2Gen). The annotation file can be downloaded from the [Google Drive](https://drive.google.com/file/d/1qR7EJkiBdHPrskfikz2adL-p9BjMRXup/view?usp=sharing). Additionally, you need to download `clip_text_features.json` from [here](https://drive.google.com/file/d/1Zyq-84VOzc-TOZBzlhMyXLwHjDNTaN9A/view?usp=sharing), the extracted text features of the training database via MIMIC pretrained [CLIP](https://stanfordmedicine.app.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml). Put all these under folder `data/mimic_cxr/`.
* **IU-Xray**: The images can be downloaded from [R2Gen](https://github.com/zhjohnchan/R2Gen) and the annotation file can be downloaded from the [Google Drive](https://drive.google.com/file/d/1zV5wgi5QsIp6OuC1U95xvOmeAAlBGkRS/view?usp=sharing). Put both images and annotation under folder `data/iu_xray/`.

Moreover, you need to download the `chexbert.pth` from [here](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) for evaluating clinical efficacy and put it under `checkpoints/stanford/chexbert/`.

You will have the following structure:
````
PromptMRG
|--data
   |--mimic_cxr
      |--base_probs.json
      |--clip_text_features.json
      |--mimic_annotation_promptmrg.json
      |--images
         |--p10
         |--p11
         ...
   |--iu_xray
      |--iu_annotation_promptmrg.json
      |--images
         |--CXR1000_IM-0003
         |--CXR1001_IM-0004
         ...
|--checkpoints
   |--stanford
      |--chexbert
         |--chexbert.pth
...
````


### Goal 1: Training a model to generate medical reports given Chest X-ray images using the [PromptMRG](https://github.com/jhb86253817/PromptMRG) framework.


## Training
* To train a model by yourself, run `bash train_mimic_cxr.sh` to train a model on MIMIC-CXR.
* Alternatively, you can download a trained model weight from [here](https://drive.google.com/file/d/1s4AoLnnGOysOQkdILhhFCL59LyQtRHGa/view?usp=drive_link). Note that this model weight was trained with images from [R2Gen](https://github.com/zhjohnchan/R2Gen). If you use images processed by yourself, you may obtain degraded performance with this weight. In this case, you need to train a model by yourself.
## Testing
Run `bash test_mimic_cxr.sh` to test a trained model on MIMIC-CXR and `bash test_iu_xray.sh` for IU-Xray.

## Acknowledgment
* [R2Gen](https://github.com/zhjohnchan/R2Gen)
* [BLIP](https://github.com/salesforce/BLIP)
* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2)
