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
The datasets and corresponding annotation files are quite large, so they are not located in this repository by default. Please follow the instructions listed below to retrieve them. To avoid unnecessary confusion and errors, these instructions are adapted from the original [PromptMRG](https://github.com/jhb86253817/PromptMRG) repository with some slight modifications.

* **MIMIC-CXR**: Do not worry about downloading the images for this dataset because they are not publicly available and require credentialed access. The annotation file can be downloaded from the [Google Drive](https://drive.google.com/file/d/1qR7EJkiBdHPrskfikz2adL-p9BjMRXup/view?usp=sharing). Additionally, you need to download `clip_text_features.json` from [here](https://drive.google.com/file/d/1Zyq-84VOzc-TOZBzlhMyXLwHjDNTaN9A/view?usp=sharing), the extracted text features of the training database via MIMIC pretrained [CLIP](https://stanfordmedicine.app.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml). Put all these under folder `data/mimic_cxr/`.
* **IU-Xray**: This dataset contains [Chest X-ray Image, Medical Report] pairs. The images can be downloaded from [R2Gen](https://github.com/zhjohnchan/R2Gen). The annotations file `iu_annotation_promptmrg.json` has already been uploaded to this repository for your convenience and because it is NOT the same as the original annotations file found in the PromptMRG repository due to some necessary modifications. Make sure the images folder and annotations file are both under folder `data/iu_xray/`.
* **VQA-RAD**: This dataset contains [Chest X-ray Image, Question, Answer] triplets. The images can be downloaded from [OSF](https://osf.io/89kps/files/osfstorage). Please ONLY download the `VQA_RAD Image Folder` as the rest of the files are already in this repository for your convenience. Also, make sure to rename the images folder to `images` and place it under `data/vqa_rad`.
* You also need to download the `chexbert.pth` from [here](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) for evaluating clinical efficacy and put it under `checkpoints/stanford/chexbert/`.

After these steps, please ensure your folder directory has the following structure:
````
GITHUBREPONAME
|--data
   |--user_questions
      |--gpt_queries.json
      |--potential_questions.txt
      |--safe_questions.txt
      |--violations_questions.txt

   |--mimic_cxr
      |--base_probs.json
      |--clip_text_features.json
      |--mimic_annotation_promptmrg.json
      |--clip-imp-pretrained_128_6_after_4.pt

   |--iu_xray
      |--iu_annotation_promptmrg.json
      |--annotation.json
      |--images
         |--CXR1000_IM-0003
         |--CXR1001_IM-0004
         ...
   |--vqa_rad
      |--Readme.docx
      |--vqa_annotations_promptmrg.json
      |--VQA_RAD Dataset Public.json
      |--VQA_RAD Dataset Public.xlsx
      |--VQA_RAD Dataset Public.xml
      |--images
         |--synpic676.jpg
         |--synpic9872.jpg
         ...
|--checkpoints
   |--stanford
      |--chexbert
         |--chexbert.pth
...
````


### Goal 1: Training a model to generate medical reports given Chest X-ray images from the IU-Xray dataset using the PromptMRG framework.
* To train the PromptMRG model on the IU-Xray dataset, run `./train_iu_xray.sh`. However, before doing so, make sure to change the `--save-dir` argument in the bash script to whatever directory you would like to save the model in.

### Goal 2: Use your trained model from Goal 1 or an existing pre-trained model to perform inferencing on the IU-Xray test dataset.
* To use the model you trained in **Goal 1**, run `./test_iu_xray.sh` and modify the `--load-pre-trained` argument in the bash script to the path where you saved the model.
* To use the model we trained for this ECE 570 project, run `./test_iu_xray.sh` without any modifications.

### Goal 3: Use the model we trained for this ECE 570 project to perform inferencing on the VQA-RAD test dataset.
* Run `./test_vqa_rad.sh` without any modifications. The reports and performance metrics will be printed to the terminal, and they are also stored in `results/promptmrg/experiment_results/base_iu_model/test/base_iu_model_vqa_rad_mrg_test_log.json`.
* **NOTE:** The performance metrics are irrelevant when inferencing on the VQA-RAD dataset as the dataset does not come with ground truth labels. The purpose of this step is solely to generate medical reports for knowledge enhancement in Stage 2b. of our proposed framework. As the focus of this project is on Stages 2a and 2b, we do not worry about the quality of generated medical reports using PromptMRG, as we are using the framework out-of-the-box without any modifications. We assume the quality of the generated reports is a constant variable. Therefore, we do not conduct any evaluation on the inferencing metrics that are outputted in this Goal and **Goal 2** above.

## Acknowledgment
* [R2Gen](https://github.com/zhjohnchan/R2Gen)
* [BLIP](https://github.com/salesforce/BLIP)
* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2)
