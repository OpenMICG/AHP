# AHP: Adapter-Enhanced Hierarchical Cross-Modal Pre-training for Lightweight Medical Report Generation

## Requirements

To install the dependencies, run:

```bash
conda env create -f environment.yml
conda activate AHP
```

## Datasets

Download the IU X-Ray 、 MIMIC-CXR  and Bladder Pathology datasets and save them in the `./data/` directory.

For `IU X-Ray` download from [here](https://iuhealth.org/find-medical-services/x-rays).

For `MIMIC-CXR` download from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

For `Bladder Pathology` download from [here](https://figshare.com/projects/nmi-wsi-diagnosis/61973).

## Pretrain Datasets

we used the ROCO、PMC-OA and 

For `ROCO` download from [here](https://github.com/razorx89/roco-dataset).

For `PMC-OA` download from [here](https://huggingface.co/datasets/axiong/pmc_oa).

For `MedICaT` download from [here](https://github.com/allenai/medicat).

## Clinical Metrics and Lexical Metrics

We evaluate RadCliQ-v1, SembScore, RadGraph, and BERTScore using [this](https://github.com/rajpurkarlab/CXR-Report-Metric/tree/main).

We evaluate RaTEScore using [this](https://github.com/MAGIC-AI4Med/RaTEScore).

## Running

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

Run `bash run_bladder.sh` to train a model on the Bladder Pathology data.

# Acknowledge

Our implementations are partially inspired by [BLIP](https://github.com/salesforce/BLIP?tab=readme-ov-file) ,  [ViT-Adapter](https://github.com/czczup/ViT-Adapter) and [R2Gen](https://github.com/zhjohnchan/R2Gen).

Thanks for their great works!
