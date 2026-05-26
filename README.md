# mSpecFusion-Net

Official repository for the MICCAI 2026 paper:

> **mSpecFusion-Net: Intelligent Multimodal Smartphone Imaging for Mobile Differential Diagnosis of Psoriasis and Dermatitis and Deep Learning**
>
> Sewoong Kim*, Seong-Hyun Kim*, Thiago Coutinho Cavalcanti*, Jonghun Lee, Moon Hwan Lee, Donghun Lee†, Jae Youn Hwang†
>
> *Equal contribution. †Corresponding authors.
>
> *Accepted at the 29th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2026), Strasbourg, France.*

---

## Overview

mSpecFusion-Net is a modality-aware iterative-fusion Transformer for smartphone-based differential diagnosis of psoriasis and seborrheic dermatitis. The model integrates three complementary optical contrasts acquired with a portable smartphone-based multimodal dermascope:

- **Co-polarized multispectral imaging (Co-P MSI)** — 8-band reflectance (435–640 nm)
- **Cross-polarized multispectral imaging (Cross-P MSI)** — 8-band reflectance with surface specular suppression
- **Ultraviolet-excited autofluorescence (UV-AF)** — biochemical-sensitive contrast

A PCA-based discrete band selection step compresses each multispectral stack to its most informative bands prior to fusion, and a parallel-branch Transformer with iterative cross-modal aggregation progressively refines complementary evidence across modalities into a unified diagnostic representation.

Under patient-wise 5-fold cross-validation on a clinical cohort of 32 patients (18 psoriasis, 14 seborrheic dermatitis), mSpecFusion-Net achieves:

| Metric | Value (mean ± SD) |
|---|---|
| F1 | 0.9057 ± 0.0612 |
| Accuracy | 0.9068 ± 0.0576 |
| ROC-AUC (micro) | 0.9461 ± 0.0540 |
| PRC-AUC (micro) | 0.9358 ± 0.0666 |

with 4.18 M trainable parameters (16.0 MB, fp32) and a measured GPU inference latency of 17.3 ms/image at batch size 1.

---

## Release Status

The full implementation code will be released in this repository **in coordination with completion of pending intellectual property filings**, expected ahead of the conference (September 2026). This repository will be updated with:

- Model architecture (`models/`)
- Training and evaluation scripts
- Pre-trained weights for the patient-wise 5-fold runs
- Inference / deployment example
- Multimodal data loader and PCA band-selection utility

For inquiries, early access requests, or collaboration, please contact the corresponding author at **jyhwang@dgist.ac.kr**.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{kim2026mspecfusion,
  title     = {mSpecFusion-Net: Intelligent Multimodal Smartphone Imaging
               for Mobile Differential Diagnosis of Psoriasis and
               Dermatitis and Deep Learning},
  author    = {Kim, Sewoong and  Kim, Seong-Hyun and Cavalcanti, Thiago Coutinho and Lee, Jonghun and
               Lee, Moon Hwan and Lee, Donghun and Hwang, Jae Youn},
  booktitle = {Medical Image Computing and Computer Assisted
               Intervention -- MICCAI 2026},
  year      = {2026},
  publisher = {Springer Nature Switzerland}
}
```

Please update this entry once the official LNCS volume and page numbers are assigned.

---

## Ethics & Data

The clinical study was approved by the Institutional Review Board of Seoul National University Hospital (IRB No. 1908-161-1059), and informed consent was obtained from all participants. **Clinical patient data will not be released** to protect participant privacy, in accordance with the approved protocol.

---

## License

The code released in this repository will be distributed under a license to be specified at the time of release. Please refer back to this page or contact the corresponding author for licensing inquiries.

---

## Contact

- **Jae Youn Hwang** (corresponding author) — `jyhwang@dgist.ac.kr`
- **Donghun Lee** (corresponding author, clinical) — `ivymed27@snu.ac.kr`
- **Seong-Hyun Kim** — `rlatjdgus224@dgist.ac.kr`

Affiliations: Department of Electrical Engineering and Computer Science / Department of Artificial Intelligence / Department of Biomedical Science and Engineering, Daegu Gyeongbuk Institute of Science and Technology (DGIST), Daegu, Republic of Korea; RNSLab Co., Ltd., Incheon, Republic of Korea; Max Planck Institute for Medical Research, Heidelberg, Germany; Department of Dermatology, Seoul National University Hospital, Seoul, Republic of Korea.
