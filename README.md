# PBDAN
Demo code release for Progressive Bridge Domain Adaptation for Open-set Fault Diagnosis: A KnowledgeCredibility-Guided Screening and Selective Alignment Paradigm

# Progressive Bridge Domain Adaptation Network (PBDAN)

## Overview
Open-set fault diagnosis transfer aims to reuse diagnostic knowledge of known faults from a controlled source domain to facilitate recognition in an uncontrolled target domain, while reliably flagging unknown faults outside the current knowledge boundary. This work proposes PBDAN to address severe negative transfer under substantial domain discrepancies by introducing:

- **Dual-dimension knowability metric**: Integrating Wasserstein distance with energy score for complementary discrimination
- **Three-stage progressive transfer**: Bridge domain construction, knowability screening, and selective attraction
- **Evidential optimization**: Dirichlet Process Gaussian Mixture Model for robust knowability modeling

## Dataset
#### The Paderborn Bearing Dataset, Available: https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download
#### The PHM 2009 Gearbox Dataset, Available: https://phmsociety.org/data-analysis-competition

## Requirements

- python 3.12
- torch 2.5.1+cu121
- torchvision 0.20.1+cu121
- Tensorflow  2.20.0
- Tensorlayer 2.2.5
- Tensorboard 2.20.0


## Quick Start

- Step 1: Run the 'Build_bearingbridge_task.py' or  'Building_gearboxbridge_task.py' to preprocessing the dataset
- Step 2: Run the 'main_bearing.py' or 'main_gearbox.py' to illustrate the proposed method


## Contact
- syyang@zust.edu.cn
