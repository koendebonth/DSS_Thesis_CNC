# CNC Machining Code Repository

**Note:** The research documentation will be added to this repository upon completion.

## Overview of the Code
This repository contains the implementation of a transfer learning framework for predictive maintenance in CNC machining processes. The code focuses on:

- **Data Preprocessing:** Reading and preparing sensor data (vibration data) from .h5 files, including filtering, normalization, and segmentation.  
  *Note: The data is not stored in this repository. Instead, the code uses the data folder from the [CNC_Machining repository](https://github.com/boschresearch/CNC_Machining/tree/main/data).*
- **Feature Engineering:** Applying the Wavelet Packet Transform to extract time and frequency domain features from the data.
- **Model Development:** Implementing baseline machine learning models such as Random Forest and Light Gradient Boosting.
- **Transfer Learning Experiments:** Setting up experiments to apply models trained on one machine to data from other machines.
- **Evaluation:** Calculating classification metrics such as Accuracy, Precision, Recall, and F1 Score to assess model performance.

## Repository Structure
- **/src:** All source code for preprocessing, feature engineering, and model implementation.
- **/experiments:** Experimental setups for transfer learning and model evaluation.
- **README.md:** This documentation file, providing an overview of the code and repository structure.

*The code utilizes the data folder from the [CNC_Machining repository](https://github.com/boschresearch/CNC_Machining), which is not part of this repository.*



## License and Attribution

This project incorporates data/code from the [CNC_Machining repository](https://github.com/boschresearch/CNC_Machining) by Bosch Research, which is licensed under the BSD 3-Clause License.

We have ensured compliance with the following requirements:
- The original copyright notice,
- The list of conditions, and
- The disclaimer have been retained in our source code and binary distributions as appropriate.

For full details, please refer to the LICENSE file included in this repository.
