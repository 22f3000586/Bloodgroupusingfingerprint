# BloodgroupUsingFingerprint

## Project Overview
This project aims to determine a person’s blood group using their fingerprint data (or biometric fingerprint features) by training a machine-learning model and deploying a simple application.
The repository includes:
- A dataset in `dataset/dataset_blood_group`
- A Jupyter notebook (`model.ipynb`) demonstrating data exploration and model training
- A Python app (`app.py`) which uses the trained model for inference
- A `requirements.txt` and `runtime.txt` for dependencies and runtime environment

## Features
- Pre-processing of fingerprint dataset (or features extracted from fingerprint images)
- Model training using Python and Jupyter notebook
- A deployable Python script (`app.py`) to predict blood group given fingerprint input
- Easily replicable environment via `requirements.txt` and `runtime.txt`

## Technology Stack
- Python (for data analysis, ML model, and application)
- Jupyter Notebook (for prototyping)
- Standard ML libraries: pandas, numpy, scikit-learn (see `requirements.txt`)
- Lightweight app/script for inference

## Repository Structure
.
├── dataset/
│   └── dataset_blood_group/       ← fingerprint + blood group data
├── .gitignore                     ← files/folders to ignore
├── app.py                         ← main inference application
├── model.ipynb                    ← notebook for training and experimenting
├── requirements.txt               ← list of Python dependencies
├── runtime.txt                    ← runtime environment specification
└── README.txt                     ← this file

## Getting Started
### Prerequisites
- Python (version compatible with requirements)
- Ensure you have the dataset downloaded/unpacked in `dataset/dataset_blood_group/`
- Install required dependencies:
  pip install -r requirements.txt

### Training the Model
1. Open `model.ipynb` in Jupyter Notebook or JupyterLab.
2. Run each cell sequentially:
   - Load data
   - Pre-process features
   - Train different algorithms
   - Evaluate performance
   - Export/save the final model (if built)

## Contributing
Contributions are welcome! You can:
- Add new features (for example, GUI or web frontend)
- Improve model performance (try deep learning or image-based fingerprint processing)
- Add more documentation or tests
Submit an issue or pull request if interested.

## Contact
For any questions, reach out via your GitHub profile or email.
