# Defect Classification Using Deep Learning

This repository contains an end-to-end pipeline for classifying defects in products using deep learning. The project covers image preprocessing, CNN model training, and deployment of a simple web application for inference.

## Overview

The system takes product images as input and predicts the corresponding defect class. It is designed to be modular: you can swap datasets (e.g., MNIST vs. real industrial defect data) and model architectures (e.g., ResNet, MobileNet, or a custom CNN) with minimal code changes.

## Features

- Image preprocessing: resizing, normalization, and train/validation/test splitting.  
- CNN classifier: configurable backbone (ResNet, MobileNet, or custom architecture).  
- Model persistence: trained weights saved for reuse in inference and deployment.  
- Web interface: simple Flask/Django app for uploading images and getting predictions.

## Project Structure

- `data/`: raw and processed datasets.  
- `notebooks/`: exploratory analysis and prototyping.  
- `src/`: core code (preprocessing, training, models, inference, utilities).  
- `webapp/`: web application (Flask or Django), templates, and static assets.  
- `saved_models/`: exported trained models.  
- `requirements.txt`: Python dependencies.

## Setup

1. Clone the repository and move into the project directory.  
2. (Optional) Create and activate a virtual environment.  
3. Install dependencies with `pip install -r requirements.txt`.  

Adjust `requirements.txt` to match your stack (e.g., PyTorch or TensorFlow plus Flask or Django).

## Data Preparation

1. Download the chosen dataset (e.g., MNIST or an industrial surface defect dataset).  
2. Organize it into `train`, `val`, and `test` splits, grouped by class.  
3. Run the preprocessing script (if present) to resize and normalize images and to persist the processed dataset.

## Training

Use the training script to fit a CNN on the chosen dataset. Typical arguments include dataset name, model backbone, number of epochs, batch size, and learning rate. The script should report training and validation metrics and save the best-performing model into `saved_models/`.

Document your final configuration and results in the repository, including:
- Dataset used and image size.  
- Model architecture and optimizer.  
- Final validation and test accuracy (and any other relevant metrics).

## Inference and Web Demo

An inference script is provided to run predictions on individual images from the command line using a saved model. The web application wraps the same inference logic behind a simple upload form: users submit an image, the backend preprocesses it, runs the model, and returns the predicted defect class.

Start the web app (Flask: `python app.py`; Django: `python manage.py runserver`) and access it in a browser at the provided URL.

## Extensibility

The codebase is structured to make it straightforward to:
- Swap datasets or add new defect classes.  
- Replace or extend the CNN backbone.  
- Integrate additional deployment options (e.g., Docker, cloud hosting, or API gateways).
