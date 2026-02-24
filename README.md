# Agent-Augmented Sequence Modeling for Explainable ICU Diagnosis Prediction

This project investigates whether a multi-agent LLM reasoning layer can improve early ICU diagnosis prediction from electronic health record time-series data.

We first train a baseline LSTM model on the MIMIC-IV dataset to predict primary ICD-10 diagnoses from the first 12 hours of ICU admission. At inference time, a team of specialized LLM agents evaluates the model’s top candidate diagnoses by generating clinical rationales, and a ranking agent selects the final prediction without retraining the underlying model.

The goal is to enhance diagnostic accuracy and interpretability by combining deep learning with structured AI reasoning, providing a scalable way to augment existing clinical prediction systems.
