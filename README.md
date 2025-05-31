# Fast-Speech-Transcription-Grammar-Scoring-Engine
 
This repository showcases a powerful pipeline for transcribing audio files and predicting grammar scores using cutting-edge machine learning techniques. Built with Python, Whisper, and RoBERTa, this project tackles the challenge of evaluating spoken language proficiency from audio data. Let’s dive into the details! 🎉

# 📋 Project Overview
This project processes audio files from the VoiceCraft Self Project dataset, transcribes them using the Whisper model, and predicts grammar proficiency scores using a fine-tuned RoBERTa-large model. The pipeline is optimized for efficiency with batch processing, GPU acceleration, and text chunking to handle long transcriptions. 🛠️
The goal? Transform raw audio into meaningful transcriptions and assign grammar scores (1 to 5) based on linguistic quality. The final output is a submission file ready for evaluation! 📊

# 🛠️ Features

🎙️ Audio Transcription: Leverages OpenAI’s Whisper (medium model) for high-accuracy transcription of audio files.
⚙️ Batch Processing: Splits audio files into batches (20 for training, 10 for testing) for efficient parallel processing on GPUs.
📝 Text Chunking: Manages long transcriptions by splitting them into overlapping chunks (~150 characters) for RoBERTa processing.
🤖 Grammar Scoring: Fine-tunes a RoBERTa-large model for regression to predict grammar scores (1–5) with MSE loss.
📈 Evaluation Metrics: Monitors training/validation MSE and accuracy for robust model performance.
💾 Output Generation: Produces clean CSV files (train_ready.csv, test_ready.csv, submission.csv) for seamless submission.



# Setup & Dependencies 🛠️

Installs required libraries like openai-whisper, transformers, and pandas.
Configures Kaggle data sources for audio and metadata.


# Audio Transcription 🎙️

Uses Whisper to transcribe .wav files from audios_train and audios_test folders.
Processes files in batches using multiple GPUs (cuda:0, cuda:1) for speed.
Saves transcriptions as batch-wise and cumulative CSVs.


# Data Preparation 📊

Merges transcriptions with metadata (train.csv, test.csv) to create train_ready.csv and test_ready.csv.
Train data includes filenames, transcriptions, and labels; test data includes filenames and transcriptions.


# Model Training 🤖

Fine-tunes a RoBERTa-large model for regression to predict grammar scores (1–5).
Splits transcriptions into chunks (150 chars, 50-char overlap) to handle long texts.
Trains for 5 epochs with a learning rate of 2e-6, tracking MSE and accuracy.


# Prediction & Submission 📤

Processes test transcriptions, predicts scores per chunk, and averages them per file.
Clamps predictions to [1, 5] and saves them as submission.csv.




# 🛠️ Setup Instructions

Install Dependencies 📦
pip install openai-whisper transformers pandas torch sklearn tqdm


Set Up Kaggle API 🔑

Ensure you have a Kaggle account and API token.
Run the Kaggle login cell in the notebook to authenticate:import kagglehub
kagglehub.login()


# Prepare Data 📂

Download the dataset from the VoiceCraft Self Project dataset and place it in the dataset/ folder.
Ensure paths in the notebook match your local setup.

Execute cells sequentially to transcribe, train, and generate predictions.

# 📈 Results

Training Performance 📊

Final epoch: Train MSE ~0.90, Validation MSE ~1.24, Validation Accuracy ~0.46
Model converges steadily, balancing regression precision and rounded-score accuracy.


# Output Files 💾

train_ready.csv: Transcriptions + labels for training data.
test_ready.csv: Transcriptions for test data.
submission.csv: Predicted grammar scores for test files (e.g., audio_706.wav: 3).




# 🧠 Challenges & Solutions

Long Transcriptions 📜

Challenge: RoBERTa has a 512-token limit, but transcriptions can be longer.
Solution: Split texts into 150-character chunks with 50-character overlap to preserve context.


GPU Utilization ⚡

Challenge: Processing thousands of audio files is computationally intensive.
Solution: Parallelized transcription across multiple GPUs using Python’s multiprocessing.


Noisy Audio 🎙️

Challenge: Audio quality varies, affecting transcription accuracy.
Solution: Used Whisper’s medium model with tuned parameters (no_speech_threshold=0.0, logprob_threshold=-1.0) for robustness.




# 🌈 Future Improvements

🧪 Experiment with Models: Try Whisper-large or DistilRoBERTa for better accuracy or faster inference.
📊 Advanced Metrics: Incorporate BLEU or ROUGE scores to evaluate transcription quality.
⚙️ Hyperparameter Tuning: Optimize LEARNING_RATE, CHUNK_SIZE, or EPOCHS using grid search.
🌐 Multilingual Support: Extend Whisper to handle non-English audio if needed.


# 🙌 Acknowledgments

OpenAI Whisper: For robust audio transcription. 🎙️
Hugging Face Transformers: For the powerful RoBERTa model. 🤗
Kaggle: For hosting the VoiceCraft Self Project dataset. 🏆
