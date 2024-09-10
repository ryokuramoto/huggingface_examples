# Hugging Face Examples

Welcome to the **Hugging Face Examples** repository! This collection showcases a variety of use cases using Hugging Face's popular libraries and models. Hugging Face is renowned for its contributions to machine learning.

The examples in this repository are designed to help you get started with Hugging Face’s tools like `transformers` and cover a range of tasks without needing a Hugging Face account. These notebooks offer hands-on demonstrations with essential Hugging Face libraries.

## Notebooks Included

### 1. `huggingface_pipeline.ipynb` - Hugging Face Pipeline Demonstration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryokuramoto/huggingface_examples/blob/main/notebooks/huggingface_pipeline.ipynb)

This notebook provides an introduction to Hugging Face's pipeline functionality, focusing on different NLP tasks such as:

- Sentiment Analysis
- Named Entity Recognition (NER)
- Question Answering
- Text Generation

### 2. `huggingface_fine_tuning.ipynb` - Fine-Tuning a Hugging Face Model for Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryokuramoto/huggingface_examples/blob/main/notebooks/huggingface_fine_tuning.ipynb)

Learn how to fine-tune a pre-trained Hugging Face model for sentiment analysis using the IMDb dataset. This notebook walks through:

- Setting up Hugging Face tools.
- Loading and preprocessing the IMDb dataset.
- Fine-tuning a pre-trained model.
- Evaluating the model's performance.

This guide is perfect for anyone looking to adapt pre-trained models for specific tasks and datasets. Running this notebook in Google Colab with GPU support is highly recommended for faster training.

### 3. `huggingface_fine_tuning_from_checkpoint.ipynb` - Fine-Tuning a Hugging Face Model from a Checkpoint

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryokuramoto/huggingface_examples/blob/main/notebooks/huggingface_fine_tuning_from_checkpoint.ipynb)

This notebook extends the fine-tuning example, introducing methods for resuming training or evaluating models from checkpoints. It includes:

- Loading a pre-trained Hugging Face model.
- Evaluating performance before and after fine-tuning.
- Using checkpoints for longer or interrupted training runs.

### 4. `huggingface_stable_diffusion.ipynb` - Image Generation using Stable Diffusion in Hugging Face Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryokuramoto/huggingface_examples/blob/main/notebooks/huggingface_stable_diffusion.ipynb)

Explore the power of text-to-image generation using the Stable Diffusion models (`v1-4`, `v2-1`, `SDXL`, and `FLUX.1 Schnell`). This notebook demonstrates:

- How to set up Stable Diffusion in Hugging Face.
- Comparing different model versions.
- Generating images from text prompts.

This notebook is ideal for anyone interested in experimenting with state-of-the-art generative models for image creation.

### 5. `huggingface_speech_recognition.ipynb` - Speech Recognition using Hugging Face's OpenAI Whisper Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryokuramoto/huggingface_examples/blob/main/notebooks/huggingface_speech_recognition.ipynb)

Discover how to use OpenAI's Whisper model for automatic speech recognition (ASR). This notebook showcases:

- Transcribing audio files or microphone recordings into text.
- Utilizing Hugging Face's integration of the Whisper model.
  
The Whisper model is a powerful tool for converting spoken language into text, making it useful for various speech-to-text applications.

## Getting Started

To use the notebooks in this repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/ryokuramoto/huggingface-examples.git
   cd huggingface-examples
   ```

2. Install the required libraries: Libraries required for each notebook are mentioned within the notebooks.

3. Open the notebooks in Jupyter or Google Colab.

4. Manage downloaded data:
   Hugging Face caches model files to avoid repeated downloads:
   - **Windows**: `C:\Users\<Your Username>\.cache\huggingface`
   
   If you wish to clear these cached files, you can manually delete the contents of the cache folder.

## Recommended Environment

For the best experience, it’s recommended to run these notebooks in Google Colab when working with large models or tasks that require significant computation, such as fine-tuning or image generation. Utilizing a GPU is highly encouraged for faster processing.

You can open the notebook directly in Google Colab by clicking the "Open in Colab" button provided in this README file.

If you'd like to save your notebook or results, here’s how:
- **To save a copy to Google Drive**: Navigate to `File > Save a copy in Drive`.
- **To save a copy locally**: Navigate to `File > Download`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For inquiries or support, please contact:

Ryo Kuramoto  
[ryoh.kuramoto@rknewtech.com](mailto:ryoh.kuramoto@rknewtech.com)
