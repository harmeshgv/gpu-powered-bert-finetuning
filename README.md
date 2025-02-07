# GPU-Powered BERT Fine-Tuning

## Introduction
This project demonstrates efficient fine-tuning of BERT models utilizing CUDA-powered GPUs. It's specifically optimized for laptops and devices equipped with NVIDIA RTX 3000/4000 series or other CUDA-compatible GPUs. The project is perfect for fast NLP model training using PyTorch and Hugging Face Transformers.

## Setup

### Creating a Virtual Environment
To ensure a clean and isolated environment for your project, it's recommended to use a virtual environment. You can create one and activate it using the following commands:

1. Create a virtual environment named `myenv`:

    ```bash
    python -m venv myenv
    ```

2. Activate the virtual environment:

   - On Windows, run:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS and Linux, run:
     ```bash
     source myenv/bin/activate
     ```

### Installing Dependencies
After activating the virtual environment, upgrade `pip` and install the required packages.

1. Upgrade `pip` to the latest version:

    ```bash
    python -m pip install --upgrade pip
    ```

2. Install all required packages via `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

3. Install PyTorch and its dependencies with CUDA support. This ensures that all operations are optimized for GPU execution:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Execution

1. **GPU Availability Check**: Ensure that your environment recognizes the GPU and its version. You can do this by running the following checks in your Python script:

    ```python
    import torch
    print(torch.cuda.is_available())  # Should print True if GPU is available
    print(torch.cuda.get_device_name(0))  # Check the GPU name
    ```

2. **Load and Fine-Tune the Model**:
   - Load the IMDb dataset and tokenize it.
   - Instantiate the BERT model and move it to the GPU device.
   - Configure training parameters and fine-tune the model using `Trainer` from the Hugging Face `transformers` library.

3. **Save and Load the Model**:
   - Save the fine-tuned model and tokenizer.
   - Reload the model using the `pipeline` function for text classification.

```python
# Load the fine-tuned model
classifier = pipeline("text-classification", model="fine_tuned_bert", device=0 if torch.cuda.is_available() else -1)
```

## Conclusion

This guide facilitates swift and effective fine-tuning of BERT models on GPU-enabled systems, significantly reducing training time. Additionally, this setup demonstrates best practices for managing Python dependencies using virtual environments.

By following this README, you will be able to reproduce the results and adapt the methods for other models or datasets as needed.