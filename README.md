# GPU-Powered BERT Fine-Tuning

## Introduction
This project demonstrates efficient fine-tuning of BERT models utilizing CUDA-powered GPUs. It's specifically optimized for laptops and devices equipped with NVIDIA RTX 3000/4000 series or other CUDA-compatible GPUs. The project is perfect for fast NLP model training using PyTorch and Hugging Face Transformers.

## Setup

### Option 1: Manual Setup

#### Creating a Virtual Environment
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

#### Installing Dependencies
After activating the virtual environment, upgrade `pip` and install the required packages.

1. Upgrade `pip` to the latest version:

    ```bash
    python -m pip install --upgrade pip
    ```

2. Install all required packages via `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

3. Install PyTorch with CUDA Support:

    Visit the [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/) page to select the appropriate installation command based on your system's CUDA version. This ensures that all operations are optimized for GPU execution. For example, if you are using CUDA 11.8, you can use the following command:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

### Option 2: Jupyter Notebook Walkthrough

For a detailed, step-by-step walkthrough of the code, you can use the `bert_finetuning_cuda_walkthrough.ipynb` Jupyter Notebook. This notebook provides an interactive way to understand and execute the code:

1. Open the Jupyter Notebook:

   ```bash
   jupyter notebook bert_finetuning_cuda_walkthrough.ipynb
   ```

2. Follow the instructions and execute the cells to see the code in action.

## Execution

1. **GPU Availability Check**: Ensure that your environment recognizes the GPU and its version. You can do this by running the `gpu_check.py` script:

    ```bash
    python gpu_check.py
    ```

2. **Load and Fine-Tune the Model**:
   - Run the `train_model.py` script to load the IMDb dataset, tokenize it, instantiate the BERT model, and fine-tune it:

     ```bash
     python train_model.py
     ```

3. **Streamlit Dashboard**:
   - Launch the Streamlit app to interact with the fine-tuned model:

     ```bash
     streamlit run dashboard.py
     ```

## Outputs
<!--start-->
| Stage                         | Description                                                                      | Model Used        | Accuracy   |
|-------------------------------|----------------------------------------------------------------------------------|-------------------|------------|
| Baseline (Before Fine-Tuning) | Directly used bert-base-uncased pretrained model on raw dataset (no fine-tuning) | Bert (Pretrained) | 52.4%      |
| Fine-Tuning (Raw data)        | Fine-tuned BERT on dataset without additional preprocessing                      | Bert (Fine-Tuned) | 89.4%      |
<!--stop-->
## Conclusion

This guide facilitates swift and effective fine-tuning of BERT models on GPU-enabled systems, significantly reducing training time. Additionally, this setup demonstrates best practices for managing Python dependencies using virtual environments.

By following this README, you will be able to reproduce the results and adapt the methods for other models or datasets as needed. If you encounter any issues or have questions, please refer to the documentation or reach out for support.

---

This README provides clear instructions for setting up the environment manually or using a Jupyter Notebook for an interactive experience. Adjust any specific paths or commands as needed based on your project's structure. If you have any further questions or need additional customization, feel free to ask!
