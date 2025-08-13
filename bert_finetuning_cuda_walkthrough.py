#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
torch.cuda.empty_cache()

# In[6]:


import torch
print(torch.cuda.is_available())  # Returns True if GPU is available
print(torch.cuda.current_device())  # Displays the current GPU device
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Displays GPU name


# In[7]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import load_dataset

import torch
import transformers
import pandas as pd
from tqdm import tqdm

# Initialize tqdm for pandas
tqdm.pandas(disable=True)


# In[8]:


# Load tokenizer and model from Hugging Face
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# In[9]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# In[10]:


# Load IMDb dataset
dataset = load_dataset("imdb")

# In[11]:


print(dataset)

# In[12]:


# Print the first 5 samples from the IMDb training dataset
for i in range(5):
    print(dataset["train"][i])

# In[ ]:


# Evaluate before fine-tuning
metrics_before = trainer.evaluate(tokenized_datasets["test"].shuffle().select(range(500)))
print("Before Fine-tuning Accuracy:", metrics_before["eval_accuracy"])
    

# In[13]:


# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# In[ ]:


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16= True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle().select(range(1000)),
    eval_dataset=tokenized_datasets["test"].shuffle().select(range(500)),
    compute_metrics=compute_metrics
)
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}





# In[16]:


# Before fine-tuning
metrics_before = trainer.evaluate()
print("Before Fine-tuning Accuracy:", metrics_before["eval_accuracy"])

# In[12]:


# Fine-tune the model
trainer.train()

# In[13]:


# Save the Fine-Tuned Model
model.save_pretrained("fine_tuned_bert")
tokenizer.save_pretrained("fine_tuned_bert")

# In[17]:


from transformers import BertForSequenceClassification, BertTokenizer

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("fine_tuned_bert")
tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert")


# In[18]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["test"].shuffle().select(range(500)),
    compute_metrics=compute_metrics
)

# Evaluate
metrics = trainer.evaluate()
print("Fine-tuned Model Accuracy:", metrics["eval_accuracy"])


# In[14]:


# Load the fine-tuned model
classifier = pipeline("text-classification", model="fine_tuned_bert", device=0 if torch.cuda.is_available() else -1)

# In[18]:


# Test with sample text
print(classifier("This movie is bad"))
print(classifier("This movie is GOOD"))

# In[19]:


dataset

# In[25]:


sample = dataset["train"][0]["text"]
sample

# In[22]:


import re

# In[ ]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    
    return text

# In[26]:


clean_text(sample)

# In[ ]:


from nltk.corpus import stopwords
