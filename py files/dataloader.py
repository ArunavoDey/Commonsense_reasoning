import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import sys
import time
#import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import random
from scipy.fft import fft, ifft, fftfreq
import copy
import scipy
from sklearn.model_selection import train_test_split
from typing import Dict, List
from datasets import Dataset, load_dataset, disable_caching
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from functools import partial
import copy
from transformers import DataCollatorForSeq2Seq
import huggingface_hub
from huggingface_hub import login



class dataLoader():
  def __init__(self, src_path, tar_path):
    self.src_path = src_path
    self.tar_path = tar_path
    self.MAX_LENGTH = 256
  def _add_text(self, rec):
    self.prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction: {instruction}\n Response:"""
    self.answer_template = """{response}"""
    instruction = rec["concepts"]
    response = rec["target"]
    # check if both exists, else raise error
    if not instruction:
        raise ValueError(f"Expected an instruction in: {rec}")
    if not response:
        raise ValueError(f"Expected a response in: {rec}")
    rec["prompt"] = self.prompt_template.format(instruction=instruction)
    rec["answer"] = self.answer_template.format(response=response)
    rec["text"] = rec["prompt"] + rec["answer"]
    return rec
  def loadData(self):
    login("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")
    disable_caching()
    dataset = load_dataset(self.src_path, split = 'train')
    self.small_dataset = dataset.select([i for i in range(200)])
    print(self.small_dataset)
    print(self.small_dataset[0])
    # creating templates
    # running through all samples
    self.small_dataset = self.small_dataset.map(self._add_text)
    print(self.small_dataset[0])
  # Function to generate token embeddings from text part of batch
  def _set_tokenizer(self,model_id):
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.tokenizer.pad_token = self.tokenizer.eos_token
  def _preprocess_batch(self, batch: Dict[str, List]):
    model_inputs = self.tokenizer(batch["text"], max_length=self.MAX_LENGTH, truncation=True, padding='max_length')
    model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
    return model_inputs


  def getData(self):
    return self.src_data, self.tar_data

  def getXY(self, model_id):
    self._set_tokenizer(model_id)
    _preprocessing_function = partial(self._preprocess_batch)
    # apply the preprocessing function to each batch in the dataset
    encoded_small_dataset = self.small_dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=["concepts", "target", "prompt", "answer"],
    )
    processed_dataset = encoded_small_dataset.filter(lambda rec: len(rec["input_ids"]) <= self.MAX_LENGTH)

    # splitting dataset
    split_dataset = processed_dataset.train_test_split(test_size=14, seed=0)
    print(split_dataset)
    return split_dataset
  def getTokenizer(self):
    return self.tokenizer
  def getMaxLength(self):
    return self.MAX_LENGTH
  # Function to format the response and filter out the instruction from the response.
  def postprocess(self, response):
    messages = response.split("Response:")
    if not messages:
        raise ValueError("Invalid template for prompt. The template should include the term 'Response:'")
    return "".join(messages[1:])
