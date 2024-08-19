import sys
import os
#import huggingface_hub
#from huggingface_hub import login
#login("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")
from typing import Dict, List
from datasets import Dataset, load_dataset, disable_caching
#disable_caching() disable huggingface cache
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset
from IPython.display import Markdown
from dataloader import dataLoader
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import bitsandbytes
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Dataset Preparation
if __name__ == "__main__":
    loader = dataLoader( "allenai/common_gen", "")
    loader.loadData()
    # loading the tokenizer for dolly model. The tokenizer converts raw text into tokens
    model_id = "databricks/dolly-v2-3b"
    #loading the model using AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # use_cache=False,
        device_map="auto", #"balanced",
        load_in_8bit=True,
        torch_dtype=torch.float16
    )
    split_dataset = loader.getXY(model_id)
    tokenizer = loader.getTokenizer()
    # resizes input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
    model.resize_token_embeddings(len(tokenizer))
    #split_dataset = loader.getXY(model_id)
    # takes a list of samples from a Dataset and collate them into a batch, as a dictionary of PyTorch tensors.
    data_collator = DataCollatorForSeq2Seq(
        model = model, tokenizer=tokenizer, max_length=loader.getMaxLength(), pad_to_multiple_of=8, padding='max_length')

    LORA_R = 256 # 512
    LORA_ALPHA = 512 # 1024
    LORA_DROPOUT = 0.05
    # Define LoRA Config
    lora_config = LoraConfig(
                 r = LORA_R, # the dimension of the low-rank matrices
                 lora_alpha = LORA_ALPHA, # scaling factor for the weight matrices
                 lora_dropout = LORA_DROPOUT, # dropout probability of the LoRA layers
                 bias="none",
                 task_type="CAUSAL_LM",
                 target_modules=["query_key_value"],
    )

    # Prepare int-8 model for training - utility function that prepares a PyTorch model for int8 quantization training. <https://huggingface.co/docs/peft/task_guides/int8-asr>
    model = prepare_model_for_kbit_training(model)
    # initialize the model with the LoRA framework
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    # define the training arguments first.
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    MODEL_SAVE_FOLDER_NAME = "dolly-3b-lora"
    training_args = TrainingArguments(
                    output_dir=MODEL_SAVE_FOLDER_NAME,
                    overwrite_output_dir=True,
                    fp16=True, #converts to float precision 16 using bitsandbytes
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    learning_rate=LEARNING_RATE,
                    num_train_epochs=EPOCHS,
                    logging_strategy="epoch",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
    )
    # training the model
    trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=split_dataset['train'],
            eval_dataset=split_dataset["test"],
            data_collator=data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    # only saves the incremental 🤗 PEFT weights (adapter_model.bin) that were trained, meaning it is super efficient to store, transfer, and load.
    trainer.model.save_pretrained(MODEL_SAVE_FOLDER_NAME)
    # save the full model and the training arguments
    trainer.save_model(MODEL_SAVE_FOLDER_NAME)
    trainer.model.config.save_pretrained(MODEL_SAVE_FOLDER_NAME)

    # Prompt for prediction
    inference_prompt = "'concepts': ['swim', 'swimmingpool', 'swimmer']"
    # Inference pipeline with the fine-tuned model
    inf_pipeline =  pipeline('text-generation', model=trainer.model, tokenizer=tokenizer, max_length=256, trust_remote_code=True)
    # Format the prompt using the `prompt_template` and generate response 
    response = inf_pipeline(loader.prompt_template.format(instruction=inference_prompt))[0]['generated_text']
    # postprocess the response
    formatted_response = loader.postprocess(response)
    print(formatted_response)



