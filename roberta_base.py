import loralib as lora
from transformers import AutoTokenizer, Trainer, TrainingArguments, RobertaForSequenceClassification, set_seed
import torch.nn as nn
from datasets import load_dataset
import pandas as pd

# This function replaces layer parameter with a lora.Linear version of the same layer. This code is adapted from the code at this source:  https://www.kaggle.com/code/krist0phersmith/lora-implementation.
def make_lora_layer(layer, lora_r=8): # Set lora_r according to Table 9
    new_layer = lora.Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is None,
        r=lora_r,
        lora_alpha=8, # Set lora_alpha according to Table 9
        merge_weights=False
    )

    new_layer.weight = nn.Parameter(layer.weight.detach())

    if layer.bias is not None:
        new_layer.bias = nn.Parameter(layer.bias.detach())

    return new_layer

# UPDATE: Random seed -- By experiment iteration
set_seed(109) #622, 991, 938, 558

# Load in tokenizer and model
model_name = "roberta-base"
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512) # Set model_max_length according to Table 9
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device=device) # UPDATE: num_labels -- GLUE task

# Mark only the query and value parameters as trainable
for i in range(len(model.roberta.encoder.layer)):
  model.roberta.encoder.layer[i].attention.self.query = make_lora_layer(model.roberta.encoder.layer[i].attention.self.query)
  model.roberta.encoder.layer[i].attention.self.value = make_lora_layer(model.roberta.encoder.layer[i].attention.self.value)

lora.mark_only_lora_as_trainable(model)

# Loads in the train, validation, and test datasets
train_dataset = load_dataset('glue', 'cola', split='train') # UPDATE: second parameter -- GLUE task
val_dataset = load_dataset('glue', 'cola', split='validation') # UPDATE: second parameter -- GLUE task, split -- only for MNLI
test_dataset = load_dataset('glue', 'cola', split='test') # UPDATE: second parameter -- GLUE task, split -- only for MNLI

# Save the test dataset indices as list
test_idxs = test_dataset['idx']

# Find labels for task and create dictionaries
labels = set(train_dataset["label"] + val_dataset["label"])
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# Create functions for tokenizing datasets
def change_label(example):
    if example["label"] != -1:
        example["label"] = label2id[example["label"]]
    return example

def tokenize(examples):
    # UPDATE -- Comment for CoLA and SST-2 tasks
    # args = ((examples["sentence1"], examples["sentence2"])) # UPDATE: args -- GLUE task
    # return tokenizer(*args, padding='max_length', truncation=True)
    
    # UPDATE -- Uncomment for CoLA and SST-2 tasks
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

# Tokenize datasets and change labels for train and validation datasets. Tokenize and remove the label column from test dataset.
train_dataset = train_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.map(change_label)

val_dataset = val_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(change_label)

test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['label'])

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30, # UPDATE -- Table 9
    per_device_train_batch_size=16, # UPDATE -- Table 9
    per_device_eval_batch_size=16, # UPDATE -- Table 9
    evaluation_strategy="epoch",
    learning_rate=5e-4, # UPDATE -- Table 9
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    warmup_ratio=0.06 # Set according to Table 9
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and save model
trainer.train()
trainer.save_model("models/roberta-base-cola-109") # UPDATE -- GLUE task and experiment iteration
trainer.evaluate()

# Predict results for test datset
predicted_results = trainer.predict(test_dataset)
predicted_labels = predicted_results.predictions.argmax(-1)
predicted_labels = predicted_labels.flatten().tolist()
predicted_labels = [id2label[l] for l in predicted_labels]

predictions_df = pd.DataFrame({'idx': test_idxs, 'label': predicted_labels})

# Save results for test dataset
predictions_df.to_csv("outputs/roberta_base/109/CoLA.tsv", sep="\t") # UPDATE -- GLUE task