## Installation
Please follow the instruction from ![LLaMafactory](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#installation)


## Training steps

### 1. Modify params file
The params file path is 'examples/train_lora/llava1_6_lora_orpo.yaml'.
Note that the important params are num_train_epochs, output_dir, max_samples.


### 2. Training command
'''
llamafactory-cli train examples/train_lora/llava1_6_lora_orpo.yaml
'''

### 3. Merge the finetuned params with the base model
First, modify the merge config file 'examples/merge_lora/llava1_6_lora_orpo.yaml'.
Note that the important params are adapter_name_or_path, export_dir.

Second, run the command below:
'''
llamafactory-cli export examples/merge_lora/llava1_6_lora_orpo.yaml
'''

Finally, you could get the finetuned checkpoint results in the export_dir directory.