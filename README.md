# Re-Designing CTRL-Adapter: Efficient and Adaptive Multi-Condition Composition
# Environment Setup
```
conda create -n new-adapter python==3.10
conda activate new-adapter
pip install -r requirements_inference.txt # install from this if you only need to perform inference
pip install -r requirements_train.txt # install from this if you plan to do some training
```
# How to train
### Step 1: Download Training Data
According to original paper, we also use 300k images from the [LAION POP](https://laion.ai/blog/laion-pop/) dataset. You can download a subset from this dataset [here](https://huggingface.co/datasets/Ejafa/ye-pop).
### Step 2: Prepare Data in Specified Format
We support loading the dataset in the form of a list from multiple image folders. You only need to fill in the correct paths in ```DATA_PATH```, ```train_data_path```, ```and train_prompt_path``` of the training configuration files.
### Step 3: Set Up Training Scripts
All training configuration files and training scripts are placed under ```./configs``` and ```train_scripts``` respectively. 
### Step 4: Multi-Condition Control 
You can run the command to do multi-condition control training on sdxl.
```
sh train_scripts/sdxl/sdxl_train_multi_condition.sh
```
If you have any problems met, you can visit the original github web (https://github.com/HL-hanlin/Ctrl-Adapter/blob/main/assets/train_guideline.md) for help.

# Run Inference Scripts
First, you should fill the checkpoint path into ```local_checkpoint_path``` of the inference scripts. If not, it will have error.

You can use the command to run inference.
```
sh inference_scripts/sdxl/sdxl_inference_extract_multi_from_raw_image.sh
```