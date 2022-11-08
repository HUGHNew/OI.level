import os
import torch
#region dataset
dataset_path = "OI-dataset"
fast_assem = "fast.txt"
assem_file = "assem.txt"

file_fast_relat = os.path.join(dataset_path, fast_assem)
file_assm_relat = os.path.join(dataset_path, assem_file)
#endregion dataset

#region models
model_path = "models"
fast_model = "fastc.bin"
assem_model = "assem.pt"
dict_model_param = "asm_model.pt"
dict_optim_param = "asm_optim.pt"

dict_model_file = os.path.join(model_path, "asm_model.pt")
dict_optim_file = os.path.join(model_path, "asm_optim.pt")
model_fast_relat = os.path.join(model_path, fast_model)
model_assm_relat = os.path.join(model_path, assem_model)
#endregion models

device = "cuda" if torch.cuda.is_available() else "cpu"