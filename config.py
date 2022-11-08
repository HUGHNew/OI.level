import os
#region dataset
dataset_path = "OI-dataset"
fast_assem = "fast.txt"
assem_file = "assem.txt"

file_fast_relat = os.path.join(dataset_path, fast_assem)
#endregion dataset

#region models
model_path = "models"
fast_model = "fastc.bin"
assem_model = "assem.pt"

model_fast_relat = os.path.join(model_path, fast_model)
#endregion models