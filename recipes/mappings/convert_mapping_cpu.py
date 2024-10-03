import pickle
import torch

# load pickle file 
file_path = './all_scores_tp1_l2_mean_32x32.pkl'
save_path = './all_scores_tp1_l2_mean_32x32_cpu.pt'
mapping = pickle.load(open(file_path, 'rb'))
print("Loaded mapping from file: ", file_path)

# for everything in the mapping, convert to cpu
for key in mapping:
    if isinstance(mapping[key], torch.Tensor):
        mapping[key] = mapping[key].cpu()
    else:
        # inner dict
        for inner_key in mapping[key]:
            mapping[key][inner_key] = mapping[key][inner_key].cpu()

torch.save(mapping, save_path)
print("Saved mapping to file: ", save_path)
