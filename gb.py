import torch, gc
import wandb

# def get_best_params(sweep_id):
#     api = wandb.Api()
#     sweep = api.sweep(sweep_id)

#     # Print available keys for debugging
#     # for run in sweep.runs:
#     #     print(run.summary.keys())

#     best_run = min(sweep.runs, key=lambda run: run.summary.get('val_loss', float('inf')))
#     return best_run.config

# # Example usage
# sweep_id = "optimize_project_aws/hzwxs0lv"  # Replace with your actual sweep ID
# best_params = get_best_params(sweep_id)
# print(best_params)

gc.collect()
torch.cuda.empty_cache()
