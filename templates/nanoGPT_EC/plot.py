import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp

# LOAD FINAL RESULTS:
datasets = ["shakespeare_char"]
folders = os.listdir("./")
final_results = {}
results_info = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()
        run_info = {}
        print("Final results keys:", list(results_dict.keys()))
        for dataset in datasets:
            run_info[dataset] = {}
            val_losses = []
            train_losses = []
            for k in results_dict.keys():
                if dataset in k and "train_info" in k:
                    run_info[dataset]["gene_id"] = [info["gene_id"] for info in results_dict[k]]
                    run_info[dataset]["generation"] = [info["generation"] for info in results_dict[k]]
                    run_info[dataset]["iters"] = [info["iter"] for info in results_dict[k]]
                    run_info[dataset]["loss"] = [info["loss"] for info in results_dict[k]]
                if dataset in k and "val_info" in k:
                    run_info[dataset]["iters"] = [info["iter"] for info in results_dict[k]]
                    val_losses.append([info["val/loss"] for info in results_dict[k]])
                    train_losses.append([info["train/loss"] for info in results_dict[k]])
                mean_val_losses = np.mean(val_losses, axis=0)
                mean_train_losses = np.mean(train_losses, axis=0)
                if len(val_losses) > 0:
                    sterr_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))
                    stderr_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
                else:
                    sterr_val_losses = np.zeros_like(mean_val_losses)
                    stderr_train_losses = np.zeros_like(mean_train_losses)
                run_info[dataset]["val_loss"] = mean_val_losses
                run_info[dataset]["train_loss"] = mean_train_losses
                run_info[dataset]["val_loss_sterr"] = sterr_val_losses
                run_info[dataset]["train_loss_sterr"] = stderr_train_losses
        results_info[folder] = run_info
        print(run_info)

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baselines",
}

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs)+10)

# Plot 1: Line plot of training loss for each dataset across the runs with labels
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    markers = [ 'o', '^','*','+','s', 'd', '*', 'p']
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["iters"]
        generation = results_info[run][dataset]["generation"]
        gene_id = results_info[run][dataset]["gene_id"]

        max_gen = generation[-1]+1
        max=len(generation)
        length = max//max_gen

        max_gene_id = gene_id[-1]+1


        mark=0

        gene_id=0

        print(f"len:{length}")

        for j in range(0,max,length):
            for k in range(0,length,length//max_gene_id):
                loss = results_info[run][dataset]["loss"][j+k:j+k+length//max_gene_id]
                iters=range(0,len(loss)*10,10)
                plt.plot(iters,loss, markers[mark%len(markers)],  color=colors[gene_id], label=f"gen:{gene_id+1} gene:{chr(65 + (mark+gene_id*4) % 26)}")
                #plt.errorbar(iters, loss, fmt=markers[mark%len(markers)],  color=colors[gene_id])  # 点でプロット
                print(k)
                mark +=1
            gene_id+=1
            mark=0



    plt.title(f"Training Loss Across Runs for {dataset} Dataset")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"train_loss_{dataset}.png")
    plt.close()

# Plot 2: Line plot of validation loss for each dataset across the runs with labels
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["iters"]
        mean = results_info[run][dataset]["val_loss"]
        sterr = results_info[run][dataset]["val_loss_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"Validation Loss Across Runs for {dataset} Dataset")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"val_loss_{dataset}.png")
    plt.close()
