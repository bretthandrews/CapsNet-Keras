# @Author: Brett Andrews <andrews>
# @Date:   2019-05-14 16:05:88
# @Last modified by:   andrews
# @Last modified time: 2019-05-22 14:05:00

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Plot accuracy and loss.")
parser.add_argument('--run', default=None, type=str, help="Run name.")
parser.add_argument('--path', default=None, type=str, help="Path to output csv files.")
args = parser.parse_args()


assert (args.run is not None) or (args.path is not None), "Must provide `model` or `path`."

if args.path is None:
    path_models = Path.home() / "projects" / "photoz" / "CapsNet-Keras" / "models"
    path = path_models / args.run.split("_")[0] / args.run / "result"

else:
    run = None

log = pd.read_csv(path / "log.csv")

fig, axes = plt.subplots(nrows=2)
axes[0].plot(log.epoch, log.capsnet_acc, label="train")
axes[0].plot(log.epoch, log.val_capsnet_acc, label="validation")
axes[0].set_ylabel("accuracy")
axes[0].legend()

axes[1].plot(log.epoch, log.capsnet_loss, label="margin")
axes[1].plot(log.epoch, log.decoder_loss, label="decoder")
axes[1].plot(log.epoch, log.loss, label="total")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("loss")
axes[1].legend()

fout = path / "accuracy_loss.pdf"
fig.savefig(fout)
print(f"Wrote: {fout}")
