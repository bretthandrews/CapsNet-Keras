# @Author: Brett Andrews <andrews>
# @Date:   2019-05-14 16:05:88
# @Last modified by:   andrews
<<<<<<< HEAD
# @Last modified time: 2019-05-22 21:05:51
=======
# @Last modified time: 2019-05-22 21:05:51
>>>>>>> refs/remotes/origin/enhancement-1-more_conv_layers

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd


@click.command()
@click.option('--run', default=None, help="Run name.")
@click.option('--path', default=None, type=click.Path(), help="Path to output csv files.")
def plot_accuracy_loss(run, path):
    assert (run is not None) or (path is not None), "Must provide `model` or `path`."

    if path is None:
        path_models = Path.home() / "projects" / "photoz" / "CapsNet-Keras" / "models"
        path = path_models / run.split("_")[0] / run / "results"

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
    click.echo(f"Wrote: {fout}")


if __name__ == "__main__":
    plot_accuracy_loss()
