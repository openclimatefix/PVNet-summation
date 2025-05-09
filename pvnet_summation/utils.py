"""Utils"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab


def plot_forecasts(y, y_hat, times, batch_idx=None, quantiles=None, y_sum=None):
    """Plot a batch of data and the forecast from that batch"""

    times_utc = times.cpu().numpy().squeeze().astype("datetime64[ns]")
    times_utc = [pd.to_datetime(t) for t in times_utc]
    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()
    y_sum = y_sum.cpu().numpy() if (y_sum is not None) else None

    batch_size = y.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, ax in enumerate(axes.ravel()):
        if i >= batch_size:
            ax.axis("off")
            continue

        ax.plot(times_utc[i], y[i], marker=".", color="k", label=r"$y$")

        if y_sum is not None:
            ax.plot(
                times_utc[i], y_sum[i], marker=".", linestyle="--", color="k", label=r"$y_{sum}$"
            )

        if quantiles is None:
            ax.plot(times_utc[i], y_hat[i], marker=".", color="r", label=r"$\hat{y}$")
        else:
            cm = pylab.get_cmap("twilight")
            for nq, q in enumerate(quantiles):
                ax.plot(
                    times_utc[i],
                    y_hat[i, :, nq],
                    color=cm(q),
                    label=r"$\hat{y}$" + f"({q})",
                    alpha=0.7,
                )

        ax.set_title(f"{times_utc[i][0].date()}", fontsize="small")

        xticks = [t for t in times_utc[i] if t.minute == 0][::2]
        ax.set_xticks(ticks=xticks, labels=[f"{t.hour:02}" for t in xticks], rotation=90)
        ax.grid()

    axes[0, 0].legend(loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (hour of day)")

    if batch_idx is not None:
        title = f"Normed National output : batch_idx={batch_idx}"
    else:
        title = "Normed National output"
    plt.suptitle(title)
    plt.tight_layout()

    return fig
