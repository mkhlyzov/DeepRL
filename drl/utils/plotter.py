from matplotlib import pyplot as plt
import pandas as pd  # for pd.Series.rolling.mean() and ewm.mean()


def plot(
    data: dict,
    x_key: str,
    smoothen: str = None,
    window: int = 100,
):
    assert len(data) < 4  # Only 2 metricses are supported

    plt.close('all')
    y_keys = [k for k in data if k != x_key]
    # base_figsize = (10, 5)

    font = {
        'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
    }

    fig, ax = plt.subplots(
        nrows=len(y_keys), ncols=1, sharex=True, constrained_layout=True,
        figsize=(10, 10)
    )
    for i, y_key in enumerate(y_keys):
        x, y = _smoothen(data[x_key], data[y_key], smoothen, window)
        ax[i].plot(
            x, y, #grid=True,
            label=y_key
        )
        ax[i].grid()
        ax[i].set_ylabel(y_key , fontdict=font)
        # ax[i].set_title(y_key, fontdict=font)
    ax[-1].set_xlabel('#' + x_key, fontdict=font)

    plt.show()


def _smoothen(x, y, mod, window):
    if mod is None:
        ...
    elif mod in ['rolling', 'window', 'simple', 'sma']:
        x, y = _rolling_average(x, y, window)
    elif mod in ['exponential', 'epx', 'ema']:
        x, y = _exponential_average(x, y, window)
    elif mod in ['cumulative', 'cma']:
        raise NotImplementedError()
    else:
        raise ValueError()

    return x, y


def _rolling_average(x, y, window):
    y = pd.Series(y).rolling(window).mean().tolist()
    x = x[window - 1:]
    y = y[window - 1:]
    return x, y


def _exponential_average(x, y, window):
    alpha = 1 / window
    y = pd.Series(y).ewm(alpha=alpha, min_periods=0).mean().tolist()
    x = x[:]
    y = y[:]
    return x, y
