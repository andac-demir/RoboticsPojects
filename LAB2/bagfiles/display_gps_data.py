import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def parser_gnss():
    gps_mobile = pd.read_csv("CarData/gps_mobile.csv")
    gps_no_engine = pd.read_csv("CarData/gps_no_engine.csv")
    gps_mobile['time (sec)'] = (gps_mobile['field.header.stamp'] -
                                    gps_mobile['field.header.stamp'].iloc[0]).div(10**9)
    gps_no_engine['time (sec)'] = (gps_no_engine['field.header.stamp'] -
                                       gps_no_engine['field.header.stamp'].iloc[0]).div(10**9)
    return gps_mobile, gps_no_engine


def _plot_path(df):
    sns.set(style="white", palette="muted", color_codes=True)
    sns.lineplot(x='field.utm_easting',  # Horizontal axis
                 y='field.utm_northing',  # Vertical axis
                 data=df)  # Data source
    plt.tight_layout()
    plt.savefig('figures/gps_path.pdf')
    plt.show()


def main():
    gps_mobile, gps_no_engine = parser_gnss()
    for df in [gps_mobile, gps_no_engine]:
        ref_x, ref_y = df['field.utm_easting'].iloc[0], df['field.utm_northing'].iloc[0]
        df['field.utm_easting'] -= ref_x
        df['field.utm_northing'] -= ref_y
    _plot_path(gps_mobile)
    #_plot_path(gps_no_engine)


if __name__ == "__main__":
    main()

