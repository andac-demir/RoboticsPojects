import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def read_lat():
    lat_stationary = pd.read_csv('stationary_data/stationary_latitude.txt',
                                 names=['time', 'lat'], skiprows=1)
    lat_active = pd.read_csv('active_data/active_latitude.txt',
                             names=['time', 'lat'], skiprows=1)
    return lat_stationary, lat_active


def read_lon():
    lon_stationary = pd.read_csv('stationary_data/stationary_longitude.txt',
                                 names=['time', 'lon'], skiprows=1)
    lon_active = pd.read_csv('active_data/active_longitude.txt',
                             names=['time', 'lon'], skiprows=1)
    return lon_stationary, lon_active


def read_alt():
    alt_stationary = pd.read_csv('stationary_data/stationary_altitude.txt',
                                 names=['time', 'alt'], skiprows=1)
    alt_active = pd.read_csv('active_data/active_altitude.txt',
                             names=['time', 'alt'], skiprows=1)
    return alt_stationary, alt_active


def read_utm_easting():
    easting_stationary = pd.read_csv('stationary_data/stationary_easting.txt',
                                     names=['time', 'utm_easting'], skiprows=1)
    easting_active = pd.read_csv('active_data/active_easting.txt',
                                 names=['time', 'utm_easting'], skiprows=1)
    return easting_stationary, easting_active


def read_utm_northing():
    northing_stationary = pd.read_csv('stationary_data/stationary_northing.txt',
                                      names=['time', 'utm_northing'], skiprows=1)
    northing_active = pd.read_csv('active_data/active_northing.txt',
                                  names=['time', 'utm_northing'], skiprows=1)
    return northing_stationary, northing_active


def read_utm_zone():
    zone_stationary = pd.read_csv('stationary_data/stationary_utm_zone.txt',
                                  names=['time', 'utm_zone'], skiprows=1)
    zone_active = pd.read_csv('active_data/active_utm_zone.txt',
                              names=['time', 'utm_zone'], skiprows=1)
    return zone_stationary, zone_active


def read_utm_letter():
    letter_stationary = pd.read_csv('stationary_data/stationary_utm_letter.txt',
                                    names=['time', 'utm_letter'], skiprows=1)
    letter_active = pd.read_csv('active_data/active_utm_letter.txt',
                                names=['time', 'utm_letter'], skiprows=1)
    return letter_stationary, letter_active


def _plot_path(df_list):
    for i, df in enumerate(df_list):
        plt.subplot(3, 1, 1)
        plt.plot(df['utm_easting'], df['utm_northing'])
        if i == 1:
            plt.axhline(y=df[df.time >= 1.2]['utm_northing'].iloc[0],
                        linestyle='--',  color='r', label="correction time")
        plt.ylabel('utm northing (m)')
        plt.xlabel('utm easting (m)')
        plt.subplots_adjust(hspace=0.9)

        plt.subplot(3, 1, 2)
        plt.plot(df['time'], df['utm_easting'])
        if i == 1:
            plt.axvline(x=1.2, linestyle='--',  color='r', label="correction time")
            plt.ylabel('utm easting (m)')
        if i == 0:
            std_easting = np.std(df['utm_easting'])
            plt.ylabel('utm easting (m)\nstd: %.2f' % std_easting)
        plt.xlabel('time (min)')
        plt.subplots_adjust(hspace=0.9)

        plt.subplot(3, 1, 3)
        plt.plot(df['time'], df['utm_northing'])
        if i == 1:
            plt.axvline(x=1.2, linestyle='--',  color='r', label="correction time")
            plt.legend(loc="lower right", frameon=False)
            plt.ylabel('utm northing (m)')
        if i == 0:
            std_northing = np.std(df['utm_northing'])
            plt.ylabel('utm northing (m)\nstd: %.2f' % std_northing)
        plt.xlabel('time (min)')

        if i == 0:
            _fig = "stationary"
        else:
            _fig = "active"
        plt.savefig(_fig + 'travel_path.pdf')
        plt.show()


def _plot_altitude(df_list):
    for i, df in enumerate(df_list):
        plt.plot(df['time'], df['alt'])
        plt.ylabel('Altitude above sea level (m)')
        plt.xlabel('time (min)')

        if i == 0:
            _fig = "stationary"
        else:
            _fig = "active"
        plt.savefig(_fig + 'altitude_path.pdf')
        plt.show()


def _get_noise_distribution(df):
    # for utm northing
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(df[df.time >= 1.2]['time'].values.reshape(-1, 1),
                         df[df.time >= 1.2]['utm_northing'].values.reshape(-1, 1))
    northing_pred = linear_regressor.predict(df[df.time >= 1.2]['time'].values.reshape(-1, 1))
    northing_noise = northing_pred - df[df.time >= 1.2]['utm_northing'].values.reshape(-1, 1)

    # for utm easting
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(df[df.time >= 1.2]['time'].values.reshape(-1, 1),
                         df[df.time >= 1.2]['utm_easting'].values.reshape(-1, 1))
    easting_pred = linear_regressor.predict(df[df.time >= 1.2]['time'].values.reshape(-1, 1))
    easting_noise = easting_pred - df[df.time >= 1.2]['utm_easting'].values.reshape(-1, 1)

    plt.subplot(2, 1, 1)
    plt.ylabel('utm_northing (m)')
    plt.plot(df[df.time >= 1.2]['time'], df[df.time >= 1.2]['utm_northing'],
             label="true meas.")
    plt.plot(df[df.time >= 1.2]['time'], northing_pred, color='r',
             linestyle='--', label="expected")

    plt.subplot(2, 1, 2)
    plt.ylabel('utm_easting (m)')
    plt.plot(df[df.time >= 1.2]['time'], df[df.time >= 1.2]['utm_easting'],
             label="true meas.")
    plt.plot(df[df.time >= 1.2]['time'], easting_pred, color='r',
             linestyle='--', label="expected")
    plt.xlabel('time (min)')
    plt.legend(loc="lower right", frameon=False)

    plt.savefig('true_and_measured.pdf')
    plt.show()

    plt.plot()
    plt.ylabel('noise distribution (m)')
    plt.plot(df[df.time >= 1.2]['time'], northing_noise, color='b',
             label="utm_northing_noise")
    plt.plot(df[df.time >= 1.2]['time'], easting_noise, color='r',
             label="utm_easting_noise")
    plt.legend(loc="lower right", frameon=False)

    plt.savefig('noise_distribution.pdf')
    plt.show()

    mean_e, std_e = np.mean(easting_noise), np.std(easting_noise)
    mean_n, std_n = np.mean(northing_noise), np.std(northing_noise)
    print("utm easting has noise distribution with mean %.4f m and std %.2f m"
          % (mean_e, std_e))
    print("utm northing has noise distribution with mean %.4f m and std %.2f m"
          % (mean_n, std_n))

def main():
    lat_stationary, lat_active = read_lat()
    lon_stationary, lon_active = read_lon()
    alt_stationary, alt_active = read_alt()
    easting_stationary, easting_active = read_utm_easting()
    northing_stationary, northing_active = read_utm_northing()
    zone_stationary, zone_active = read_utm_zone()
    letter_stationary, letter_active = read_utm_letter()

    stationary_df = pd.concat([lat_stationary['time'].div(10**9), # to secs
                               lat_stationary['lat'],
                               lon_stationary['lon'],
                               alt_stationary['alt'],
                               easting_stationary['utm_easting'],
                               northing_stationary['utm_northing'],
                               zone_stationary['utm_zone'],
                               letter_stationary['utm_letter']], axis=1)

    active_df = pd.concat([lat_active['time'].div(10**9),
                           lat_active['lat'],
                           lon_active['lon'],
                           alt_active['alt'],
                           easting_active['utm_easting'],
                           northing_active['utm_northing'],
                           zone_active['utm_zone'],
                           letter_active['utm_letter']], axis=1)

    # convert sec to min.
    stationary_df['time'] = (stationary_df['time'] - stationary_df['time'].iloc[0]) / 60
    active_df['time'] = (active_df['time'] - active_df['time'].iloc[0]) / 60

    for df in [stationary_df, active_df]:
        ref_x, ref_y = df['utm_easting'].iloc[0], df['utm_northing'].iloc[0]
        df['utm_easting'] -= ref_x
        df['utm_northing'] -= ref_y

    _plot_path([stationary_df, active_df])
    _plot_altitude([stationary_df, active_df])
    _get_noise_distribution(active_df)


if __name__ == "__main__":
    main()

