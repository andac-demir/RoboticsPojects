import pandas as pd
import numpy as np
import math
from scipy import integrate
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

f_imu = 40  # imu sensor frequency: 40Hz

def parser_imu():
    stationary_imu = pd.read_csv("StationaryData/imu_stationary.csv")
    stationary_mag = pd.read_csv("StationaryData/magneto_stationary.csv")
    imu_mobile = pd.read_csv("CarData/imu_mobile.csv")
    imu_no_engine = pd.read_csv("CarData/imu_no_engine.csv")
    mag_mobile = pd.read_csv("CarData/mag_mobile.csv")
    mag_no_engine = pd.read_csv("CarData/mag_no_engine.csv")
    gps_mobile = pd.read_csv("CarData/gps_mobile.csv")
    gps_no_engine = pd.read_csv("CarData/gps_no_engine.csv")

    # along with time headers, add new column, and put there the corrected time
    stationary_imu['corrected_time'] = (stationary_imu['field.header.stamp'] -
                                        stationary_imu['field.header.stamp'].iloc[0]).div(10 ** 9)
    stationary_mag['corrected_time'] = (stationary_mag['field.header.stamp'] -
                                        stationary_mag['field.header.stamp'].iloc[0]).div(10 ** 9)
    imu_mobile['corrected_time'] = (imu_mobile['field.header.stamp'] -
                                    imu_mobile['field.header.stamp'].iloc[0]).div(10 ** 9)
    imu_no_engine['corrected_time'] = (imu_no_engine['field.header.stamp'] -
                                       imu_no_engine['field.header.stamp'].iloc[0]).div(10 ** 9)
    mag_mobile['corrected_time'] = (mag_mobile['field.header.stamp'] -
                                    mag_mobile['field.header.stamp'].iloc[0]).div(10 ** 9)
    gps_mobile['corrected_time'] = (gps_mobile['field.header.stamp'] -
                                    gps_mobile['field.header.stamp'].iloc[0]).div(10 ** 9)
    gps_no_engine['corrected_time'] = (gps_no_engine['field.header.stamp'] -
                                       gps_no_engine['field.header.stamp'].iloc[0]).div(10 ** 9)

    # circular data to calibrate magnetometer is cut from mobile data:
    gps_circular = gps_mobile[gps_mobile['corrected_time'] > 545]
    gps_mobile = gps_mobile[gps_mobile['corrected_time'] <= 545]
    gps_circular['pos_x'] = gps_circular['field.utm_easting'] - \
                            gps_circular['field.utm_easting'].iloc[0]
    gps_circular['pos_y'] = gps_circular['field.utm_northing'] - \
                            gps_circular['field.utm_northing'].iloc[0]
    plt.plot(gps_circular['pos_x'], gps_circular['pos_y'],
             label='Circular data collection GPS')
    plt.legend()
    plt.show()

    # corresponding magnetometer data
    mag_circular = mag_mobile[mag_mobile['field.header.stamp'] >=
                              gps_circular['field.header.stamp'].iloc[0]]

    # do not do integration during circular travels, so ignore these data:
    imu_mobile = imu_mobile[imu_mobile['field.header.stamp'] <=
                            gps_circular['field.header.stamp'].iloc[0]]
    mag_mobile = mag_mobile[mag_mobile['field.header.stamp'] <=
                            gps_circular['field.header.stamp'].iloc[0]]

    # To time align GPS and IMU data
    gps_mobile = gps_mobile[gps_mobile['field.header.stamp'] >=
                            imu_mobile['field.header.stamp'].iloc[0]]
    gps_mobile['corrected_time'] = (gps_mobile['field.header.stamp'] -
                                    gps_mobile['field.header.stamp'].iloc[0]).div(10 ** 9)

    return stationary_imu, stationary_mag, imu_mobile, imu_no_engine, \
           mag_mobile, mag_no_engine, gps_mobile, gps_no_engine, \
           gps_circular, mag_circular


def remove_bias(imu_df, imu_no_engine_df):
    imu_df['raw_linear_acceleration.x'] = imu_df['field.linear_acceleration.x']
    imu_df['raw_linear_acceleration.y'] = imu_df['field.linear_acceleration.y']
    imu_df['raw_linear_acceleration.z'] = imu_df['field.linear_acceleration.z']

    # removes the bias in imu sensor measurements
    bias_acc_x = (imu_df['field.linear_acceleration.x'].mean() + \
                  imu_no_engine_df['field.linear_acceleration.x'].mean()) / 2
    bias_acc_y = (imu_df['field.linear_acceleration.y'].mean() + \
                  imu_no_engine_df['field.linear_acceleration.y'].mean()) / 2
    bias_acc_z = (imu_df['field.linear_acceleration.z'].mean() + \
                  imu_no_engine_df['field.linear_acceleration.z'].mean()) / 2

    imu_df['field.linear_acceleration.x'] -= bias_acc_x
    imu_df['field.linear_acceleration.y'] -= bias_acc_y
    imu_df['field.linear_acceleration.z'] -= bias_acc_z

    imu_df['field.angular_velocity.x'] -= imu_no_engine_df['field.angular_velocity.x'].mean()
    imu_df['field.angular_velocity.y'] -= imu_no_engine_df['field.angular_velocity.y'].mean()
    imu_df['field.angular_velocity.z'] -= imu_no_engine_df['field.angular_velocity.z'].mean()
    return imu_df


def wrap_to_pi(x):
    '''
    Helper function to wrap angle in radians to [−pi pi]
    '''
    xwrap = np.remainder(x, 2*np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2*np.pi * np.sign(xwrap[mask])
    return xwrap


def _plot_acceleration(df):
    sns.set(style="white", palette="muted", color_codes=True)
    # Set up the matplotlib figure
    f, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    sns.despine(left=True)
    # Create scatterplot of dataframe
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.linear_acceleration.x',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[0])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.linear_acceleration.y',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[1])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.linear_acceleration.z',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[2])
    plt.tight_layout()
    axes[2].set(xlabel="Time (sec)")
    f.savefig('figures/part1_linear_acc.pdf')
    plt.show()


def _plot_orientation(df):
    sns.set(style="white", palette="muted", color_codes=True)
    # Set up the matplotlib figure
    f, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    sns.despine(left=True)
    # Create scatterplot of dataframe
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.orientation.x',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[0])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.orientation.y',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[1])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.orientation.z',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[2])
    plt.tight_layout()
    axes[2].set(xlabel="Time (sec)")
    f.savefig('figures/part1_orientation.pdf')
    plt.show()


def _plot_gyro(df):
    sns.set(style="white", palette="muted", color_codes=True)
    # Set up the matplotlib figure
    f, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    sns.despine(left=True)
    # Create scatterplot of dataframe
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.angular_velocity.x',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[0])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.angular_velocity.y',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[1])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.angular_velocity.z',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[2])
    plt.tight_layout()
    axes[2].set(xlabel="Time (s)")
    f.savefig('figures/part1_angular_vel.pdf')
    plt.show()


def _plot_magneto_bytime(df):
    sns.set(style="white", palette="muted", color_codes=True)
    # Set up the matplotlib figure
    f, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    sns.despine(left=True)
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.magnetic_field.x',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[0])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.magnetic_field.y',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[1])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='field.magnetic_field.z',  # Vertical axis
                 data=df,  # Data source
                 ax=axes[2])
    plt.tight_layout()
    axes[2].set(xlabel="Time (s)")
    f.savefig('figures/part1_magneto.pdf')
    plt.show()


def _plot_magneto_2d(df, fname):
    # our motion w.r.t. z-axis is stationary during data acquisition, so we only
    # look at the magnetic field measurements from x and y axes.
    sns.set(style="white", palette="muted", color_codes=True)
    sns.scatterplot(x="field.magnetic_field.x",
                    y="field.magnetic_field.y",
                    data=df,
                    s=10)
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def hard_iron_correction(circular_df, mobile_df):
    '''
    produced by materials that exhibit a constant, additive field to the
    earth's magnetic field, thereby generating a constant additive value to the
    output of each of the magnetometer axes. A speaker magnet, for example,
    will produce a hard-iron distortion.
    '''
    bias_x = (circular_df['field.magnetic_field.x'].max() +
              circular_df['field.magnetic_field.x'].min()) / 2
    bias_y = (circular_df['field.magnetic_field.y'].max() +
              circular_df['field.magnetic_field.y'].min()) / 2

    circular_df['field.magnetic_field.x'] -= bias_x
    circular_df['field.magnetic_field.y'] -= bias_y

    mobile_df['field.magnetic_field.x'] -= bias_x
    mobile_df['field.magnetic_field.y'] -= bias_y

    return circular_df, mobile_df


def soft_iron_correction(df):
    '''
    Unlike hard-iron distortion where the magnetic field is additive to the
    earth's field, soft-iron distortion is the result of material that
    influences, or distorts, a magnetic field—but does not necessarily generate
    a magnetic field itself, and is therefore not additive.
    r: radius of major axis
       to identifying r, calculate the magnitude of each data point and then
       identify the maximum of these computed values
    q: radius of minor axis
       to identifying q, calculate the magnitude of each data point and then
       identify the minimum of these computed values
    theta: angle between major axis and x-axis
    '''
    r_circular = 0
    q_circular = 1e18
    for i in range(len(df)):
        x = math.sqrt(df['field.magnetic_field.x'].iloc[i]**2 +
                      df['field.magnetic_field.y'].iloc[i]**2)
        if x > r_circular:
            r_circular = x
            y1 = df['field.magnetic_field.x'].iloc[i]
        if x < q_circular:
            q_circular = x
    theta = math.sin(y1/r_circular)
    rotation_matrix = np.array([[math.cos(theta), math.sin(theta)],
                                [-math.sin(theta), math.cos(theta)]])

    # Apply rotation matrix:
    def rotate(x, y):
        rot = np.dot(rotation_matrix, np.array([[x], [y]]))
        return pd.Series([rot[0], rot[1]])

    df[['field.magnetic_field.x', 'field.magnetic_field.y']] = \
        df.apply(lambda row: rotate(row['field.magnetic_field.x'], row['field.magnetic_field.y']),
                 axis=1)

    # rescaling (commented because it makes the zero centered circle elliptical):
    scaling_coeff = q_circular/r_circular
    df['field.magnetic_field.x'] /= scaling_coeff
    df['field.magnetic_field.y'] /= scaling_coeff
    return df


def _get_yaw_from_mag(imu_df, mag_df, phase=9*math.pi/32):
    '''
    calculates the yaw angle from the corrected magnetometer readings
    magnetometer measure magnetism, but also helps to find orientation using
    the earth’s magnetic field, similar to a compass
    Accelerometer works on the fact that gravitational force is always constant
    in direction i.e. towards the earth, but in case of yaw, the yaw axis is
    perpendicular to gravitational force, so if we keep roll and pitch same and
    just change the yaw angle we will not be able to measure any difference in
    accelerometer values. Hence accelerometer fails to measure yaw
    '''
    # these equations didn't work right:
    # yaw_from_mag = np.zeros(len(mag_df))
    # for i in range(len(imu_df)):
    #     acc_x = imu_df['field.linear_acceleration.x'].iloc[i]
    #     acc_y = imu_df['field.linear_acceleration.y'].iloc[i]
    #     acc_z = imu_df['field.linear_acceleration.z'].iloc[i]
    #
    #     mag_x = mag_df['field.magnetic_field.x'].iloc[i]
    #     mag_y = mag_df['field.magnetic_field.y'].iloc[i]
    #     mag_z = mag_df['field.magnetic_field.z'].iloc[i]
    #
    #     pitch = math.atan2(acc_x, math.sqrt(acc_y**2 + acc_z**2))
    #     roll = math.atan2(acc_y, math.sqrt(acc_x**2 + acc_z**2))
    #     x = mag_x * math.cos(pitch) + mag_y * math.sin(roll) * \
    #         math.sin(pitch) + mag_z * math.cos(roll) * math.sin(pitch)
    #     y = mag_y * math.cos(roll) - mag_z * math.sin(roll)
    #     yaw_from_mag[i] = math.atan2(-y, x)
    # mag_df['yaw_from_mag'] = wrap_to_pi(yaw_from_mag)

    mag_df['yaw_from_mag'] = \
        mag_df.apply(lambda row: math.atan2(row['field.magnetic_field.x'],
                                            row['field.magnetic_field.y'])
                                 + phase + 88*math.pi/100, axis=1)
    mag_df['yaw_from_mag'] = wrap_to_pi(mag_df['yaw_from_mag'].values)
    return mag_df


def _get_yaw_from_gyro(imu_df, phase=9*math.pi/32):
    '''
    orientation readings of the sensor are in units:
    pitch (+/= 90): x, yaw (+/= 180): y, roll (+/= 180): z
    and angular velocity is in rad/s
    integrates yaw rate of the gyro to return yaw angle
    '''
    t = imu_df['corrected_time'].values
    yaw_angle = integrate.cumtrapz(imu_df['field.angular_velocity.z'], t,
                                   initial=0)
    imu_df['yaw_from_gyro'] = wrap_to_pi(yaw_angle + phase)
    return imu_df


def compare_yaw(imu_df, mag_df):
    '''
    compares the yaw angle calculated with magnetometer and the yaw angle
    calculated with gyro
    '''
    sns.set(style="white", palette="muted", color_codes=True)
    # Set up the matplotlib figure
    f, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    sns.despine(left=True)
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='yaw_from_gyro',  # Vertical axis
                 data=imu_df,  # Data source
                 ax=axes[0])
    sns.lineplot(x='corrected_time',  # Horizontal axis
                 y='yaw_from_mag',  # Vertical axis
                 data=mag_df,  # Data source
                 ax=axes[1])
    plt.tight_layout()
    axes[1].set(xlabel="Time (s)")
    f.savefig('figures/part3_yaw_comparison.pdf')
    plt.show()

    diff = (imu_df['yaw_from_gyro'] - mag_df['yaw_from_mag']).values
    plt.plot(imu_df['corrected_time'].values, diff)
    plt.savefig('figures/part3_yaw_difference.pdf')
    plt.show()


def sensor_fusion(imu_df, mag_df, weight):
    '''
    filter the yaw angle estimate of magnetometer using a low pass filter
    filter the yaw angle estimate of gyro using a high pass filter
    https://engineering.stackexchange.com/questions/3348/calculating-pitch-yaw-and-roll-from-mag-acc-and-gyro-data
    '''
    t = imu_df['corrected_time'].values
    gyro_data = imu_df['yaw_from_gyro']
    mag_data = mag_df['yaw_from_mag'].values
    fused_data = (1-weight) * gyro_data + weight * mag_data
    imu_df['fused_yaw'] = fused_data
    plt.figure()
    plt.plot(t, fused_data)
    plt.xlabel('Time(sec')
    plt.savefig('figures/part1_sensor_fusion.pdf')
    plt.show()
    return imu_df


def estimate_forward_velocity(imu_df, gps_df):
    # integrate the forward acceleration to estimate the forward velocity:
    t_imu = imu_df['corrected_time'].values
    imu_df['v_x_est_by_imu'] = integrate.cumtrapz(imu_df['field.linear_acceleration.x'].values,
                                                  t_imu, initial=0)
    imu_df['v_y_est_by_imu'] = integrate.cumtrapz(imu_df['field.linear_acceleration.y'].values,
                                                  t_imu, initial=0)
    up_shift_x = abs(imu_df['v_x_est_by_imu'].min())
    up_shift_y = abs(imu_df['v_y_est_by_imu'].min())
    imu_df['v_x_est_by_imu'] += up_shift_x
    imu_df['v_y_est_by_imu'] += up_shift_y

    plt.plot(imu_df['field.linear_acceleration.x'])
    plt.title('Linear Acceleration')
    plt.savefig('figures/forward_lin_acc.pdf')
    plt.show()
    # remove dead reckoning:
    row_index1 = imu_df[imu_df.corrected_time > 56].index[0]
    row_index2 = imu_df[imu_df.corrected_time > 156].index[0]
    imu_df['v_x_est_by_imu'].iloc[:row_index1] = 0
    offset = imu_df['v_x_est_by_imu'].iloc[row_index1]
    bias_reckoning = np.linspace(offset, 0, row_index2-row_index1)
    imu_df['v_x_est_by_imu'].iloc[row_index1:row_index2] -= bias_reckoning

    t_gps = gps_df['corrected_time'].values
    gps_df['delta_t'] = gps_df['corrected_time'] - gps_df['corrected_time'].shift(1)
    gps_df['v_est_by_gps_easting'] = gps_df['field.utm_easting'] - \
                                     gps_df['field.utm_easting'].shift(1)
    gps_df['v_est_by_gps_easting'] /= gps_df['delta_t']
    gps_df['v_est_by_gps_northing'] = gps_df['field.utm_northing'] - \
                                      gps_df['field.utm_northing'].shift(1)
    gps_df['v_est_by_gps_northing'] /= gps_df['delta_t']
    gps_df['v_est_by_gps'] = np.sqrt(gps_df['v_est_by_gps_easting']**2 + \
                                     gps_df['v_est_by_gps_northing']**2)

    plt.plot(t_imu, imu_df['v_x_est_by_imu'],
             label='vel est. from IMU')
    plt.plot(t_gps, gps_df['v_est_by_gps'], label='vel est. from GPS')
    plt.legend()
    plt.savefig('figures/vel_estimation.pdf')
    plt.show()
    return imu_df, gps_df


def estimate_displacement(imu_df, gps_df, declination=14.4):
    '''
    Estimate the vehicle trajectory
    delineation is 14.4 degrees
    '''
    declination = 14.4 * math.pi / 180 # magnetic declination in boston
    t_imu = imu_df['corrected_time'].values
    delta_t = 1 / f_imu
    pos_x = np.zeros(len(imu_df))
    pos_y = np.zeros(len(imu_df))
    for i in range(1, len(imu_df['v_x_est_by_imu'])):
        yaw = imu_df['fused_yaw'].iloc[i-1]
        pos_x[i] = pos_x[i-1] + \
                   imu_df['v_x_est_by_imu'].iloc[i-1] * math.cos(yaw+declination) * delta_t
        pos_y[i] = pos_y[i-1] + \
                   imu_df['v_x_est_by_imu'].iloc[i-1] * -math.sin(yaw+declination) * delta_t
    imu_df['pos_x'], imu_df['pos_y'] = pos_x, pos_y

    t_gps = gps_df['corrected_time'].values
    gps_df['pos_x'] = gps_df['field.utm_easting'] - \
                      gps_df['field.utm_easting'].iloc[0]
    gps_df['pos_y'] = gps_df['field.utm_northing'] - \
                      gps_df['field.utm_northing'].iloc[0]

    f, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    axes[0].plot(t_imu, imu_df['pos_x'], label='displacement est. X from IMU')
    axes[0].plot(t_gps, gps_df['pos_x'], label='displacement est. X from GPS')
    axes[1].plot(t_imu, imu_df['pos_y'], label='displacement est. Y from IMU')
    axes[1].plot(t_gps, gps_df['pos_y'], label='displacement est. Y from GPS')
    axes[2].plot(imu_df['pos_x'], imu_df['pos_y'], label='IMU data in 2D')
    axes[2].plot(gps_df['pos_x'], gps_df['pos_y'], label='GPS data in 2D')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.savefig('figures/displ_estimation.pdf')
    plt.show()
    return imu_df, gps_df


def dead_reckoning(imu_df, gps_df):
    '''
    dead reckoning is the process of calculating robot's current position
    by using a previously determined position with the use of IMU, without any
    data from a gps navigation system or wireless signal transmission.
    In particular, it's used in underwater navigation since GPS connection gets
    lost under water. Dead reckoning is also common in autonomous vehicles that
    get activated during a GPS outage.
    '''
    t = imu_df['corrected_time']
    # Obtain X_dot: imu_df['pos_x_est_by_imu']
    # compute wX_dot
    imu_df['wX_dot'] = imu_df['field.angular_velocity.z'] * imu_df['v_x_est_by_imu']
    plt.figure()
    plt.plot(t, imu_df['wX_dot'], label='wX_dot')
    plt.plot(t, imu_df['field.linear_acceleration.y'], label='y_dot_dot')
    plt.legend()
    plt.savefig('figures/wX_dot_vs_y_dotdot.pdf')
    plt.show()

    plt.figure()
    plt.plot(t, imu_df['wX_dot'], label='wX_dot')
    plt.legend()
    plt.savefig('figures/wX_dot.pdf')
    plt.show()
    return imu_df


def center_of_mass(imu_df):
    x_dot = integrate.cumtrapz(imu_df['field.linear_acceleration.x'].values,
                               imu_df['corrected_time'].values, initial=0)
    omega = imu_df['field.angular_velocity.z'].values
    omega_dot = np.zeros(len(omega))
    for i in range(1, len(omega)):
        omega_dot[i] = omega[i] - omega[i-1]
        if omega_dot[i] == 0:
            omega_dot[i] = .000001

    omega_dot_xc = np.multiply(x_dot, omega)

    xc = np.divide(omega_dot_xc[1:], omega_dot[1:])
    # Remove outliers
    for i in range(len(xc)):
        if xc[i] == np.inf:
            xc[i] = 0
        if abs(xc[i]) > 1000000:
            xc[i] = np.sign(xc[i]) * 1000000

    print(np.mean(xc))

    plt.plot(imu_df['corrected_time'][1:], xc)
    plt.xlabel('Time (s)')
    plt.ylabel('Center of mass (cm)')
    plt.show()

def main():
    # parse IMU and GPS data from csv files
    stationary_imu, stationary_mag, imu_mobile, imu_no_engine, \
        mag_mobile, mag_no_engine, gps_mobile, gps_no_engine, \
        gps_circular, mag_circular = parser_imu()

    # remove the bias in IMU readings:
    stationary_imu = remove_bias(stationary_imu, imu_no_engine)
    imu_mobile = remove_bias(imu_mobile, imu_no_engine)

    # plot linear acceleration data
    _plot_acceleration(stationary_imu)
    # plot angular velocity data
    _plot_gyro(stationary_imu)
    # plot quaternion vector
    _plot_orientation(stationary_imu)
    # plot magnetometer data
    _plot_magneto_bytime(stationary_mag)

    print('Uncalibrated magnetometer data')
    _plot_magneto_2d(stationary_mag,
                     fname='figures/part1_magneto_2D.pdf')
    # plot the uncalibrated measurements from magnetometer (car data)
    _plot_magneto_2d(mag_circular,
                     fname='figures/part3_magneto_2D_circular.pdf')
    _plot_magneto_2d(mag_mobile,
                     fname='figures/part3_magneto_2D_mobile.pdf')

    # hard and soft iron corrections for magnetometer calibration:
    print('hard iron corrections')
    mag_circular, mag_mobile = hard_iron_correction(mag_circular, mag_mobile)

    print('soft iron corrections:')
    mag_circular = soft_iron_correction(mag_circular)
    mag_mobile = soft_iron_correction(mag_mobile)
    _plot_magneto_2d(mag_mobile,
                     fname='figures/part3_magneto_2D_corrected_mobile.pdf')
    _plot_magneto_2d(mag_circular,
                     fname='figures/part3_magneto_2D_corrected_circular.pdf')
    # calculate yaw angle from magnetometer readings
    mag_mobile = _get_yaw_from_mag(imu_mobile, mag_mobile)
    # calculates yaw angle by integrating yaw rates of gyro:
    imu_mobile = _get_yaw_from_gyro(imu_mobile)
    compare_yaw(imu_mobile, mag_mobile)
    imu_mobile = sensor_fusion(imu_mobile, mag_mobile, weight=0.01)
    imu_mobile, gps_mobile = estimate_forward_velocity(imu_mobile, gps_mobile)
    estimate_displacement(imu_mobile, gps_mobile)
    imu_mobile = dead_reckoning(imu_mobile, gps_mobile)
    center_of_mass(imu_mobile)


if __name__ == '__main__':
    main()

