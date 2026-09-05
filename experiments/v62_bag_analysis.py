#!/usr/bin/env python3
"""Extract and fit the V6.2 command/odometry response from a ROS 2 bag."""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _read_bag(uri):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: item.type
             for item in reader.get_all_topics_and_types()}
    required = {"/cmd_vel", "/odom"}
    missing = required - set(types)
    if missing:
        raise RuntimeError(f"bag is missing required topics: {sorted(missing)}")
    message_types = {name: get_message(types[name]) for name in required}
    commands, odometry = [], []
    while reader.has_next():
        topic, raw, timestamp = reader.read_next()
        if topic not in required:
            continue
        message = deserialize_message(raw, message_types[topic])
        time_s = timestamp * 1e-9
        if topic == "/cmd_vel":
            commands.append((time_s, message.linear.x, message.angular.z))
        else:
            odometry.append((
                time_s, message.twist.twist.linear.x,
                message.twist.twist.angular.z,
                message.pose.pose.position.x, message.pose.pose.position.y))
    return np.asarray(commands, dtype=float), np.asarray(odometry, dtype=float)


def _zero_order(times, values, query):
    indices = np.searchsorted(times, query, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1)
    return values[indices]


def _estimate_delay(time_s, command, measured, active_threshold):
    active = np.abs(command) >= active_threshold
    if active.sum() < 20:
        return np.nan
    scale = max(float(np.std(measured[active])), 1e-6)
    candidates = np.arange(0.0, 0.501, 0.01)
    errors = []
    for delay in candidates:
        shifted = np.interp(time_s - delay, time_s, command,
                            left=command[0], right=command[-1])
        errors.append(np.mean(((measured[active] - shifted[active]) / scale) ** 2))
    return float(candidates[int(np.argmin(errors))])


def _steady_mask(time_s, command_v, command_w, settle_time=0.6):
    changed = np.concatenate(([True],
        (np.abs(np.diff(command_v)) > 1e-6)
        | (np.abs(np.diff(command_w)) > 1e-6)))
    change_times = time_s[changed]
    previous = np.searchsorted(change_times, time_s, side="right") - 1
    age = time_s - change_times[np.clip(previous, 0, len(change_times) - 1)]
    return age >= settle_time


def _gain(command, measured, mask):
    usable = mask & (np.abs(command) > 0.05)
    if usable.sum() < 20:
        return np.nan, np.nan
    ratio = measured[usable] / command[usable]
    return float(np.median(ratio)), float(np.median(np.abs(
        measured[usable] - command[usable])))


def _stopping_distances(time_s, command_v, measured_v):
    transitions = np.flatnonzero(
        (np.abs(command_v[:-1]) > 0.1) & (np.abs(command_v[1:]) <= 0.02)) + 1
    distances = []
    dt = float(np.median(np.diff(time_s)))
    for start in transitions:
        stop = min(start + int(round(3.0 / dt)), len(time_s) - 1)
        quiet = np.abs(measured_v[start:stop]) < 0.03
        candidates = np.flatnonzero(np.convolve(
            quiet.astype(int), np.ones(max(1, int(round(0.2 / dt))), dtype=int),
            mode="same") >= max(1, int(round(0.2 / dt))))
        if len(candidates):
            stop = start + int(candidates[0])
        distances.append(float(np.trapezoid(
            np.abs(measured_v[start:stop + 1]), time_s[start:stop + 1])))
    return distances


def analyze(commands, odometry, rate_hz=50.0):
    start = max(commands[0, 0], odometry[0, 0])
    end = min(commands[-1, 0], odometry[-1, 0])
    time_s = np.arange(start, end, 1.0 / rate_hz)
    cmd_v = _zero_order(commands[:, 0], commands[:, 1], time_s)
    cmd_w = _zero_order(commands[:, 0], commands[:, 2], time_s)
    odom_v = np.interp(time_s, odometry[:, 0], odometry[:, 1])
    odom_w = np.interp(time_s, odometry[:, 0], odometry[:, 2])
    odom_x = np.interp(time_s, odometry[:, 0], odometry[:, 3])
    odom_y = np.interp(time_s, odometry[:, 0], odometry[:, 4])
    steady = _steady_mask(time_s, cmd_v, cmd_w)
    linear_mask = steady & (np.abs(cmd_w) < 0.05)
    rotation_mask = steady & (np.abs(cmd_v) < 0.05)
    linear_gain, linear_mae = _gain(cmd_v, odom_v, linear_mask)
    yaw_gain, yaw_mae = _gain(cmd_w, odom_w, rotation_mask)
    window = min(51, len(time_s) // 2 * 2 - 1)
    smooth_v = savgol_filter(odom_v, window, 3) if window >= 5 else odom_v
    smooth_w = savgol_filter(odom_w, window, 3) if window >= 5 else odom_w
    acceleration = np.gradient(smooth_v, time_s)
    angular_acceleration = np.gradient(smooth_w, time_s)
    stops = _stopping_distances(time_s, cmd_v, odom_v)
    simultaneous = steady & (np.abs(cmd_v) > 0.05) & (np.abs(cmd_w) > 0.05)
    report = {
        "duration_s": float(end - start),
        "command_messages": int(len(commands)),
        "odometry_messages": int(len(odometry)),
        "analysis_rate_hz": rate_hz,
        "linear_delay_s": _estimate_delay(time_s, cmd_v, odom_v, 0.05),
        "angular_delay_s": _estimate_delay(time_s, cmd_w, odom_w, 0.10),
        "linear_gain": linear_gain,
        "linear_steady_mae_mps": linear_mae,
        "in_place_yaw_gain": yaw_gain,
        "angular_steady_mae_radps": yaw_mae,
        "linear_accel_p95_mps2": float(np.percentile(np.abs(acceleration), 95)),
        "angular_accel_p95_radps2": float(np.percentile(
            np.abs(angular_acceleration), 95)),
        "stop_events": len(stops),
        "median_stopping_distance_m": (float(np.median(stops))
                                        if stops else np.nan),
        "simultaneous_turn_samples": int(simultaneous.sum()),
        "speed_yaw_loss_identifiable": bool(simultaneous.sum() >= 50),
    }
    series = np.column_stack((time_s - start, cmd_v, odom_v, cmd_w, odom_w,
                              odom_x, odom_y))
    return report, series


def _write_outputs(output, report, series):
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "v62_report.json"), "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=True)
    with open(os.path.join(output, "v62_timeseries.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "cmd_v_mps", "odom_v_mps", "cmd_w_radps",
                         "odom_w_radps", "odom_x_m", "odom_y_m"))
        writer.writerows(series)
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(series[:, 0], series[:, 1], label="command", linewidth=1.5)
    axes[0].plot(series[:, 0], series[:, 2], label="odometry", linewidth=1.0)
    axes[0].set_ylabel("linear velocity [m/s]")
    axes[1].plot(series[:, 0], series[:, 3], label="command", linewidth=1.5)
    axes[1].plot(series[:, 0], series[:, 4], label="odometry", linewidth=1.0)
    axes[1].set_ylabel("yaw rate [rad/s]")
    axes[2].plot(series[:, 5], series[:, 6])
    axes[2].set_xlabel("odom x [m]")
    axes[2].set_ylabel("odom y [m]")
    axes[2].axis("equal")
    for axis in axes[:2]:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("V6.2 Gazebo command/odometry calibration")
    figure.tight_layout()
    figure.savefig(os.path.join(output, "v62_response.png"), dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--out", default="artifacts/v62/manual_rep01")
    args = parser.parse_args()
    commands, odometry = _read_bag(os.path.abspath(args.bag))
    if not len(commands) or not len(odometry):
        parser.error("bag contains no usable /cmd_vel or /odom messages")
    report, series = analyze(commands, odometry, rate_hz=args.rate)
    _write_outputs(os.path.abspath(args.out), report, series)
    print(json.dumps(report, indent=2, allow_nan=True))
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
