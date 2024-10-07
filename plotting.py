"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""Plotting functions."""

import numpy as np

def calculate_mean_and_std(sequence, window_size):
    """Calculate mean and standard deviation of a sequence."""

    num_windows = len(sequence) - window_size + 1
    means = []
    stds = []

    for i in range(num_windows):
        window = sequence[i : i + window_size]
        num_events = sum(window)
        mean_rate = num_events / window_size
        std = np.std(window)
        means.append(mean_rate)
        stds.append(std)

    return np.array(means), np.array(stds)


def bots_mean_and_std(results_dicts, window_size):
    """Calculate mean and standard deviation of set-response bot results."""

    moving_average_results = {}

    result_mapping = {
        "REJECTED": 0,
        "INVALID OFFER": 0,
        "ACCEPTED": 1,
        False: 0,
        True: 1,
        0: 0,
        1: 1,
    }

    for bot_results in results_dicts:
        int_result_list = []
        for item in bot_results["result history"]:
            int_result_list.append(result_mapping[item])
        moving_average_results[bot_results["bot name"]] = calculate_mean_and_std(
            int_result_list, window_size
        )

    return moving_average_results
