def variance_reward(prediction, target):
    var_open = prediction[:, 0, :].var(unbiased=False)
    var_close = prediction[:, 1, :].var(unbiased=False)
    var_low = prediction[:, 2, :].var(unbiased=False)
    var_high = prediction[:, 3, :].var(unbiased=False)

    variance_reward = -(var_open + var_close + var_low + var_high)
    return variance_reward