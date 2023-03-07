
def piecewise_decay(now_step, anchor, anchor_value):
    """piecewise linear decay scheduler

    Parameters
    ---------
    now_step : int
        current step
    anchor : list of integer
        step anchor
    anchor_value: list of float
        value at corresponding anchor
    """
    i = 0
    while i < len(anchor) and now_step >= anchor[i]:
        i += 1

    if i == len(anchor):
        return anchor_value[-1]
    else:
        return anchor_value[i-1] + (now_step - anchor[i-1]) * \
                                   ((anchor_value[i] - anchor_value[i-1]) / (anchor[i] - anchor[i-1]))

if __name__ == "__main__":
    for k in range(100000):
        eps = piecewise_decay(k, [0, 7000, 14000], [1, 0.2, 0.05])
        if k % 100 == 0:
            print(eps)