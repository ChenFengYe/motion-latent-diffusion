import numpy as np


# TODO: use a real subsampler..
def subsample(num_frames, last_framerate, new_framerate):
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames


# TODO: use a real upsampler..
def upsample(motion, last_framerate, new_framerate):
    step = int(new_framerate / last_framerate)
    assert step >= 1

    # Alpha blending => interpolation
    alpha = np.linspace(0, 1, step+1)
    last = np.einsum("l,...->l...", 1-alpha, motion[:-1])
    new = np.einsum("l,...->l...", alpha, motion[1:])

    chuncks = (last + new)[:-1]
    output = np.concatenate(chuncks.swapaxes(1, 0))
    # Don't forget the last one
    output = np.concatenate((output, motion[[-1]]))
    return output


if __name__ == "__main__":
    motion = np.arange(105)
    submotion = motion[subsample(len(motion), 100.0, 12.5)]
    newmotion = upsample(submotion, 12.5, 100)

    print(newmotion)
