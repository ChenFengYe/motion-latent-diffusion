from .frames import get_frameix_from_data_index

class FrameSampler:
    def __init__(self, sampling="conseq", sampling_step=1, request_frames=None,threshold_reject=0.75,max_len=1000,min_len=10):
        self.sampling  = sampling

        self.sampling_step = sampling_step
        self.request_frames = request_frames
        self.threshold_reject = threshold_reject
        self.max_len = max_len
        self.min_len = min_len

    def __call__(self, num_frames):

        return get_frameix_from_data_index(num_frames,
                                           self.request_frames,
                                           self.sampling,
                                           self.sampling_step)

    def accept(self, duration):
        # Outputs have original lengths
        # Check if it is too long
        if self.request_frames is None:
            if duration > self.max_len:
                return False
            elif duration < self.min_len:
                return False
        else:
            # Reject sample if the length is
            # too little relative to
            # the request frames
            min_number = self.threshold_reject * self.request_frames
            if duration < min_number:
                return False
        return True

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)
