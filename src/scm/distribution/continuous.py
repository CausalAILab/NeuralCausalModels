import torch as T

from .distribution import Distribution


class UniformDistribution(Distribution):
    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device
        return dict(zip(self.u, T.rand(len(self.u), n, 1, device=device)))
