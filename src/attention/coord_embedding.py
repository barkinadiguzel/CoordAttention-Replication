from pooling.height_pool import HeightPool
from pooling.width_pool import WidthPool

class CoordEmbedding:
    def __init__(self):
        self.h_pool = HeightPool()
        self.w_pool = WidthPool()

    def __call__(self, x):
        z_h = self.h_pool(x)
        z_w = self.w_pool(x)
        return z_h, z_w
