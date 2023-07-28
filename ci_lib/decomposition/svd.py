import numpy as np
import wfield

def blockwise_svd(pixel_data, n_components, mask=None, logger=None, **kwargs):
    """
    Computes a truncated blockwise SVD and returns spatial and singular value scaled temporal components

    :param pixel_data: some object supporting pixel_data.shape (T, X, Y)/(T, C, X, Y)
    and __getitem__ slicing in those dimensions. E.g. DecompData.pixel

    :param n_components: Number of components the SVD gets truncated to.
    :type n_components: int

    :param n_components: Mask indicating which pixels should be considered;
    all other pixel will be set zero; default is None
    :type n_components: np.ndarray or None

    :return: (Vc, U), where Vc (T x N) are the temporal and U (N x X x Y)/(N, C, X, N) the spatial components
    :rtype: tuple(np.ndarray, np.ndarray)
    """

    add_channel_dim = len(pixel_data.shape) < 4
    if add_channel_dim:
        class add_channel:
            def __init__(self, pixel):
                self._pixel = pixel
            def __getitem__(self, keys):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                return self._pixel.__getitem__((keys[0], *keys[2:]))
            @property
            def shape(self):
                shape = self._pixel.shape
                return (shape[0], 1, *shape[1:])
        # overwrite __getitem__ and shape, so that it ignores the channel dimension
        pixel_data = add_channel(pixel_data)

    if mask is not None:
        pixel_data[:,:,np.logical_not(mask)] = 0

    t, c, x, y = pixel_data.shape

    # assume avrg ist already zero, because recalculation is expensive
    # TODO check
    avrg = np.zeros((c,x,y), dtype=float)

    u, svt, s, _ = wfield.decomposition.svd_blockwise(pixel_data, avrg, k=n_components, divide_by_average=False, **kwargs)

    logger.debug(f"{u.shape=}")
    # u.shape = (X * Y) x N - > spatials.shape = N x X x Y
    spatials = np.moveaxis(u,-1,0).reshape(-1, x, y)
    logger.debug(f"{spatials.shape=}")

    logger.debug(f"{svt.shape=}")
    if add_channel_dim:
        # svt.shape = N x (T * C) - > temporals.shape = T x N
        temporals = svt.T
    else:
        # svt.shape = N x (T * C) - > temporals.shape = C x T x N
        temporals = np.moveaxis(svt.reshape(-1,t,c), (0,1,2), (2,1,0))
    logger.debug(f"{temporals.shape=}")

    # s.shape = N
    logger.debug(f"{s.shape=}")
    # s is already sorted and multiplied into svt/temporals
    logger.debug(f"{s=}")

    return temporals, spatials, s

def svd(data, n_components=None, mask=None):
    """
    Recomputes an SVD from a DecompData object; uses `blockwise_svd`

    :param data: Some DecompData object
    :type data: DecompData

    :param n_components: Number of components.
    :type n_components: int or None

    :return: SVD DecompData object
    :rtype: DecompData
    """
    if n_components is None:
        n_components = data.n_components
    Vc, U = blockwise_svd(data.pixel, n_components, mask)
    return data.recreate(Vc, U)

def postprocess_SVD(data, n_components):
    """
    Modifies an SVD DecompData object, for example to crop to the number of components

    :param data: DecompData object with SVD parcellation
    :type data: DecompData

    :param n_components: Number of components (w.r.t. time).
    :type n_components: int

    :return: DecompData object containing the processes SVD
    :rtype: DecompData
    """

    return data.recreate(data.temporals_flat[:,:n_components], data.spatials[:n_components,:,:])
