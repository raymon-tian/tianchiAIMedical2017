* 关于预处理，一定检查 CT 的 H 和 W是否相等，存在bug，否则会出现error
step1.py 165行
current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
nan_mask的尺寸可能与image[i]的尺寸不一致
* 图像的索引方式为  (N,D,H,W)