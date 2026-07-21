from skimage import exposure
from skimage.filters import sobel,rank,gaussian
import numpy as np

def _auto_gamma(image,
                target=0.05,
                p_low=1,
                p_high=99.999,
                gamma_min=0.8,
                gamma_max=1.3):
    """
    自动估计 gamma，使用 99% 分位数代表图像亮度（适合神经元图像）
    image: 已归一化到 [0, 1] 的图像
    """
    eps = 1e-6
    img = image.astype(np.float32, copy=False)

    # 鲁棒亮度估计：先做百分位裁剪，减少异常亮点影响
    v_low, v_high = np.percentile(img, [p_low, p_high])
    img_clip = np.clip(img, v_low, v_high)
    img_clip = (img_clip - v_low) / (v_high - v_low + eps)

    # 使用 99% 分位数作为亮度指标（前景主导）
    m = float(np.percentile(img_clip, 95))

    m = np.clip(m, eps, 1.0 - eps)
    target = np.clip(float(target), eps, 1.0 - eps)
    print("m: ",m, target)
    gamma = np.log(target) / np.log(m)
    gamma = float(np.clip(gamma, gamma_min, gamma_max))

    out = exposure.adjust_gamma(img, gamma)
    print("out: ",out.max())
    out = np.clip(out,0,1.0)
    return out, gamma

def image_preprocessing(config,image):
    EPS = 1e-5
    # step 1: 初始归一化
    image = image.astype(np.float32, copy=False)
    image = image / (image.max() + EPS)

    # step 2: Gaussian smoothing
    if config.get("gaussian_use", False):
        sigma = config.get("gaussian_sigma", 0.5)
        image = gaussian(image, sigma=sigma)

    # step 3: CLAHE
    if config.get('eqa_use', False):
        kernel_size = config.get('eqa_kernel_size', (7, 7, 7))
        image = exposure.equalize_adapthist(image, kernel_size=kernel_size)

    # step 4: Gamma correction
    if config.get("auto_gamma_use", False):
        image, gamma_val = _auto_gamma(
            image,
            target=config.get("auto_gamma_target", 0.05),
            p_low=config.get("auto_gamma_p_low", 1),
            p_high=config.get("auto_gamma_p_high", 99.999),
            gamma_min=config.get("auto_gamma_min", 0.8),
            gamma_max=config.get("auto_gamma_max", 1.3)
        )
        print(f"[preprocess] auto gamma = {gamma_val:.4f}")
        print(image.max())
    else:
        gamma_val = config.get("gamma", 1.0)
        if gamma_val != 1.0:
            image = exposure.adjust_gamma(image, gamma_val)
        image = np.clip(image, 0.0, 1.0)
        print(f"[preprocess] fixed gamma = {gamma_val:.4f}")

    # step 4: Normalization
    image_min = image.min()
    image_max = image.max()
    print('[preprocess] image min = ',image_min," max = ", image_max)
    if config.get('norm_percent_use', False):
        v_low, v_high = np.percentile(image, [0.001, 99.999])
        image = np.clip(image, v_low, v_high)
        image = (image - v_low) / (v_high - v_low + EPS)
        print(f"[preprocess] percentile norm: v_low={v_low:.4f}, v_high={v_high:.4f}")
    else:
        image = (image - image_min) / (image_max - image_min + EPS)

    image = image.astype(np.float32, copy=False)
    print(f"[preprocess] done: max={image.max():.4f}, min={image.min():.4f}")
    return image