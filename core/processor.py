from __future__ import annotations

import cv2
import numpy as np


def optimize_preview_bytes(
    raw_bytes: bytes,
    *,
    large_image_threshold_bytes: int = 1024 * 1024,
    max_preview_side: int = 1600,
    webp_quality: int = 80,
) -> bytes:
    if not raw_bytes:
        raise ValueError("Empty image payload")

    if len(raw_bytes) <= large_image_threshold_bytes:
        return raw_bytes

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image decode failed")

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > max_preview_side:
        scale = max_preview_side / float(longest_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    webp_quality = int(np.clip(webp_quality, 30, 95))
    ok, encoded = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, webp_quality])
    if not ok:
        raise ValueError("Preview optimize encode failed")

    return encoded.tobytes()


def upload_image_to_gpu(raw_bytes: bytes) -> cv2.cuda.GpuMat:
    if not raw_bytes:
        raise ValueError("Empty image payload")

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image decode failed")

    gpu_src = cv2.cuda_GpuMat()
    gpu_src.upload(image)
    return gpu_src


def process_frame_cuda(gpu_src: cv2.cuda.GpuMat, params: dict) -> bytes:
    if gpu_src is None:
        raise ValueError("gpu_src is required")

    try:
        if gpu_src.empty():
            raise ValueError("gpu_src is empty")
    except Exception as exc:
        raise ValueError("gpu_src is invalid") from exc

    gaussian_blur = float(params.get("gaussian_blur", 0) or 0)
    median_blur = float(params.get("median_blur", 0) or 0)
    bilateral_blur = float(params.get("bilateral_blur", 0) or 0)
    sharpen = float(params.get("sharpen", 0) or 0)
    enhance = float(params.get("enhance", 0) or 0)
    denoise = float(params.get("denoise", 0) or 0)
    brightness = float(params.get("brightness", 0) or 0)
    grayscale = float(params.get("grayscale", 0) or 0)
    webp_quality = int(params.get("webp_quality", 80) or 80)
    contrast_bias = float(params.get("contrast_bias", 0) or 0)
    threshold = float(params.get("threshold", 0) or 0)
    log_transform = float(params.get("log_transform", 0) or 0)
    power_law = float(params.get("power_law", params.get("gamma", 1.0)) or 1.0)
    histogram_r = float(params.get("histogram.r", 0) or 0)
    histogram_g = float(params.get("histogram.g", 0) or 0)
    histogram_b = float(params.get("histogram.b", 0) or 0)
    histogram_a = float(
        params.get("histogram.a", params.get("histogram_a", 1.0)) or 1.0
    )
    histogram_a = float(np.clip(histogram_a, 0.0, 1.0))

    try:
        gpu_out = gpu_src.clone()
    except Exception:
        gpu_out = gpu_src

    try:
        if gpu_out.empty():
            raise ValueError("gpu_out is empty before processing")
    except Exception as exc:
        raise ValueError("gpu_out is invalid before processing") from exc

    # --- Denoise (CPU only, applied first) ---
    if denoise > 0:
        cpu_dn = gpu_out.download()
        h_val = float(np.clip(3.0 + denoise * 3.4, 3, 30))
        cpu_dn = cv2.fastNlMeansDenoisingColored(cpu_dn, None, h_val, h_val, 7, 21)
        gpu_out = cv2.cuda_GpuMat()
        gpu_out.upload(cpu_dn)

    # --- Blur (GPU Gaussian) ---
    if gaussian_blur > 0:
        kernel_size = max(3, int(round(3 + gaussian_blur * 2)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blur_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3,
            cv2.CV_8UC3,
            (kernel_size, kernel_size),
            0,
        )
        gpu_out = blur_filter.apply(gpu_out)

    # --- Median Blur (CPU) ---
    if median_blur > 0:
        cpu_mb = gpu_out.download()
        ksize = max(3, int(round(3 + median_blur * 2)))
        if ksize % 2 == 0:
            ksize += 1
        cpu_mb = cv2.medianBlur(cpu_mb, ksize)
        gpu_out = cv2.cuda_GpuMat()
        gpu_out.upload(cpu_mb)

    # --- Bilateral Blur (CPU) ---
    if bilateral_blur > 0:
        cpu_bl = gpu_out.download()
        d = max(5, int(round(5 + bilateral_blur * 4)))
        sigma = 10.0 + bilateral_blur * 15.0
        cpu_bl = cv2.bilateralFilter(cpu_bl, d, sigma, sigma)
        gpu_out = cv2.cuda_GpuMat()
        gpu_out.upload(cpu_bl)

    # --- Sharpen (GPU linear filter, requires BGRA) ---
    if sharpen > 0:
        sharpen_strength = float(np.clip(sharpen / 5.0, 0.0, 1.0))
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        try:
            gpu_base = gpu_out.clone()
            gpu_bgra = cv2.cuda.cvtColor(gpu_out, cv2.COLOR_BGR2BGRA)
            sharpen_filter = cv2.cuda.createLinearFilter(
                cv2.CV_8UC4, cv2.CV_8UC4, kernel
            )
            gpu_bgra = sharpen_filter.apply(gpu_bgra)
            gpu_sharp = cv2.cuda.cvtColor(gpu_bgra, cv2.COLOR_BGRA2BGR)
            gpu_out = cv2.cuda.addWeighted(
                gpu_sharp, sharpen_strength, gpu_base, 1.0 - sharpen_strength, 0
            )
        except Exception:
            cpu_base = gpu_out.download()
            cpu_sh = gpu_out.download()
            cpu_sh = cv2.filter2D(cpu_sh, -1, kernel)
            cpu_sh = cv2.addWeighted(
                cpu_sh, sharpen_strength, cpu_base, 1.0 - sharpen_strength, 0
            )
            gpu_out = cv2.cuda_GpuMat()
            gpu_out.upload(cpu_sh)

    # --- Enhance / CLAHE (GPU CLAHE on L channel of Lab) ---
    if enhance > 0:
        try:
            gpu_lab = cv2.cuda.cvtColor(gpu_out, cv2.COLOR_BGR2Lab)
            cpu_lab = gpu_lab.download()
            l_ch, a_ch, b_ch = cv2.split(cpu_lab)
            gpu_l = cv2.cuda_GpuMat()
            gpu_l.upload(l_ch)
            clip_limit = 1.0 + enhance * 0.6
            clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            gpu_l_eq = clahe.apply(gpu_l, cv2.cuda.Stream_Null())
            cpu_lab_eq = cv2.merge([gpu_l_eq.download(), a_ch, b_ch])
            gpu_lab_eq = cv2.cuda_GpuMat()
            gpu_lab_eq.upload(cpu_lab_eq)
            gpu_out = cv2.cuda.cvtColor(gpu_lab_eq, cv2.COLOR_Lab2BGR)
        except Exception:
            cpu_en = gpu_out.download()
            cpu_lab_en = cv2.cvtColor(cpu_en, cv2.COLOR_BGR2Lab)
            l_en, a_en, b_en = cv2.split(cpu_lab_en)
            clip_limit = 1.0 + enhance * 0.6
            clahe_cpu = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_en = clahe_cpu.apply(l_en)
            cpu_en = cv2.cvtColor(cv2.merge([l_en, a_en, b_en]), cv2.COLOR_Lab2BGR)
            gpu_out = cv2.cuda_GpuMat()
            gpu_out.upload(cpu_en)

    # --- Brightness / Contrast ---
    brightness_offset = float(np.clip(brightness, -1.0, 1.0)) * 255.0
    contrast_alpha = float(np.clip(1.0 + contrast_bias, 0.2, 3.0))

    # --- Grayscale blend ---
    if grayscale > 0:
        try:
            gpu_gray = cv2.cuda.cvtColor(gpu_out, cv2.COLOR_BGR2GRAY)
            gpu_gray_bgr = cv2.cuda.cvtColor(gpu_gray, cv2.COLOR_GRAY2BGR)
            gpu_out = cv2.cuda.addWeighted(
                gpu_gray_bgr, grayscale, gpu_out, 1.0 - grayscale, 0
            )
        except Exception:
            cpu_gs = gpu_out.download()
            gray_gs = cv2.cvtColor(cpu_gs, cv2.COLOR_BGR2GRAY)
            gray_bgr_gs = cv2.cvtColor(gray_gs, cv2.COLOR_GRAY2BGR)
            cpu_gs = cv2.addWeighted(gray_bgr_gs, grayscale, cpu_gs, 1.0 - grayscale, 0)
            gpu_out = cv2.cuda_GpuMat()
            gpu_out.upload(cpu_gs)

    cpu_out = gpu_out.download()
    if cpu_out is None or cpu_out.size == 0:
        raise ValueError("Downloaded preview frame is empty")

    if abs(brightness_offset) > 1e-6 or abs(contrast_alpha - 1.0) > 1e-6:
        cpu_float = cpu_out.astype(np.float32)
        cpu_out = np.clip(
            (cpu_float - 127.5) * contrast_alpha + 127.5 + brightness_offset,
            0,
            255,
        ).astype(np.uint8)

    # --- Logarithmic transform (CPU) ---
    if log_transform > 0:
        log_strength = float(np.clip(log_transform, 0.0, 1.0))
        cpu_float = cpu_out.astype(np.float32)
        c = 255.0 / np.log1p(255.0)
        cpu_log = np.clip(c * np.log1p(cpu_float), 0, 255).astype(np.uint8)
        if log_strength >= 1.0:
            cpu_out = cpu_log
        else:
            cpu_out = cv2.addWeighted(
                cpu_out, 1.0 - log_strength, cpu_log, log_strength, 0
            )

    # --- Power-law / Gamma transform (CPU) ---
    if abs(power_law - 1.0) > 1e-6:
        gamma = max(1e-6, power_law)
        cpu_norm = cpu_out.astype(np.float32) / 255.0
        cpu_out = np.clip(np.power(cpu_norm, gamma) * 255.0, 0, 255).astype(np.uint8)

    # --- Threshold (CPU, binary) ---
    if threshold > 0:
        threshold_value = threshold * 255.0 if threshold <= 1.0 else threshold
        threshold_value = float(np.clip(threshold_value, 0.0, 255.0))
        gray = cv2.cvtColor(cpu_out, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        cpu_out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # --- Per-channel histogram adjustment (R, G, B) with alpha strength ---
    if abs(histogram_r) > 1e-6 or abs(histogram_g) > 1e-6 or abs(histogram_b) > 1e-6:
        b_ch, g_ch, r_ch = cv2.split(cpu_out.astype(np.float32))
        r_ch = np.clip(r_ch + histogram_r * 255.0 * histogram_a, 0, 255)
        g_ch = np.clip(g_ch + histogram_g * 255.0 * histogram_a, 0, 255)
        b_ch = np.clip(b_ch + histogram_b * 255.0 * histogram_a, 0, 255)
        cpu_out = cv2.merge([b_ch, g_ch, r_ch]).astype(np.uint8)

    webp_quality = max(1, min(100, webp_quality))
    ok, encoded = cv2.imencode(
        ".webp",
        cpu_out,
        [cv2.IMWRITE_WEBP_QUALITY, webp_quality],
    )
    if not ok:
        raise ValueError("WEBP encode failed")

    return encoded.tobytes()
