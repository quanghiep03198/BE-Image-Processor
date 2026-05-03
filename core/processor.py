from __future__ import annotations

import cv2
import numpy as np


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

    blur = float(params.get("blur", 0) or 0)
    sharpen = float(params.get("sharpen", 0) or 0)
    enhance = float(params.get("enhance", 0) or 0)
    denoise = float(params.get("denoise", 0) or 0)
    brightness = float(params.get("brightness", 0) or 0)
    grayscale = float(params.get("grayscale", 0) or 0)
    jpeg_quality = int(params.get("jpeg_quality", 80) or 80)
    contrast_bias = float(params.get("contrast_bias", 0) or 0)

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
    if blur > 0:
        kernel_size = max(3, int(round(3 + blur * 2)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blur_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3,
            cv2.CV_8UC3,
            (kernel_size, kernel_size),
            0,
        )
        gpu_out = blur_filter.apply(gpu_out)

    # --- Sharpen (GPU linear filter, requires BGRA) ---
    if sharpen > 0:
        center = float(np.clip(9.0 + sharpen, 9.0, 14.0))
        kernel = np.array([[0, -1, 0], [-1, center, -1], [0, -1, 0]], dtype=np.float32)
        try:
            gpu_bgra = cv2.cuda.cvtColor(gpu_out, cv2.COLOR_BGR2BGRA)
            sharpen_filter = cv2.cuda.createLinearFilter(
                cv2.CV_8UC4, cv2.CV_8UC4, kernel
            )
            gpu_bgra = sharpen_filter.apply(gpu_bgra)
            gpu_out = cv2.cuda.cvtColor(gpu_bgra, cv2.COLOR_BGRA2BGR)
        except Exception:
            cpu_sh = gpu_out.download()
            cpu_sh = cv2.filter2D(cpu_sh, -1, kernel)
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
    alpha = 1.0 + brightness
    beta = contrast_bias
    apply_cpu_brightness = False
    if abs(alpha - 1.0) > 1e-6 or abs(beta) > 1e-6:
        try:
            gpu_out = cv2.cuda.addWeighted(gpu_out, alpha, gpu_out, 0.0, beta)
            if gpu_out.empty():
                apply_cpu_brightness = True
        except Exception:
            apply_cpu_brightness = True

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

    if apply_cpu_brightness:
        cpu_float = cpu_out.astype(np.float32)
        cpu_out = np.clip(cpu_float * alpha + beta, 0, 255).astype(np.uint8)

    jpeg_quality = max(1, min(100, jpeg_quality))
    ok, encoded = cv2.imencode(
        ".jpg",
        cpu_out,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise ValueError("JPEG encode failed")

    return encoded.tobytes()
