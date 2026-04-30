# Hướng Dẫn Sử Dụng OpenCV API - Image Processing

Tài liệu mô tả các hàm OpenCV được dùng trong backend xử lý ảnh, bao gồm giải thích tham số, cách hoạt động, và ý nghĩa thực tế.

---

## Mục Lục

1. [Đọc ảnh từ bytes](#1-đọc-ảnh-từ-bytes)
2. [Gaussian Blur — Làm mờ ảnh](#2-gaussian-blur--làm-mờ-ảnh)
3. [Sharpen — Làm sắc nét](#3-sharpen--làm-sắc-nét)
4. [CLAHE Enhance — Tăng tương phản](#4-clahe-enhance--tăng-tương-phản)
5. [Denoise — Khử nhiễu](#5-denoise--khử-nhiễu)
6. [Brightness — Điều chỉnh độ sáng](#6-brightness--điều-chỉnh-độ-sáng)
7. [Grayscale — Chuyển xám](#7-grayscale--chuyển-xám)
8. [Mã hóa ảnh ra bytes](#8-mã-hóa-ảnh-ra-bytes)

---

## 1. Đọc ảnh từ bytes

```python
nparr = np.frombuffer(contents, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
```

### `np.frombuffer(buffer, dtype)`

Chuyển đổi raw bytes nhận từ request thành mảng NumPy 1D.

| Tham số  | Giá trị    | Ý nghĩa                                     |
| -------- | ---------- | ------------------------------------------- |
| `buffer` | `bytes`    | Nội dung file ảnh dạng bytes                |
| `dtype`  | `np.uint8` | Mỗi byte là 1 số nguyên 0–255 (pixel value) |

### `cv2.imdecode(buf, flags)`

Giải mã mảng bytes thành ảnh OpenCV (ma trận NumPy 3D: `height × width × channels`).

| Tham số | Giá trị                | Ý nghĩa                               |
| ------- | ---------------------- | ------------------------------------- |
| `buf`   | `np.ndarray`           | Mảng bytes đã chuyển đổi ở bước trên  |
| `flags` | `cv2.IMREAD_COLOR`     | Đọc ảnh màu BGR (bỏ qua alpha nếu có) |
|         | `cv2.IMREAD_GRAYSCALE` | Đọc thành ảnh xám                     |
|         | `cv2.IMREAD_UNCHANGED` | Giữ nguyên tất cả kênh kể cả alpha    |

> **Lưu ý:** OpenCV dùng thứ tự kênh **BGR** (Blue-Green-Red), ngược với RGB thông thường.  
> Nếu `imdecode` trả về `None` → file không phải ảnh hợp lệ hoặc bị hỏng.

---

## 2. Gaussian Blur — Làm mờ ảnh

```python
kernel_size = max(3, int(round(3 + blur * 2)))
if kernel_size % 2 == 0:
    kernel_size += 1

processed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```

### Nguyên lý

Gaussian Blur áp dụng bộ lọc (kernel) có trọng số hình chuông Gaussian lên mỗi pixel, lấy trung bình có trọng số các pixel lân cận. Kết quả là ảnh mịn, giảm nhiễu tần số cao.

### `cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)`

| Tham số  | Kiểu         | Ý nghĩa                                                                                          |
| -------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `src`    | `np.ndarray` | Ảnh đầu vào                                                                                      |
| `ksize`  | `(int, int)` | Kích thước kernel `(width, height)`. **Bắt buộc là số lẻ dương** (3, 5, 7,...). Càng lớn càng mờ |
| `sigmaX` | `float`      | Độ lệch chuẩn theo chiều ngang. `0` = tự tính từ `ksize`                                         |
| `sigmaY` | `float`      | Độ lệch chuẩn theo chiều dọc. Mặc định bằng `sigmaX`                                             |

### Công thức tính kernel_size theo tham số `blur` (0–5)

```
kernel_size = max(3, round(3 + blur × 2))
```

| `blur` | `kernel_size` | Mức độ  |
| ------ | ------------- | ------- |
| 0      | —             | Tắt     |
| 0.5    | 5             | Rất nhẹ |
| 1.0    | 5             | Nhẹ     |
| 2.0    | 7             | Vừa     |
| 3.0    | 9             | Rõ      |
| 5.0    | 13            | Mạnh    |

> **Lưu ý:** `ksize` bắt buộc phải là số nguyên lẻ. Nếu kết quả `round()` ra số chẵn thì `+1`.

---

## 3. Sharpen — Làm sắc nét

```python
center = 9.0 + sharpen
kernel = np.array([[-1, -1, -1],
                   [-1, center, -1],
                   [-1, -1, -1]])
processed_image = cv2.filter2D(image, -1, kernel)
```

### Nguyên lý

Sử dụng kernel tùy chỉnh để tăng cường độ tương phản cục bộ. Giá trị trung tâm lớn → tăng pixel hiện tại; các giá trị xung quanh âm → trừ đi pixel lân cận, làm nổi bật biên.

Tổng kernel = `center - 8`. Với `center = 9` → tổng = 1 (nhẹ), `center = 14` → tổng = 6 (mạnh).

### `cv2.filter2D(src, ddepth, kernel)`

| Tham số  | Kiểu         | Ý nghĩa                                           |
| -------- | ------------ | ------------------------------------------------- |
| `src`    | `np.ndarray` | Ảnh đầu vào                                       |
| `ddepth` | `int`        | Độ sâu ảnh đầu ra. `-1` = giữ nguyên kiểu ảnh gốc |
| `kernel` | `np.ndarray` | Ma trận bộ lọc tùy chỉnh                          |

### Giá trị center theo tham số `sharpen` (0–5)

| `sharpen` | `center` | Tổng kernel | Mức độ sắc nét |
| --------- | -------- | ----------- | -------------- |
| 0         | —        | —           | Tắt            |
| 1.0       | 10       | 2           | Nhẹ            |
| 2.0       | 11       | 3           | Vừa            |
| 3.0       | 12       | 4           | Rõ             |
| 5.0       | 14       | 6           | Mạnh           |

> **Lưu ý:** Sharpen quá mạnh có thể gây hiệu ứng "halo" (viền sáng quanh biên vật thể).

---

## 4. CLAHE Enhance — Tăng tương phản

```python
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=1.0 + enhance * 0.6, tileGridSize=(8, 8))
l = clahe.apply(l)

enhanced = cv2.merge([l, a, b])
processed_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

### Nguyên lý

**CLAHE** (Contrast Limited Adaptive Histogram Equalization) tăng tương phản cục bộ bằng cách cân bằng histogram theo từng vùng nhỏ, đồng thời giới hạn mức tăng để tránh khuếch đại nhiễu.

Thao tác được thực hiện trên **kênh L** (độ sáng) trong không gian màu **LAB**, giúp tăng tương phản mà không thay đổi màu sắc.

### Tại sao dùng LAB thay vì BGR trực tiếp?

Không gian màu LAB tách biệt thông tin độ sáng (L) với màu sắc (A, B). Xử lý chỉ trên kênh L đảm bảo màu sắc không bị biến đổi khi tăng tương phản.

### `cv2.cvtColor(src, code)`

| Tham số | Giá trị              | Ý nghĩa                       |
| ------- | -------------------- | ----------------------------- |
| `src`   | `np.ndarray`         | Ảnh đầu vào                   |
| `code`  | `cv2.COLOR_BGR2LAB`  | Chuyển từ BGR sang LAB        |
|         | `cv2.COLOR_LAB2BGR`  | Chuyển từ LAB về BGR          |
|         | `cv2.COLOR_BGR2GRAY` | Chuyển từ BGR sang grayscale  |
|         | `cv2.COLOR_GRAY2BGR` | Chuyển grayscale thành 3 kênh |

### `cv2.createCLAHE(clipLimit, tileGridSize)`

| Tham số        | Kiểu        | Ý nghĩa                                                                                  |
| -------------- | ----------- | ---------------------------------------------------------------------------------------- |
| `clipLimit`    | `float`     | Giới hạn khuếch đại histogram. Càng cao → tương phản càng mạnh. Khuyến nghị: **1.5–4.0** |
| `tileGridSize` | `(int,int)` | Chia ảnh thành lưới ô để xử lý cục bộ. `(8,8)` = 64 ô, phù hợp hầu hết ảnh               |

### `clahe.apply(channel)`

Áp dụng CLAHE lên một kênh đơn (grayscale hoặc kênh L của LAB).

### Công thức clipLimit theo tham số `enhance` (0–5)

```
clipLimit = 1.0 + enhance × 0.6
```

| `enhance` | `clipLimit` | Hiệu ứng |
| --------- | ----------- | -------- |
| 0         | —           | Tắt      |
| 1.0       | 1.6         | Nhẹ      |
| 2.0       | 2.2         | Vừa      |
| 3.0       | 2.8         | Rõ       |
| 5.0       | 4.0         | Mạnh     |

---

## 5. Denoise — Khử nhiễu

```python
h_val = 3.0 + denoise * 3.4

processed_image = cv2.fastNlMeansDenoisingColored(
    image,
    None,
    h=h_val,
    hColor=h_val,
    templateWindowSize=7,
    searchWindowSize=21,
)
```

### Nguyên lý

**Non-Local Means Denoising** khử nhiễu bằng cách so sánh từng patch (vùng nhỏ) với tất cả các patch khác trong vùng tìm kiếm, lấy trung bình có trọng số. Patch nào giống nhau thì trọng số cao hơn → giữ lại chi tiết tốt hơn Gaussian Blur.

`fastNlMeansDenoisingColored` là phiên bản cho ảnh màu: xử lý kênh luminance và kênh màu riêng biệt.

### `cv2.fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize)`

| Tham số              | Kiểu         | Ý nghĩa                                                                                                       |
| -------------------- | ------------ | ------------------------------------------------------------------------------------------------------------- |
| `src`                | `np.ndarray` | Ảnh màu đầu vào (BGR, uint8)                                                                                  |
| `dst`                | `np.ndarray` | Ảnh đầu ra (None = tạo mới)                                                                                   |
| `h`                  | `float`      | Cường độ lọc cho kênh luminance. **3–5**: nhẹ, giữ chi tiết. **10**: chuẩn. **15–20**: mạnh. **> 25**: ảnh mờ |
| `hColor`             | `float`      | Cường độ lọc cho kênh màu. Thường bằng `h`                                                                    |
| `templateWindowSize` | `int` (lẻ)   | Kích thước patch để so sánh. Mặc định `7` (7×7). Lớn hơn → chính xác hơn nhưng chậm hơn                       |
| `searchWindowSize`   | `int` (lẻ)   | Vùng tìm kiếm patch tương đồng. Mặc định `21` (21×21). Lớn hơn → tốt hơn nhưng **rất chậm**                   |

### Công thức h_val theo tham số `denoise` (0–5)

```
h_val = 3.0 + denoise × 3.4
```

| `denoise` | `h_val` | Hiệu ứng  |
| --------- | ------- | --------- |
| 0         | —       | Tắt       |
| 0.5       | ~4.7    | Rất nhẹ   |
| 1.0       | 6.4     | Nhẹ       |
| 2.0       | 9.8     | Vừa       |
| 3.0       | 13.2    | Rõ        |
| 5.0       | 20.0    | Mạnh nhất |

> **Lưu ý hiệu năng:** `fastNlMeansDenoisingColored` **rất chậm** so với các filter khác vì phải so sánh patch toàn vùng tìm kiếm. Ảnh lớn (> 2MP) có thể mất vài giây.

---

## 6. Brightness — Điều chỉnh độ sáng

```python
offset = 255.0 * brightness
processed_image = np.clip(
    image.astype(np.float32) + offset, 0, 255
).astype(np.uint8)
```

### Nguyên lý

Cộng/trừ một giá trị cố định vào tất cả pixel. Sau đó `np.clip` giới hạn kết quả trong khoảng [0, 255] để không bị tràn số.

### `np.clip(array, a_min, a_max)`

| Tham số | Ý nghĩa                            |
| ------- | ---------------------------------- |
| `array` | Mảng đầu vào                       |
| `a_min` | Giá trị tối thiểu (pixel < 0 → 0)  |
| `a_max` | Giá trị tối đa (pixel > 255 → 255) |

### Tại sao cần ép kiểu `float32` trước?

Ảnh OpenCV mặc định là `uint8` (0–255). Phép cộng `uint8 + offset` sẽ **tràn số** (overflow) nếu kết quả > 255. Chuyển sang `float32` trước, xử lý, rồi `clip` về [0,255] và đổi lại `uint8` đảm bảo kết quả chính xác.

### Giá trị offset theo tham số `brightness` (–1 đến 1)

```
offset = 255 × brightness
```

| `brightness` | `offset` | Hiệu ứng         |
| ------------ | -------- | ---------------- |
| -1.0         | -255     | Tối hoàn toàn    |
| -0.3         | -76      | Tối nhẹ          |
| 0            | 0        | Giữ nguyên (tắt) |
| 0.2          | +51      | Sáng nhẹ         |
| 0.5          | +127     | Sáng rõ          |
| 1.0          | +255     | Trắng hoàn toàn  |

---

## 7. Grayscale — Chuyển xám

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
alpha = max(0.0, min(1.0, grayscale))
processed_image = cv2.addWeighted(gray_bgr, alpha, image, 1.0 - alpha, 0)
```

### Nguyên lý

Chuyển ảnh sang xám rồi **blend** với ảnh màu gốc theo tỷ lệ `grayscale`. Cho phép hiệu ứng "xám một phần" thay vì chỉ xám hoàn toàn.

### `cv2.COLOR_BGR2GRAY`

Công thức chuyển đổi (theo chuẩn ITU-R BT.601):

```
Gray = 0.114×B + 0.587×G + 0.299×R
```

Trọng số khác nhau vì mắt người nhạy cảm với màu xanh lá (Green) hơn.

### `cv2.addWeighted(src1, alpha, src2, beta, gamma)`

Blend hai ảnh theo công thức:

```
output = src1 × alpha + src2 × beta + gamma
```

| Tham số | Kiểu         | Ý nghĩa                                            |
| ------- | ------------ | -------------------------------------------------- |
| `src1`  | `np.ndarray` | Ảnh thứ nhất (ảnh xám ở dạng BGR)                  |
| `alpha` | `float`      | Trọng số ảnh thứ nhất (= `grayscale`)              |
| `src2`  | `np.ndarray` | Ảnh thứ hai (ảnh màu gốc)                          |
| `beta`  | `float`      | Trọng số ảnh thứ hai (= `1 - grayscale`)           |
| `gamma` | `float`      | Hằng số cộng thêm vào kết quả. `0` = không thêm gì |

### Kết quả theo tham số `grayscale` (0–1)

| `grayscale` | Kết quả                   |
| ----------- | ------------------------- |
| 0           | Tắt (giữ nguyên ảnh màu)  |
| 0.3         | 30% xám, 70% màu          |
| 0.5         | Blend đều giữa màu và xám |
| 0.8         | 80% xám, 20% màu          |
| 1.0         | Xám hoàn toàn             |

---

## 8. Mã hóa ảnh ra bytes

```python
success, encoded_image = cv2.imencode(".png", processed_image)
return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")
```

### `cv2.imencode(ext, img, params)`

Mã hóa ảnh NumPy thành buffer bytes theo định dạng chỉ định.

| Tham số  | Kiểu         | Ý nghĩa                                                                              |
| -------- | ------------ | ------------------------------------------------------------------------------------ |
| `ext`    | `str`        | Định dạng đầu ra: `".png"`, `".jpg"`, `".bmp"`, `".webp"`, ...                       |
| `img`    | `np.ndarray` | Ảnh đầu vào                                                                          |
| `params` | `list`       | Tham số nén (optional). VD: `[cv2.IMWRITE_JPEG_QUALITY, 90]` cho JPEG chất lượng 90% |

**Giá trị trả về:**

- `success` (`bool`): `True` nếu mã hóa thành công
- `encoded_image` (`np.ndarray`): Buffer bytes của ảnh đã mã hóa

### PNG vs JPEG

| Định dạng | Nén      | Chất lượng        | Dùng khi                           |
| --------- | -------- | ----------------- | ---------------------------------- |
| `.png`    | Lossless | Không mất dữ liệu | Cần giữ chất lượng gốc, ảnh xử lý  |
| `.jpg`    | Lossy    | Mất một phần      | Cần file nhỏ, ảnh cuối để hiển thị |

> Backend hiện tại luôn trả về PNG để đảm bảo chất lượng khi FE tiếp tục xử lý.

---

## Tóm Tắt Thứ Tự Pipeline

```
File upload (bytes)
  ↓  np.frombuffer + cv2.imdecode
[Ảnh gốc BGR]
  ↓  fastNlMeansDenoisingColored  (denoise)
  ↓  cv2.GaussianBlur             (blur)
  ↓  cv2.filter2D                 (sharpen)
  ↓  CLAHE trên kênh L (LAB)      (enhance)
  ↓  np.clip với offset           (brightness)
  ↓  cv2.addWeighted              (grayscale)
[Ảnh đã xử lý BGR]
  ↓  cv2.imencode(".png")
PNG bytes → StreamingResponse
```

Mỗi bước chỉ được thực thi nếu tham số tương ứng **khác `None` và khác `0`** (riêng `brightness` khác `None` và khác `0.0`).
