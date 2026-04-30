# API Tài Liệu - Image Processing Backend

## Tổng Quan

Backend API xử lý ảnh được xây dựng bằng **FastAPI** và **OpenCV**, cung cấp các chức năng xử lý ảnh, lưu trữ và quản lý hình ảnh.

### Base URL

```
http://localhost:8000
```

### Định dạng phản hồi

Tất cả phản hồi JSON bao gồm:

- **Thành công (200)**: Dữ liệu yêu cầu
- **Lỗi (4xx/5xx)**: JSON với trường `error` hoặc `message`

---

## API Endpoints

### 1. POST `/api/process-image` - Xử Lý Ảnh

Xử lý ảnh với nhiều bộ lọc cùng lúc. Tất cả tham số filter đều optional.

#### Request Headers

```
Content-Type: multipart/form-data
```

#### Request Body (multipart/form-data)

| Trường    | Kiểu    | Bắt buộc | Mô tả                 |
| --------- | ------- | -------- | --------------------- |
| `file`    | File    | Có       | Ảnh đầu vào           |
| `blur`    | integer | Không    | Mức độ blur từ 1-5    |
| `sharpen` | integer | Không    | Mức độ sharpen từ 1-5 |
| `enhance` | integer | Không    | Mức độ enhance từ 1-5 |
| `denoise` | integer | Không    | Mức độ denoise từ 1-5 |

Lưu ý:

- Cần gửi ít nhất 1 filter (blur/sharpen/enhance/denoise)
- Thứ tự áp dụng cố định trên backend: `denoise -> blur -> sharpen -> enhance`

#### Response (Thành công - 200)

```
Content-Type: image/png
File nhị phân của ảnh đã xử lý
```

#### Response (Lỗi)

```json
{
	"error": "Cần ít nhất một filter: blur, sharpen, enhance, denoise"
}
```

#### Ví dụ Request cơ bản (JavaScript/Fetch)

```javascript
const formData = new FormData();
formData.append("file", imageFile);
formData.append("blur", "2");
formData.append("sharpen", "1");

const response = await fetch("http://localhost:8000/api/process-image", {
	method: "POST",
	body: formData,
});

const blob = await response.blob();
```

#### Chỉnh Ảnh Lần Kế Tiếp: FE Gửi Blob Như Thế Nào?

Backend là stateless, nên lần tiếp theo FE phải tự gửi lại ảnh đầu vào (file hoặc blob) trong field `file`.

1. Trường hợp muốn cộng thêm filter (chain)

- Dùng blob ảnh đã xử lý ở lần trước làm input mới.
- Gửi lại blob qua `FormData.append("file", blob, "processed.png")`.

```javascript
async function processNext(currentBlob) {
	const formData = new FormData();
	formData.append("file", currentBlob, "processed.png");
	formData.append("enhance", "2");

	const response = await fetch("http://localhost:8000/api/process-image", {
		method: "POST",
		body: formData,
	});

	if (!response.ok) {
		throw new Error("Process failed");
	}

	return await response.blob();
}
```

2. Trường hợp muốn đổi cấu hình từ ảnh gốc (không chồng hiệu ứng cũ)

- Không dùng blob đã xử lý trước đó.
- Gửi lại file gốc ban đầu (`originalFile`) cùng filter mới.

```javascript
async function processFromOriginal(originalFile) {
	const formData = new FormData();
	formData.append("file", originalFile);
	formData.append("blur", "3");

	const response = await fetch("http://localhost:8000/api/process-image", {
		method: "POST",
		body: formData,
	});

	return await response.blob();
}
```

Khuyến nghị FE:

- Luôn giữ cả `originalFile` và `currentBlob`
- `currentBlob` dùng để chain filter
- `originalFile` dùng khi user đổi/reset filter để tránh xử lý chồng nhiều lần

#### Các Bộ Lọc Hỗ Trợ

| Loại      | Mô tả                            | Tham số Intensity                            |
| --------- | -------------------------------- | -------------------------------------------- |
| `blur`    | Mịn ảnh (Gaussian Blur)          | Kernel size tăng dần (3->5->7->9->11)        |
| `sharpen` | Tăng sắc nét                     | Hệ số kernel tăng (9->10->11->12->13)        |
| `enhance` | Tăng cường độ tương phản (CLAHE) | clipLimit tăng dần (2.0->3.0->4.0->5.0->6.0) |
| `denoise` | Khử nhiễu                        | Cường độ tăng dần (h = 10->50)               |

---

### 2. POST `/save-image` - Lưu Ảnh

Lưu ảnh vào thư mục `images/` trên backend.

#### Request Headers

```
Content-Type: multipart/form-data
```

#### Query Parameters

| Tham số    | Kiểu   | Mặc định | Mô tả                                                     |
| ---------- | ------ | -------- | --------------------------------------------------------- |
| `filename` | string | `null`   | Tên file (tùy chọn, nếu bỏ trống sẽ tự tạo với timestamp) |

#### Request Body

```
file: File (multipart/form-data)
```

#### Response (Thành công - 200)

```json
{
	"message": "Ảnh đã được lưu thành công",
	"filename": "my_image.png",
	"path": "images/my_image.png"
}
```

#### Response (Lỗi)

```json
{
	"error": "Tên file không hợp lệ"
}
```

#### Ví dụ Requests

**JavaScript/Fetch:**

```javascript
const formData = new FormData();
formData.append("file", imageFile);
formData.append("filename", "my_vacation_photo");

const response = await fetch("http://localhost:8000/save-image", {
	method: "POST",
	body: formData,
});

const result = await response.json();
console.log("Saved as:", result.filename);
```

**Python/Requests:**

```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    params = {'filename': 'sunset_photo'}
    response = requests.post(
        'http://localhost:8000/save-image',
        files=files,
        params=params
    )
    result = response.json()
    print(f"Saved: {result['filename']}")
```

#### Ghi Chú

- Nếu không cung cấp `filename`, hệ thống sẽ tự tạo tên theo định dạng: `image_YYYYMMDD_HHMMSS.ext`
- Nếu `filename` không có phần mở rộng, sẽ tự thêm `.png`
- Các ký tự đặc biệt (/, \) sẽ bị loại bỏ vì lý do an toàn

---

### 3. GET `/get-saved-images` - Lấy Danh Sách Ảnh

Lấy danh sách tất cả ảnh đã lưu trong thư mục `images/`.

#### Response (Thành công - 200)

```json
{
	"images": [
		{
			"filename": "sunset_photo.png",
			"size": 245678,
			"created": "2026-04-24T10:30:45.123456"
		},
		{
			"filename": "image_20260424_103015.jpg",
			"size": 512000,
			"created": "2026-04-24T10:30:15.654321"
		}
	]
}
```

#### Response (Thư mục trống)

```json
{
	"images": []
}
```

#### Ví dụ Requests

**JavaScript/Fetch:**

```javascript
const response = await fetch("http://localhost:8000/get-saved-images");
const data = await response.json();

data.images.forEach((img) => {
	console.log(`${img.filename} - ${img.size} bytes`);
});
```

**Python/Requests:**

```python
import requests

response = requests.get('http://localhost:8000/get-saved-images')
images = response.json()['images']

for img in images:
    print(f"{img['filename']}: {img['size']} bytes")
```

---

### 4. GET `/download-image/{filename}` - Tải Ảnh

Tải ảnh đã lưu từ backend về client.

#### URL Parameters

| Tham số    | Kiểu   | Mô tả                                         |
| ---------- | ------ | --------------------------------------------- |
| `filename` | string | Tên file cần tải (từ API `/get-saved-images`) |

#### Response (Thành công - 200)

```
Content-Type: image/png
File nhị phân của ảnh
```

#### Response (Lỗi - 404)

```json
{
	"error": "File không tồn tại"
}
```

#### Response (Lỗi - 400)

```json
{
	"error": "Tên file không hợp lệ"
}
```

#### Ví dụ Requests

**JavaScript/Fetch:**

```javascript
const filename = "sunset_photo.png";
const response = await fetch(
	`http://localhost:8000/download-image/${filename}`,
);

if (response.ok) {
	const blob = await response.blob();
	const url = URL.createObjectURL(blob);

	// Hiển thị ảnh
	document.getElementById("image").src = url;

	// Hoặc tải về
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	a.click();
}
```

**Python/Requests:**

```python
import requests

filename = 'sunset_photo.png'
response = requests.get(f'http://localhost:8000/download-image/{filename}')

if response.status_code == 200:
    with open(f'downloaded_{filename}', 'wb') as f:
        f.write(response.content)
```

---

## Xử Lý Lỗi

### HTTP Status Codes

| Code | Ý Nghĩa                                                       |
| ---- | ------------------------------------------------------------- |
| 200  | Thành công                                                    |
| 400  | Yêu cầu không hợp lệ (file sai định dạng, tên file sai, v.v.) |
| 404  | Tài nguyên không tìm thấy (file không tồn tại)                |
| 500  | Lỗi server                                                    |

### Error Response Format

```json
{
	"error": "Mô tả chi tiết của lỗi"
}
```

---

## Ví Dụ Quy Trình Hoàn Chỉnh (Frontend)

### 1. Xử Lý và Hiển Thị Ảnh

```javascript
async function processAndDisplayImage(file) {
	const formData = new FormData();
	formData.append("file", file);

	try {
		const response = await fetch(
			"http://localhost:8000/process-image?filter_type=sharpen&intensity=2",
			{
				method: "POST",
				body: formData,
			},
		);

		if (!response.ok) {
			const error = await response.json();
			alert(`Lỗi: ${error.error}`);
			return;
		}

		const blob = await response.blob();
		const url = URL.createObjectURL(blob);
		document.getElementById("processedImage").src = url;
	} catch (err) {
		console.error("Error:", err);
		alert("Không thể xử lý ảnh");
	}
}
```

### 2. Lưu Ảnh và Lấy Danh Sách

```javascript
async function saveAndListImages(file, filename) {
	const formData = new FormData();
	formData.append("file", file);
	if (filename) formData.append("filename", filename);

	try {
		const saveResponse = await fetch("http://localhost:8000/save-image", {
			method: "POST",
			body: formData,
		});

		if (!saveResponse.ok) {
			const error = await saveResponse.json();
			alert(`Lỗi: ${error.error}`);
			return;
		}

		const saveResult = await saveResponse.json();
		console.log(`Lưu thành công: ${saveResult.filename}`);

		// Lấy danh sách ảnh đã lưu
		const listResponse = await fetch("http://localhost:8000/get-saved-images");
		const imageList = await listResponse.json();

		// Hiển thị danh sách
		const ul = document.getElementById("imageList");
		ul.innerHTML = "";
		imageList.images.forEach((img) => {
			const li = document.createElement("li");
			li.innerHTML = `
        ${img.filename} 
        <button onclick="downloadImage('${img.filename}')">Tải</button>
      `;
			ul.appendChild(li);
		});
	} catch (err) {
		console.error("Error:", err);
		alert("Lỗi xử lý");
	}
}

async function downloadImage(filename) {
	const response = await fetch(
		`http://localhost:8000/download-image/${filename}`,
	);
	const blob = await response.blob();
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	a.click();
}
```

---

## Cài Đặt & Chạy Backend

### 1. Cài Dependencies

```bash
cd server
pip install -r requirements.txt
```

### 2. Chạy Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server sẽ chạy tại: `http://localhost:8000`

### 3. API Documentation (Swagger UI)

Truy cập: `http://localhost:8000/docs`

---

## CORS Configuration (Nếu Frontend Ở Khác Domain)

Nếu Frontend chạy ở domain khác (ví dụ: `localhost:3000`), thêm CORS middleware vào `main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Thêm domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Giới Hạn & Lưu Ý

- **Kích thước file**: Không có giới hạn cụ thể, tùy thuộc vào cấu hình server
- **Định dạng hỗ trợ**: JPEG, PNG, BMP, GIF, TIFF (OpenCV hỗ trợ)
- **Cách xử lý tệp**: File ảnh được tạm lưu trong bộ nhớ, nên lớn không quá 100MB
- **Thư mục lưu trữ**: `images/` được tạo tự động trong thư mục server

---

## Troubleshooting

### Lỗi "CORS" khi gọi từ Frontend

**Giải pháp**: Cấu hình CORS middleware như phần trên

### Lỗi "File input/output error"

**Giải pháp**: Kiểm tra quyền ghi vào thư mục `images/`

### Ảnh được trả về lỗi

**Giải pháp**: Đảm bảo định dạng ảnh được hỗ trợ (JPEG, PNG, BMP, GIF)

---

## Liên Hệ & Support

Nếu có vấn đề, kiểm tra logs của server hoặc liên hệ team phát triển.
