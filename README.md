### Hệ thống nhận diện tin tức giả mạo
Sử dụng mô hình Transformers để xử lí dữ liệu văn bản đầu vào, hoạt động như một mô hình NLP xử lí dữ liệu văn bản dưới dạng các vector đặc trưng biểu diễn nội dung văn bản.
Sử dụng mô hình học máy mạng tích chập để huấn luyện dữ liệu sau khi được xử lí.

### Dữ liệu
- Path: data/raw/
- Định dạng: .csv

- Format dữ liệu huấn luyện
```python
|   title   |   author  |   text    |  label    |
```

- Format dữ liệu đầu vào
```python
|   title   |   author  |   text    |
```

### Cách vận hành
- Cài đặt Python phiên bản >= 3.10
- Cài đặt các thư viện cần thiết
```bash
pip install -r requirement.txt
```

- Thay đổi được dẫn đến thư mục dự án ở các file .env
```python
#example
FOLDER_PATH=/home/huy31/Projects/KDLKP/Fake-news-detection-system
```

- Huấn luyện dữ liệu
```bash
python utils/train.py
```

- Dự đoán nhãn từ dữ liệu đầu vào
```bash
python utils/predict.py
```

- Run server
```bash
fastapi dev main.py
```

### Mô hình
- Path: data/model/
- Mã nguồn mô hình: utils/model