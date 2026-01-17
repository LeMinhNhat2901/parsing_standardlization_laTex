# Reference Matching Pipeline - Hướng Dẫn Sử Dụng

## ⚠️ YÊU CẦU QUAN TRỌNG từ text2.txt

Theo **Section 2.2.2**:
> "Manually label references for at least 5 publications"

**NGHĨA LÀ:** Sinh viên **PHẢI TỰ TAY** label ít nhất 5 publications với tổng ít nhất 20 pairs.

**KHÔNG được** sử dụng automatic matching cho manual labels!

---

## 📋 Quy Trình Đầy Đủ

### Bước 1: Tạo Manual Labels (BẮT BUỘC TỰ TAY)

```bash
# Chạy tool tạo manual labels
python src/create_manual_labels.py --output-dir output --num-pubs 5

# Tool sẽ hiển thị:
# 1. BibTeX entry cần label
# 2. Top 3 candidates (chỉ để GỢI Ý)
# 3. YÊU CẦU bạn CHỌN TỰ TAY match nào đúng

# Output: manual_labels.json
```

**Lưu ý:** 
- Tool CHỈ gợi ý candidates dựa trên similarity scores
- **BẠN** phải tự xem xét và quyết định match nào đúng
- Nhập số thứ tự (1-3) để chọn, 'n' để bỏ qua, 'q' để dừng

**Test mode (CHỈ để kiểm tra nhanh, KHÔNG dùng cho submission):**
```bash
python src/create_manual_labels.py --auto
# ⚠️ KHÔNG hợp lệ cho submission!
```

### Bước 2: Chạy ML Pipeline

**Cách 1: Sử dụng wrapper (Khuyến nghị)**
```bash
python run_matching.py --data-dir output
```

**Cách 2: Chạy trực tiếp main_matcher.py**
```bash
# Từ thư mục gốc
python src/main_matcher.py --data-dir output --manual-labels manual_labels.json

# Từ thư mục src
cd src
python main_matcher.py --data-dir ../output --manual-labels ../manual_labels.json
```

**Các options:**
```bash
python run_matching.py \
  --data-dir output \
  --manual-labels manual_labels.json \
  --output-dir ml_results \
  --model-type classifier \
  --tune-hyperparams \
  --use-gpu
```

---

## 📁 Cấu Trúc File

### Input Files

1. **manual_labels.json** (YÊU CẦU TỰ TAY TẠO)
```json
{
  "2504-13946": {
    "abramoff2018pivotal": "1234-56789",
    "biden2023executive": "9876-54321"
  },
  "2504-13947": {
    ...
  }
}
```

2. **output/** directory
```
output/
├── 2504-13946/
│   ├── refs.bib
│   ├── references.json
│   ├── hierarchy.json
│   └── metadata.json
├── 2504-13947/
│   └── ...
```

### Output Files

```
ml_output/
├── pred_2504-13946.json     # Predictions cho từng publication
├── pred_2504-13947.json
├── evaluation_report.json   # Metrics (MRR, Hit@K)
└── feature_importance.csv   # Feature analysis
```

---

## 🔍 So Sánh 2 Cách Chạy

### run_matching.py (Wrapper)

**Ưu điểm:**
- ✅ Tự động chuyển đường dẫn tương đối thành tuyệt đối
- ✅ Default values cho manual_labels.json
- ✅ Đơn giản, dễ sử dụng
- ✅ Tương thích với cả Windows và Linux

**Nhược điểm:**
- ❌ Thêm 1 lớp wrapper (phức tạp hơn một chút)

```bash
# Chỉ cần chỉ định data-dir
python run_matching.py --data-dir output
```

### main_matcher.py (Direct)

**Ưu điểm:**
- ✅ Trực tiếp, không qua wrapper
- ✅ Rõ ràng hơn về flow
- ✅ Dễ debug

**Nhược điểm:**
- ❌ Phải chỉ định đầy đủ các đường dẫn
- ❌ Phải cẩn thận với relative vs absolute paths

```bash
# Phải chỉ định đầy đủ
python src/main_matcher.py --data-dir output --manual-labels manual_labels.json
```

---

## ✅ Compliance với text2.txt

Pipeline này đáp ứng 100% yêu cầu Section 2.2:

| Requirement | Implementation | Status |
|------------|----------------|--------|
| 2.2.1 Data Cleaning | Text preprocessing, lowercasing, tokenization | ✅ |
| 2.2.2 Manual Labeling | ≥5 pubs, ≥20 pairs (TỰ TAY) | ✅ |
| 2.2.2 Auto Labeling | ≥10% auto-labeled | ✅ |
| 2.2.3 Feature Engineering | 37 features across 5 groups | ✅ |
| 2.2.4 Data Modeling | m×n pairs, proper split | ✅ |
| 2.2.5 Evaluation | MRR on top-5 predictions | ✅ |

---

## 🐛 Troubleshooting

### RecursionError
```bash
# Đã fix: Thay isinstance() bằng type().__name__
# Tăng recursion limit: sys.setrecursionlimit(3000)
```

### bibtexparser not found
```bash
pip install bibtexparser
```

### Manual labels không hợp lệ
```bash
# Kiểm tra format:
python -c "import json; print(json.load(open('manual_labels.json')))"

# Yêu cầu:
# - ≥5 publications
# - ≥20 total pairs
# - Phải TỰ TAY tạo (không dùng --auto)
```

### Path not found
```bash
# Sử dụng absolute paths
python src/main_matcher.py --data-dir "D:/GitHub/parsing_standardlization_laTex/output"

# Hoặc dùng wrapper tự động xử lý
python run_matching.py --data-dir output
```

---

## 📝 Khuyến Nghị

1. **Tạo manual labels:** Dùng `create_manual_labels.py` KHÔNG có `--auto`
2. **Chạy pipeline:** Dùng `run_matching.py` (đơn giản nhất)
3. **Kiểm tra output:** Xem `ml_output/evaluation_report.json`
4. **Submit:** Chỉ submit code + report, không submit data

---

## 📚 Reference

- Lab requirement: `text2.txt`
- Manual labels tool: `src/create_manual_labels.py`
- ML pipeline: `src/main_matcher.py`
- Wrapper: `run_matching.py`
