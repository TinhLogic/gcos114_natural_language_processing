import os # Thư viện tương tác với hệ điều hành (duyệt file, đường dẫn)
import re # Thư viện Regular Expression để xử lý chuỗi theo quy tắc
import numpy as np # Thư viện tính toán ma trận và mảng đa chiều
from sklearn.svm import LinearSVC # Thuật toán phân loại Support Vector Machine tuyến tính

# ======================================================
# 1 ĐỌC FILE & GIỮ NGUYÊN XML <s ...>
# ======================================================
def read_sentences(file_path): # Định nghĩa hàm đọc các câu từ file
    sentences = [] # Khởi tạo danh sách trống để lưu các câu
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f: # Mở file, bỏ qua lỗi mã hóa
        for line in f: # Duyệt từng dòng trong file
            line = line.strip() # Xóa khoảng trắng thừa ở hai đầu dòng
            if line.startswith("<s ") and line.endswith("</s>"): # Kiểm tra dòng có nằm trong thẻ câu <s>
                sentences.append(line) # Thêm dòng thỏa mãn vào danh sách
    return sentences # Trả về danh sách các câu dạng XML


# ======================================================
# 2 LẤY TEXT BÊN TRONG <s>
# ======================================================
def clean_sentence(xml_sentence): # Hàm loại bỏ các thẻ XML để lấy nội dung text
    return re.sub(r"<.*?>", "", xml_sentence).strip() # Thay thế các thẻ <...> bằng rỗng và xóa khoảng trắng


# ======================================================
# 3 TRÍCH 5 ĐẶC TRƯNG
# ======================================================
def extract_features(sentence, index, total_sentences, thematic_words): # Hàm trích xuất vector đặc trưng
    words = sentence.split() # Tách câu thành danh sách các từ
    features = [] # Khởi tạo danh sách lưu các giá trị đặc trưng

    # 1 Sentence Length
    features.append(len(words)) # Đặc trưng 1: Độ dài của câu (tính theo số từ)

    # 2 Fixed Phrase Feature
    discourse_phrases = [ # Danh sách các cụm từ cố định thường xuất hiện trong tóm tắt
        "in conclusion", "in summary", "overall",
        "this paper", "this study"
    ]
    features.append(1 if any(p in sentence.lower() for p in discourse_phrases) else 0) # Gán 1 nếu có cụm từ, 0 nếu không

    # 3 Sentence Position (one-hot)
    if index == 0: # Nếu là câu đầu tiên
        features.extend([1, 0, 0]) # Mã hóa vị trí đầu: [1, 0, 0]
    elif index == total_sentences - 1: # Nếu là câu cuối cùng
        features.extend([0, 0, 1]) # Mã hóa vị trí cuối: [0, 0, 1]
    else: # Các câu nằm ở giữa văn bản
        features.extend([0, 1, 0]) # Mã hóa vị trí giữa: [0, 1, 0]

    # 4 Thematic Word Feature
    thematic_count = sum(1 for w in words if w.lower() in thematic_words) # Đếm số từ thuộc tập từ chủ đề
    features.append(thematic_count) # Đặc trưng 4: Số lượng từ chủ đề trong câu

    # 5 Uppercase Word Feature
    uppercase_count = sum(1 for w in words if w.isupper()) # Đếm các từ viết hoa hoàn toàn (tên riêng, từ viết tắt)
    features.append(uppercase_count) # Đặc trưng 5: Số lượng từ viết hoa

    return features # Trả về danh sách đặc trưng của một câu


# ======================================================
# 4 GÁN NHÃN – SO KHỚP CÂU
# ======================================================
def create_labels(text_xml, sum_clean): # Hàm tạo nhãn đúng/sai cho huấn luyện
    labels = [] # Khởi tạo danh sách nhãn
    sum_set = set(sum_clean) # Chuyển danh sách tóm tắt sang tập hợp để tìm kiếm nhanh O(1)

    for s in text_xml: # Duyệt từng câu trong văn bản gốc
        clean = clean_sentence(s) # Làm sạch câu gốc (bỏ XML)
        labels.append(1 if clean in sum_set else 0) # Gán 1 nếu câu có trong bản tóm tắt mẫu, 0 nếu không

    return labels # Trả về danh sách nhãn (0 hoặc 1)


# ======================================================
# 5 TÓM TẮT 1 DOCUMENT
# ======================================================
def summarize(doc_name, text_file, sum_file, max_sentences=20): # Hàm thực hiện tóm tắt cho một tài liệu
    text_xml = read_sentences(text_file) # Đọc câu từ file gốc
    sum_xml = read_sentences(sum_file) # Đọc câu từ file tóm tắt mẫu

    text_clean = [clean_sentence(s) for s in text_xml] # Làm sạch toàn bộ câu gốc
    sum_clean = [clean_sentence(s) for s in sum_xml] # Làm sạch toàn bộ câu tóm tắt mẫu

    thematic_words = set(" ".join(sum_clean).lower().split()) # Tạo tập từ chủ đề từ bản tóm tắt mẫu

    X = [] # Khởi tạo ma trận đặc trưng
    for i, sent in enumerate(text_clean): # Duyệt qua từng câu và chỉ số của nó
        X.append(extract_features(sent, i, len(text_clean), thematic_words)) # Trích đặc trưng cho từng câu
    X = np.array(X) # Chuyển X sang định dạng mảng Numpy để đưa vào mô hình

    y = create_labels(text_xml, sum_clean) # Tạo danh sách nhãn tương ứng cho các câu

    summary = [] # Danh sách lưu kết quả tóm tắt cuối cùng
    used = set() # Tập hợp kiểm tra trùng lặp nội dung

    # =========================
    # TRƯỜNG HỢP 1 CLASS
    # =========================
    if len(set(y)) < 2: # Kiểm tra nếu dữ liệu thiếu nhãn 0 hoặc 1 (không đủ điều kiện để mô hình SVM học phân loại)
        print("⚠️ Chỉ có 1 lớp nhãn → sử dụng thuật toán heuristic") # Thông báo dùng phương pháp xếp hạng thủ công thay vì máy học

        scores = [] # Khởi tạo danh sách điểm số cho các câu
        for i, s in enumerate(text_clean): # Duyệt từng câu
            score = len(s.split()) # Điểm dựa trên độ dài câu
            if i == 0: # Nếu là câu đầu tiên
                score += 10 # Ưu tiên cộng thêm điểm (thường câu đầu quan trọng)
            scores.append(score) # Thêm điểm vào danh sách

        ranked = sorted( # Sắp xếp các câu dựa trên điểm số
            zip(text_xml, scores), # Ghép câu với điểm tương ứng
            key=lambda x: x[1], # Sắp xếp theo giá trị điểm (index 1)
            reverse=True # Sắp xếp giảm dần
        )

        for s, _ in ranked: # Duyệt qua danh sách đã xếp hạng
            clean = clean_sentence(s) # Làm sạch để kiểm tra trùng
            if clean not in used: # Nếu nội dung câu chưa được chọn
                summary.append(s) # Thêm câu vào bản tóm tắt
                used.add(clean) # Đánh dấu nội dung đã sử dụng
            if len(summary) >= max_sentences: # Dừng lại nếu đủ số lượng câu tối đa
                break

    # =========================
    # TRAIN SVM
    # =========================
    else: # Trường hợp dữ liệu có đủ cả nhãn 0 và 1 (đủ điều kiện huấn luyện mô hình phân loại)
        model = LinearSVC(max_iter=5000) # Khởi tạo mô hình SVM với 5000 vòng lặp tối đa
        model.fit(X, y) # Huấn luyện mô hình dựa trên đặc trưng X và nhãn y
        preds = model.predict(X) # Dự đoán nhãn cho các câu trong văn bản

        for s, p in zip(text_xml, preds): # Duyệt qua câu gốc và kết quả dự đoán tương ứng
            clean = clean_sentence(s) # Làm sạch câu
            if p == 1 and clean not in used: # Nếu dự đoán là quan trọng (1) và chưa trùng
                summary.append(s) # Thêm câu vào bản tóm tắt
                used.add(clean) # Đánh dấu đã sử dụng
            if len(summary) >= max_sentences: # Dừng lại nếu đủ số lượng câu
                break

    # =========================
    # IN KẾT QUẢ
    # =========================
    print("\n" + "=" * 50) # In đường kẻ phân cách trên
    print(f"TÓM TẮT VĂN BẢN CỦA DOCUMENT {doc_name}") # In tiêu đề tài liệu đang xử lý
    print("=" * 50 + "\n") # In đường kẻ phân cách dưới và xuống dòng

    for i, s in enumerate(summary, 1): # Duyệt danh sách tóm tắt kèm số thứ tự
        print(f"{i}. {s}") # In số thứ tự và nội dung câu XML


# ======================================================
# 6 MAIN
# ======================================================
if __name__ == "__main__": # Điểm bắt đầu của chương trình
  
    TEXT_DIR = "DUC_TEXT/train"
    # r"..." là raw string: không cần escape \ (hữu ích cho path Windows)
    SUM_DIR  = "DUC_SUM"

    doc_names = sorted(os.listdir(SUM_DIR))[:3] # Lấy danh sách tên file trong thư mục SUM và chọn 3 file đầu

    for doc in doc_names: # Lặp qua từng tên tài liệu đã chọn
        text_path = os.path.join(TEXT_DIR, doc) # Tạo đường dẫn đầy đủ đến file gốc
        sum_path = os.path.join(SUM_DIR, doc) # Tạo đường dẫn đầy đủ đến file tóm tắt chuẩn

        if os.path.exists(text_path) and os.path.exists(sum_path): # Kiểm tra nếu cả 2 file đều tồn tại
            summarize(doc, text_path, sum_path) # Gọi hàm tóm tắt cho tài liệu đó