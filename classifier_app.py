import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

DATA_DIR = "DUC_TEXT/test"

# ============== Kiểm tra đọc path lưu trữ dữ liệu có tồn tại không ==============

if not os.path.exists(DATA_DIR):
    print(f"Đường dẫn {DATA_DIR} không tồn tại. Vui lòng kiểm tra lại.")
else:
    print("Đã tìm thấy thông tin DATA DUC.")

# ============== Xử lý text ==============
def parse_s_tag(text: str):
    # Parse chuỗi XML
    elem = ET.fromstring(text.strip())
    
    return {
        "docid": elem.attrib.get("docid"),
        "num": int(elem.attrib.get("num")),
        "wdcount": int(elem.attrib.get("wdcount")),
        "content": elem.text.strip() if elem.text else ""
    }


# ============== features ==============

# True nếu chứa danh từ riêng (viết hoa)
def uppercase_word_feature(text):
    return int(any(word[0].isupper() for word in text.split()))

# Độ dài của câu
def sentence_length_feature(sentence, min_words = 10):
    return int(len(sentence.split()) > min_words) 

# Trích xuất đặc trưng
def extract_features(sentence):
    features = []
    features.append(sentence_length_feature(sentence))
    features.append(uppercase_word_feature(sentence))

    return features

# ============== Đọc thông tin dữ liệu test ==============

for root, dirs, files in os.walk(DATA_DIR):
    for file in tqdm(files, desc=f"Đang xử lý {os.path.basename(root)}"):
        path = os.path.join(root, file)
        try:
            print(f"path: {path}")
            sentences = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if(line.startswith("<s")):
                        data = parse_s_tag(line)
                        if data:
                            sentences.append(data)

            X = []
            for s in sentences:
                text = s["content"] if isinstance(s, dict) else s
                X.append(extract_features(sentence=text))

            # test code nên chỉ lấy 1 file
            break

        except Exception as e:
            print("ERROR:", e)
            continue
    
    break

