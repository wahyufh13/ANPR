# # utilFFMPEG.py
# import re
# import cv2
# import string
# from paddleocr import PaddleOCR

# # Initialize PaddleOCR sekali saja
# ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False, use_space_char=True, rec_char_type='en') 

# # Mapping dictionaries untuk koreksi OCR
# dict_char_to_int = { 'I': '1','J': '3', 'A': '4', 'G': '6', 'S': '5'}
# dict_int_to_char = { '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# # Bersihkan simbol selain huruf/angka
# def clean_text(text):
#     return re.sub(r'[^A-Z0-9]', '', text.upper())

# def license_complies_format(text): #
#     """
#     Validasi format plat Indonesia:
#     1-2 huruf + 1-4 angka + 1-3 huruf
#     """
#     text = text.upper().replace(' ', '')

#     if len(text) < 3 or len(text) > 10:
#         return False
    
#     # Pattern regex untuk format plat Indonesia
#     # ^[A-Z]{1,2} : 1-2 huruf di depan
#     # [0-9]{1,4}  : 1-4 angka di tengah  
#     # [A-Z]{1,3}$ : 1-3 huruf di belakang
#     pattern = r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$'
    
#     if not re.match(pattern, text):
#         return False
    
#    # Validasi tambahan: cek posisi karakter
#     # Temukan posisi angka pertama dan terakhir
#     first_digit_pos = -1
#     last_digit_pos = -1
    
#     for i, char in enumerate(text):
#         if char.isdigit():
#             if first_digit_pos == -1:
#                 first_digit_pos = i
#             last_digit_pos = i
    
#     # Pastada ada angka
#     if first_digit_pos == -1:
#         return False
    
#     # Pastikan karakter sebelum angka pertama adalah huruf
#     if first_digit_pos == 0:
#         return False
    
#     for i in range(first_digit_pos):
#         if not text[i].isalpha():
#             return False
    
#     # Pastikan karakter setelah angka terakhir adalah huruf
#     if last_digit_pos == len(text) - 1:
#         return False
        
#     for i in range(last_digit_pos + 1, len(text)):
#         if not text[i].isalpha():
#             return False
    
#     # Pastikan bagian tengah (antara huruf depan dan belakang) hanya angka
#     for i in range(first_digit_pos, last_digit_pos + 1):
#         if not text[i].isdigit():
#             return False
    
#     return True



# def format_license(text): #
#     """
#     Koreksi format plat nomor Indonesia agar huruf/angka benar
#     """
#     text = text.upper().replace(' ', '')

#     if len(text) < 3:
#         return text
    
#     # Temukan posisi angka pertama dan terakhir
#     first_digit_pos = -1
#     last_digit_pos = -1
    
#     for i, char in enumerate(text):
#         if char.isdigit() or char in dict_char_to_int:
#             if first_digit_pos == -1:
#                 first_digit_pos = i
#             last_digit_pos = i
    
#     if first_digit_pos == -1:
#         return text  # Tidak ada angka, kembalikan apa adanya
    
#     # Pisahkan bagian-bagian
#     front_letters = text[:first_digit_pos]
#     middle_digits = text[first_digit_pos:last_digit_pos + 1]
#     back_letters = text[last_digit_pos + 1:]
    
#     # Koreksi huruf depan (paksa jadi huruf)
#     fixed_front = ''
#     for char in front_letters:
#         if char in dict_int_to_char:
#             fixed_front += dict_int_to_char[char]
#         elif char.isdigit():
#             # Jika ada angka di depan, coba konversi ke huruf
#             fixed_front += dict_int_to_char.get(char, char)
#         else:
#             fixed_front += char
    
#     # Koreksi angka tengah (paksa jadi angka)
#     fixed_middle = ''
#     for char in middle_digits:
#         if char in dict_char_to_int:
#             fixed_middle += dict_char_to_int[char]
#         elif char.isalpha() and char not in dict_char_to_int:
#             # Jika ada huruf yang tidak bisa dikonversi, coba mapping manual
#             if char == 'B': fixed_middle += '8'
#             elif char == 'Z': fixed_middle += '2'
#             elif char == 'T': fixed_middle += '7'
#             else: fixed_middle += char  # Biarkan apa adanya jika tidak ada mapping
#         else:
#             fixed_middle += char
    
#     # Koreksi huruf belakang (paksa jadi huruf)
#     fixed_back = ''
#     for char in back_letters:
#         if char in dict_int_to_char:
#             fixed_back += dict_int_to_char[char]
#         elif char.isdigit():
#             # Jika ada angka di belakang, coba konversi ke huruf
#             fixed_back += dict_int_to_char.get(char, char)
#         else:
#             fixed_back += char
    
#     result = fixed_front + fixed_middle + fixed_back
    
#     # Validasi ulang hasil koreksi
#     if license_complies_format(result):
#         return result
#     else:
#         return text  # Jika tidak valid setelah koreksi, kembalikan teks asli



# def read_license_plate(license_plate_crop): #
#     if license_plate_crop is None or license_plate_crop.size == 0:
#         return None, None

#     h, w, _ = license_plate_crop.shape

#     crop = license_plate_crop[:int(h * 0.85), :]
#     # print("Crop size for OCR:", crop.shape)
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#     # Preprocess ringan dulu
#     enhanced = cv2.convertScaleAbs(gray, alpha=1.4, beta=15)

#     try:
#         result = ocr.ocr(enhanced, cls=False)
#     except:
#         return None, None

#     if not result or not result[0]:
#         return None, None

#     for box, (text, score) in result[0]:
#         if score < 0.6:
#             continue

#         text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
#         text = format_license(text)

#         if license_complies_format(text):
#             return text, score

#     return None, None
# utilFFMPEG.py
import re
import cv2
import string
from paddleocr import PaddleOCR

# Initialize PaddleOCR sekali saja
ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False, use_space_char=True, rec_char_type='en')

# Mapping dictionaries untuk koreksi OCR
dict_char_to_int = { 'I': '1','J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = { '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# Bersihkan simbol selain huruf/angka
def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def license_complies_format(text): #
    """
    Validasi format plat Indonesia:
    1-2 huruf + 1-4 angka + 1-3 huruf
    """
    text = text.upper().replace(' ', '')

    if len(text) < 3 or len(text) > 10:
        return False
    
    # Pattern regex untuk format plat Indonesia
    # ^[A-Z]{1,2} : 1-2 huruf di depan
    # [0-9]{1,4}  : 1-4 angka di tengah  
    # [A-Z]{1,3}$ : 1-3 huruf di belakang
    pattern = r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$'
    
    if not re.match(pattern, text):
        return False
    
   # Validasi tambahan: cek posisi karakter
    # Temukan posisi angka pertama dan terakhir
    first_digit_pos = -1
    last_digit_pos = -1
    
    for i, char in enumerate(text):
        if char.isdigit():
            if first_digit_pos == -1:
                first_digit_pos = i
            last_digit_pos = i
    
    # Pastada ada angka
    if first_digit_pos == -1:
        return False
    
    # Pastikan karakter sebelum angka pertama adalah huruf
    if first_digit_pos == 0:
        return False
    
    for i in range(first_digit_pos):
        if not text[i].isalpha():
            return False
    
    # Pastikan karakter setelah angka terakhir adalah huruf
    if last_digit_pos == len(text) - 1:
        return False
        
    for i in range(last_digit_pos + 1, len(text)):
        if not text[i].isalpha():
            return False
    
    # Pastikan bagian tengah (antara huruf depan dan belakang) hanya angka
    for i in range(first_digit_pos, last_digit_pos + 1):
        if not text[i].isdigit():
            return False
    
    return True



def format_license(text): #
    """
    Koreksi format plat nomor Indonesia agar huruf/angka benar
    """
    text = text.upper().replace(' ', '')

    if len(text) < 3:
        return text
    
    # Temukan posisi angka pertama dan terakhir
    first_digit_pos = -1
    last_digit_pos = -1
    
    for i, char in enumerate(text):
        if char.isdigit() or char in dict_char_to_int:
            if first_digit_pos == -1:
                first_digit_pos = i
            last_digit_pos = i
    
    if first_digit_pos == -1:
        return text  # Tidak ada angka, kembalikan apa adanya
    
    # Pisahkan bagian-bagian
    front_letters = text[:first_digit_pos]
    middle_digits = text[first_digit_pos:last_digit_pos + 1]
    back_letters = text[last_digit_pos + 1:]
    
    # Koreksi huruf depan (paksa jadi huruf)
    fixed_front = ''
    for char in front_letters:
        if char in dict_int_to_char:
            fixed_front += dict_int_to_char[char]
        elif char.isdigit():
            # Jika ada angka di depan, coba konversi ke huruf
            fixed_front += dict_int_to_char.get(char, char)
        else:
            fixed_front += char
    
    # Koreksi angka tengah (paksa jadi angka)
    fixed_middle = ''
    for char in middle_digits:
        if char in dict_char_to_int:
            fixed_middle += dict_char_to_int[char]
        elif char.isalpha() and char not in dict_char_to_int:
            # Jika ada huruf yang tidak bisa dikonversi, coba mapping manual
            if char == 'B': fixed_middle += '8'
            elif char == 'Z': fixed_middle += '2'
            elif char == 'T': fixed_middle += '7'
            else: fixed_middle += char  # Biarkan apa adanya jika tidak ada mapping
        else:
            fixed_middle += char
    
    # Koreksi huruf belakang (paksa jadi huruf)
    fixed_back = ''
    for char in back_letters:
        if char in dict_int_to_char:
            fixed_back += dict_int_to_char[char]
        elif char.isdigit():
            # Jika ada angka di belakang, coba konversi ke huruf
            fixed_back += dict_int_to_char.get(char, char)
        else:
            fixed_back += char
    
    result = fixed_front + fixed_middle + fixed_back
    
    # Validasi ulang hasil koreksi
    if license_complies_format(result):
        return result
    else:
        return text  # Jika tidak valid setelah koreksi, kembalikan teks asli



# def read_license_plate(license_plate_crop): #
#     if license_plate_crop is None or license_plate_crop.size == 0:
#         return None, None

#     h, w, _ = license_plate_crop.shape

#     crop = license_plate_crop[:int(h * 0.85), :]
#     # print("Crop size for OCR:", crop.shape)
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#     # Preprocess ringan dulu
#     enhanced = cv2.convertScaleAbs(gray, alpha=1.4, beta=15)

#     try:
#         result = ocr.ocr(enhanced, cls=False)
#     except:
#         return None, None

#     if not result or not result[0]:
#         return None, None

#     for box, (text, score) in result[0]:
#         if score < 0.6:
#             continue

#         text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
#         text = format_license(text)

#         if license_complies_format(text):
#             return text, score

#     return None, None

def read_license_plate(license_plate_crop, min_score=0.45, debug=True):
    if license_plate_crop is None or license_plate_crop.size == 0:
        if debug:
            print("   ⚠️ crop kosong / None")
        return None, None

    h, w = license_plate_crop.shape[:2]
    if h < 5 or w < 5:
        if debug:
            print(f"   ⚠️ crop terlalu kecil: {w}x{h}")
        return None, None

    crop = license_plate_crop[:int(h * 0.85), :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale kalau plat kecil
    if gray.shape[1] < 200:
        scale = 200 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    variants = [
        ("alpha1.4", cv2.convertScaleAbs(gray, alpha=1.4, beta=15)),
        ("gray",     gray),
        ("alpha1.8", cv2.convertScaleAbs(gray, alpha=1.8, beta=0)),
    ]

    best_text, best_score = None, 0.0

    for vname, img in variants:
        try:
            result = ocr.ocr(img, cls=False)
        except Exception as e:
            if debug:
                print(f"   ❌ OCR exception [{vname}]: {e}")
            continue

        if not result or not result[0]:
            if debug:
                print(f"   [{vname}] (tidak ada teks terdeteksi)")
            continue

        for box, (text, score) in result[0]:
            raw = text
            clean = text.upper().replace(' ', '').replace('-', '').replace('.', '')
            fixed = format_license(clean)
            valid = license_complies_format(fixed)

            if debug:
                print(f"   🔤 [{vname}] RAW='{raw}' | score={score:.3f} "
                      f"| clean='{clean}' | fixed='{fixed}' | valid={valid}")

            if score < min_score:
                continue
            if valid and score > best_score:
                best_text, best_score = fixed, score

    if debug:
        print(f"   ➡️ HASIL AKHIR: {best_text} (score={best_score:.3f})")

    return (best_text, best_score) if best_text else (None, None)
