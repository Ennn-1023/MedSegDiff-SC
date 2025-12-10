import os
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# --- 設定路徑 ---
# 原始資料路徑 (根據您之前的觀察設定)
RAW_DATA_PATH = r"raw_data/PH2Dataset/PH2 Dataset images" 
# 輸出目標路徑
OUTPUT_PATH = r"data/PH2"

# --- 參數設定 ---
IMG_SIZE = 256
SEED = 42

# 設定切分比例 (總和必須為 1.0)
RATIOS = {
    'Train': 0.85,  # 70% 用於訓練
    'Test':  0.15   # 20% 用於最終測試
}

def process_ph2():
    # 設定隨機種子以確保每次切分結果一致
    random.seed(SEED)
    
    # 1. 檢查原始資料路徑
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ 錯誤：找不到原始資料路徑: {RAW_DATA_PATH}")
        return

    # 2. 建立輸出資料夾結構 (Train, Val, Test)
    for split in RATIOS.keys():
        os.makedirs(os.path.join(OUTPUT_PATH, split, 'Image'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, split, 'Mask'), exist_ok=True)

    # 3. 搜尋病歷資料夾
    raw_path = Path(RAW_DATA_PATH)
    # 只抓取 IMD 開頭的資料夾
    cases = [d for d in raw_path.iterdir() if d.is_dir() and d.name.startswith("IMD")]
    
    if not cases:
        print("❌ 錯誤：找不到任何 IMD 資料夾。")
        return

    total_cases = len(cases)
    print(f"找到 {total_cases} 筆資料，準備進行切分...")
    
    # 打亂順序
    random.shuffle(cases)
    
    # 計算切分索引
    train_count = int(total_cases * RATIOS['Train'])
    # 剩下的全部給 Test (避免浮點數誤差導致少算)
    
    # 進行切片
    datasets = {
        'Train': cases[:train_count],
        'Test':  cases[train_count:],
    }

    # 4. 開始處理圖片
    for split, case_list in datasets.items():
        print(f"正在處理 {split} 集 ({len(case_list)} 筆)...")
        
        for case_dir in tqdm(case_list):
            case_id = case_dir.name  # 例如 IMD002
            
            # --- A. 找原圖 ---
            img_folder = case_dir / f"{case_id}_Dermoscopic_Image"
            img_files = list(img_folder.glob("*.bmp"))
            
            if not img_files:
                # 備用方案：直接在下一層找
                img_files = list(img_folder.glob(f"{case_id}.bmp"))
                if not img_files:
                    continue
            src_img_path = str(img_files[0])
            
            # --- B. 找遮罩 ---
            mask_folder = case_dir / f"{case_id}_lesion"
            mask_files = list(mask_folder.glob(f"{case_id}_lesion.bmp"))
            
            if not mask_files:
                continue
            src_mask_path = str(mask_files[0])
            
            # --- C. 讀取與縮放 ---
            img = cv2.imread(src_img_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                continue

            # 縮放到 256x256
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            
            # --- D. 存檔 ---
            save_img_path = os.path.join(OUTPUT_PATH, split, 'Image', f"{case_id}.png")
            save_mask_path = os.path.join(OUTPUT_PATH, split, 'Mask', f"{case_id}.png")
            
            cv2.imwrite(save_img_path, img_resized)
            cv2.imwrite(save_mask_path, mask_resized)

    print("\n✅ 資料預處理完成！")
    print(f"   Train: {len(datasets['Train'])} 筆")
    print(f"   Test : {len(datasets['Test'])} 筆")

if __name__ == "__main__":
    process_ph2()