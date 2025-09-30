
# -------------------- Block 1: 載入函式庫 --------------------
import pandas as pd
import numpy as np
import GAI_v1 as gai

# -------------------- Block 2: 讀取資料並進行初步處理 --------------------
# 讀取 CSV 檔案的前 100000 行
reader = pd.read_csv('Gossiping-QA-Dataset-2_0.csv')[0:10000]

# 移除包含 NaN 的行
reader = reader.dropna()
# 移除重複的行
reader = reader.drop_duplicates()
# 轉換為 NumPy 陣列
reader = reader.to_numpy()


# -------------------- Block 3: 建立資料流 (Stream) --------------------
# 在問題和答案之間加入特殊標記，並將所有資料連接成一個字串


stream = '/BOS' + reader[:, 0] + '/SEP' + reader[:, 1] + '/EOS'
stream = ''.join(stream)



# -------------------- Block 4: 訓練 Tokenizer --------------------
# 創建 Tokenizer 物件
t = gai.tokenizer()
# 使用資料流訓練 Tokenizer
t.train(stream)
# 添加特殊標記到 Tokenizer 的詞彙表
t.addVocab(['/BOS', '/SEP', '/EOS'])

# -------------------- Block 5: 分詞並訓練 BiGram 語言模型 --------------------
# 將資料流分詞
tokens = t.split(stream)
# 創建 BiGram 語言模型物件
LM = gai.BiGramModel()
# 使用分詞後的結果訓練 BiGram 語言模型
LM.train(tokens)

# -------------------- Block 6: 保存模型 --------------------
# 保存 BiGram 語言模型
LM.save()
# 保存 Tokenizer
t.save()

# -------------------- Block 7: 載入模型 --------------------
# 創建新的 Tokenizer 和 BiGram 語言模型物件
t = gai.tokenizer()
LM = gai.BiGramModel()

# 載入已保存的 Tokenizer
t.load()
LM.load()

# -------------------- Block 8: 產生回應 (Prompt Processing) --------------------
# 設定提示
prompt = '早餐吃什麼'
# 在提示後加入分隔符號
prompt += '/SEP'

# -------------------- Block 9: 分詞提示並產生回應  --------------------
# 將提示分詞
tokens = t.split(prompt)
res = LM.response(tokens)
print(res)
