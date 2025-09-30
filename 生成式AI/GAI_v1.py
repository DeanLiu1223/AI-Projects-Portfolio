import numpy as np

# ============================================================================
# Block 1: 定義 books 類別 - 作為基礎模型，包含詞彙表和凝結度計算等功能
# ============================================================================
class books():

    def __init__(self):
        # __book: 儲存詞頻的字典 (word: count)
        self.__book = dict()
        # __book2: 儲存相鄰詞組頻率的字典 (word+nextWord: count)
        self.__book2 = dict()
        # __vocab: 儲存模型詞彙表的列表
        self.__vocab = []
        # __MIScore: 互信息 (Mutual Information) 的閾值，用於判斷是否加入詞彙
        self.__MIScore = 10
        # __totalWords: 訓練資料中的總詞數
        self.__totalWords = 0

    def getBook(self):
        return self.__book
    def getBook2(self):
        return self.__book2
    def getVocab(self):
        return self.__vocab

    def train(self, dataset):
        # 增加總詞數
        self.__totalWords += len(dataset)
        
        for i in range(len(dataset) - 1):
            word = dataset[i]
            nextWord = dataset[i + 1]

            # 每隔一段時間印出訓練進度
            if i % 100000 == 0:
                print('\r+ 建置Book : ' + str(i) + '/' + str(len(dataset) - 2), end='')

            # 更新單詞頻率(book)
          ###########################  Code   #####################
            
            if word not in self.__book:
                self.__book[word] = 1
            else:
                self.__book[word] += 1
          
          #--------------------------------------------------------
            # 更新相鄰詞組頻率(book2)
            
          ###########################  Code   #####################
          
            if word+nextWord not in self.__book2:
                self.__book2[word+nextWord] = 1
            else:
                self.__book2[word+nextWord] += 1 
          
          
          #--------------------------------------------------------
                

        print('\r+ 建置Book : ' + str(i) + '/' + str(len(dataset) - 2), end='\n')
        print('\nFinish..')

        # 第二次迭代訓練資料，計算凝結度 (互信息) 並建立詞彙表
        for i in range(len(dataset) - 1):
            word = dataset[i]
            nextWord = dataset[i + 1]

            # 每隔一段時間印出計算凝結度進度
            if i % 100000 == 0:
                print('\r+ 計算凝結度 : ' + str(i) + '/' + str(len(dataset) - 2), end='')

            # 計算單詞的機率:   p_a,p_b
            
          ###########################  Code   #####################
            
            p_a = self.__book[word]/(len(dataset)-1)
            p_b = self.__book[nextWord]/(len(dataset)-1)
          
          #--------------------------------------------------------
          
          
            # 計算相鄰詞組的機率    p_ab
            
          ###########################  Code   #####################
            
            p_ab = self.__book2[word+nextWord]/(len(dataset)-1)
          
          
          #--------------------------------------------------------

            # 計算互信息  MI
          ###########################  Code   #####################
            
            MI = np.log(p_ab/(p_a*p_b))
          
          #--------------------------------------------------------

            # 如果互信息高於閾值或詞組出現頻率高於一定值，則加入詞彙表
            if MI >= self.__MIScore:
                self.__vocab.append(word + nextWord)
            if self.__book2[word + nextWord] > 100:
                self.__vocab.append(word + nextWord)

        print('\r+ 計算凝結度 : ' + str(i) + '/' + str(len(dataset) - 2), end='')
        print('\nFinish..')
        
        # 對詞彙表進行除去重複
          ###########################  Code   #####################
            
        self.__vocab = list(np.unique(self.__vocab))
          
          #--------------------------------------------------------

    def addVocab(self, vocabs):
        # 將新的詞彙加入詞彙表
        
          ###########################  Code   #####################
        
        for i in range(len(vocabs)):
            self.__vocab.append(vocabs[i])
        
        
          #--------------------------------------------------------

        # 對詞彙表進行除去重複
          ###########################  Code   #####################
            
        self.__vocab = list(np.unique(self.__vocab))
          
          #--------------------------------------------------------

    def save(self, path='./'):
        # 保存詞頻字典
        np.save(path + 'b1.npy', self.__book)
        # 保存相鄰詞組頻率字典
        np.save(path + 'b2.npy', self.__book2)
        # 保存詞彙表
        np.save(path + 'v1.npy', self.__vocab)
        # 保存互信息閾值
        np.save(path + 'MI.npy', self.__MIScore)

    def load(self, path='./'):
        # 載入詞頻字典
        self.__book = np.load(path + 'b1.npy', allow_pickle=True).item()
        # 載入相鄰詞組頻率字典
        self.__book2 = np.load(path + 'b2.npy', allow_pickle=True).item()
        # 載入詞彙表
        self.__vocab = list(np.load(path + 'v1.npy', allow_pickle=True))
        # 載入互信息閾值
        self.__MIScore = np.load(path + 'MI.npy', allow_pickle=True).item()

# ============================================================================
# Block 2: 定義 tokenizer 類別 - 繼承自 books，用於將文本分割成詞符 (tokens)
# ============================================================================
class tokenizer(books):

    def __init__(self):
        super().__init__()

    def getBook(self):
        return super().getBook()
    def getBook2(self):
        return super().getBook2()
    def getVocab(self):
        return super().getVocab()

    def train(self, dataset):
        # 調用父類的 train 方法來建立詞頻和詞彙表
        super().train(dataset)

    def addVocab(self, vocabs):
        # 調用父類的 addVocab 方法來添加新的詞彙
        super().addVocab(vocabs)

    def split(self, prompt):
        ans = []
        skip = 0
        # 獲取父類的詞彙表
        vocab = super().getVocab()
        # 根據詞彙長度降序排列，以便優先匹配較長的詞彙 : table= ???
        
          ###########################  Code   #####################
        
        
        table = np.unique(list(map(len,vocab)))[::-1]
          
          #--------------------------------------------------------

        # 迭代輸入文本進行分詞
        for i in range(len(prompt)):
            # 每隔一段時間印出分詞進度
            if i % 100000 == 0:
                print('\r+ 分割中 : ' + str(i) + '/' + str(len(prompt) - 2), end='')

            isFind = False
            # 如果 skip 大於 0，表示前一個位置已經匹配到一個多字詞，跳過後續幾個字元
            if skip > 0:
                skip -= 1
                continue

            # 嘗試匹配不同長度的詞彙
            for j in range(len(table)):
                
          ###########################  Code   #####################
            
                if prompt[i:i+table[j]] in vocab:
                    ans.append(prompt[i:i+table[j]])
                    skip += (table[j]-1)
                    isFind = True
                    break

          #--------------------------------------------------------

            # 如果沒有匹配到任何詞彙，則將單個字元作為一個詞符
            if not isFind:
                ans.append(prompt[i])
        print('\r+ 分割中 : ' + str(i) + '/' + str(len(prompt) - 2), end='\n')
        return ans

    def save(self, path='./'):
        # 調用父類的 save 方法保存模型
        super().save(path)

    def load(self, path='./'):
        # 調用父類的 load 方法載入模型
        super().load(path)

# ============================================================================
# Block 3: 定義 BiGramModel 類別 - 繼承自 books，實現 BiGram 語言模型
# ============================================================================
class BiGramModel(books):
#
    def __init__(self, stop=['。', '？', '!', '/BOS', '/EOS']):
        super().__init__()
        # __stop: 定義生成文本時的停止標記
        self.__stop = stop

    def getBook(self):
        return super().getBook()
    def getBook2(self):
        return super().getBook2()


    def train(self, dataset):
        # 獲取父類的詞頻字典
        book = self.getBook()
#        book2=self.getBook2() # BiGram 模型通常只基於單詞頻率

        # 訓練 BiGram 模型，建立單詞之間的條件機率
        for i in range(len(dataset) - 1):
            
          ###########################  Code   #####################
          
          word = dataset[i]
          nextWord = dataset[i+1]
          
          if word not in book:
              book[word] = dict()
              book[word][nextWord] = 1
          else:
              if nextWord not in book[word]:
                  book[word][nextWord] = 1
              else:
                  book[word][nextWord] += 1   
          
          #--------------------------------------------------------

    def response(self, prompt, maxLength=100):
        # 獲取父類的詞頻字典
        book = self.getBook()
        # 初始化回應，以輸入的 prompt 的最後一個詞開始
        ans = prompt
            
        # 生成指定長度的回應
        for i in range(maxLength):
            # 獲取當前回應的最後一個詞
            words = ans[-1]

            # 如果最後一個詞在模型的詞頻字典中
            if words in book:
                # 獲取下一個詞的頻率列表和詞彙列表
                info = list(book[words].values())
                wordList = list(book[words].keys())

                # 計算下一個詞的機率分佈 (PMF)
          ###########################  Code   #####################
                         
                PMF = info/np.sum(info)
          
          #--------------------------------------------------------

                # 計算累積機率分佈 (CDF)
          ###########################  Code   #####################
            
                CDF = np.cumsum(PMF)
          
          #--------------------------------------------------------
                # 生成一個 0 到 1 之間的隨機數
          ###########################  Code   #####################
            
                pick = np.random.uniform(0,1)
          
          #--------------------------------------------------------
                # 根據 CDF 選擇下一個詞 pick=???
          ###########################  Code   #####################
            
                pick = np.where(pick<CDF)[0][0]
          
          #--------------------------------------------------------

                # 選擇下一個動作 (詞)
                action = wordList[pick]
                # 將下一個詞加入回應
                ans += action
                # 如果下一個詞是停止標記，則停止生成
                if action in self.__stop:
                    break
            else:
                break

        return ''.join(ans)

    def save(self, path='./LM'):
        super().save(path)

    def load(self, path='./LM'):
        super().load(path)