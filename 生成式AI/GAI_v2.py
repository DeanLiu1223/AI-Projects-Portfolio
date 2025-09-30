
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
        # 迭代訓練資料，建立詞頻和相鄰詞組頻率
        for i in range(len(dataset) - 2): #-1
            word = dataset[i]
            nextWord = dataset[i + 1]
            # 每隔一段時間印出訓練進度
            if i % 100000 == 0:
                print('\r+ 建置Book : ' + str(i) + '/' + str(len(dataset) - 2), end='')

                #################### 更新單詞頻率#######################
                
            if word not in self.__book:
                self.__book[word] = 1
            else:
                self.__book[word] += 1
                
                
                
                #--------------------------------------------------------------
                
                ####################  更新相鄰詞組頻率  ####################

            if (word + nextWord) not in self.__book2:
                self.__book2[word+nextWord] = 1
            else:
                self.__book2[word+nextWord] += 1



                
                #--------------------------------------------------------------
   
               
        print('\r+ 建置Book : ' + str(i) + '/' + str(len(dataset) - 2), end='')
        print('\nFinish..')
        print('\nFinish..')

        # 第二次迭代訓練資料，計算凝結度 (互信息) 並建立詞彙表
        for i in range(len(dataset) - 2): #-1
            word = dataset[i]
            nextWord = dataset[i + 1]
            # 每隔一段時間印出計算凝結度進度
            if i % 100000 == 0:
                print('\r+ 計算凝結度 : ' + str(i) + '/' + str(len(dataset) - 2), end='')

                ####################  計算單詞的機率 p(a) =?  ####################
            
            p_a = self.__book[word] / self.__totalWords #分母再-1
                        
                        
                        
                #--------------------------------------------------------------            
                ####################  計算單詞的機率 p(b) =?      #################### 
            
            p_b = self.__book[nextWord] / self.__totalWords #分母再-1
            
                        
                #--------------------------------------------------------------
                ##################### 計算相鄰詞組的機率 p(a&b)=?  ####################
            
            p_ab = self.__book2[word + nextWord] / (self.__totalWords -1)
                        
                        
                #--------------------------------------------------------------
                #####################  計算互信息 ( MI )####################
            
            MI = np.log(p_ab / (p_a * p_b))
                        
                        
                        
                #--------------------------------------------------------------
            # 如果互信息高於閾值或詞組出現頻率高於一定值，則加入詞彙表

            if MI >= self.__MIScore:
                self.__vocab.append(word + nextWord)
            if self.__book2[word + nextWord] > 200:
                self.__vocab.append(word + nextWord)

        print('\r+ 計算凝結度 : ' + str(i) + '/' + str(len(dataset) - 2), end='')
        print('\nFinish..')
        # 對詞彙表進行去重
        self.__vocab = list(np.unique(self.__vocab))

    def addVocab(self, vocabs):
        ##################### # 將新的詞彙加入詞彙表 ##################### 
        vocabs = list(vocabs)
        for i in range(len(vocabs)):
            self.__vocab.append(vocabs[i])


            
        #--------------------------------------------------------------
        ##################### # 對詞彙表進行去重  ##################### 
        
        self.__vocab = list(np.unique(self.__vocab))

    
        #--------------------------------------------------------------
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
        self.__dictionary = dict()
        
    def getBook(self):
        return super().getBook()
    def getBook2(self):
        return super().getBook2()
    def getVocab(self):
        return super().getVocab()
    def getDict(self):
        return self.__dictionary

    def train(self, dataset):
        # 調用父類的 train 方法來建立詞頻和詞彙表
        super().train(dataset)
        
        ####### 把 token id 列表新增至 dictionary############## 
        
        # self.__dictionary = list(super().getBook().keys())
        
        for i in range(len(dataset)):
            if dataset[i] not in self.__dictionary:
                self.__dictionary[dataset[i]] = len(self.__dictionary)
                
        vocabs = super().getVocab()
        for i in range(len(vocabs)):
            if vocabs[i] not in self.__dictionary:
                self.__dictionary[vocabs[i]] = len(self.__dictionary)
            
        
        
        #--------------------------------------------------------  
    def tokenize(self,prompt):
        ####### 將輸入的prompt映射為 token id 的 list##############
        # ans = []
        
        # for i in range(len(prompt)):
        #     ans.append(self.__dictionary.index(prompt[i]))    
        
        # return ans
        
        return list(map(lambda x: self.__dictionary[x], prompt))
        
        
        
        
        #--------------------------------------------------------  
    def token2Str(self,tokensList):
        ######### 將token id 的 list 解碼變成 字串##############
        # ans = []
        
        # for i in range(len(tokensList)):
        #     ans.append(self.__dictionary[ tokensList[i] ])  
        
        # return ans
        
        
        keys = np.array(list(self.__dictionary.keys()))
        
        return ("".join(keys[np.array(tokensList)]))
        
        
        
        #--------------------------------------------------------      
    def addVocab(self, vocabs):
        # 調用父類的 addVocab 方法來添加新的詞彙
        super().addVocab(vocabs)
    
    def split(self, prompt):
        ans = []
        skip = 0
        # 獲取父類的詞彙表
        vocab = super().getVocab()
        # 根據詞彙長度降序排列，以便優先匹配較長的詞彙 : table= ???
        
        ###########################  Code   ###################
        
        # table = sorted(np.unique(list(map(len, vocab))), reverse=True)

        table = np.unique(list(map(len, vocab)))[::-1]

        #-------------------------------------------------------

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
                
              ######################  Code   #####################
              word = prompt[i: (i + table[j])]
              if word in vocab:
                  isFind = True
                  ans.append(word)
                  skip += (table[j] - 1)
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
# Block 3: 定義 quadGramBayesModel 類別 - 繼承自 books，用 Naiive Bays實現 考慮輸出前三個字的 語言模型
# ============================================================================


class quadGramBayesModel(books):
#
    def __init__(self, stop=['。', '？', '!', '/BOS', '/EOS']):
        super().__init__()
        self.__book3 = dict()
        self.__ibook = dict()
        self.__ibook2 = dict()
        self.__ibook3 = dict()
        
        # __stop: 定義生成文本時的停止標記
        self.__stop = stop

    def getBook(self):
        return super().getBook()
    def getBook2(self):
        return super().getBook2()
    def getBook3(self):
        return self.__book3


    
    def train(self, dataset):
        # 獲取父類 books 的詞頻字典
        book = self.getBook()
        book2 = self.getBook2() 
        book3 = self.__book3
        ibook = self.__ibook
        ibook2 = self.__ibook2
        ibook3 = self.__ibook3
        # 訓練模型，建立單詞、雙詞組、三詞組的頻率以及反向頻率字典
        for i in range(len(dataset) - 3):
            w = dataset[i]
            nw = dataset[i + 1]
            nnw = dataset[i + 2]
            nnnw = dataset[i + 3]
            # 更新單詞頻率
            if w not in book:
                book[w] = dict()
                book[w][nw] = 1
            else:
                if nw not in book[w]:
                    book[w][nw] = 1
                else:
                    book[w][nw] += 1

            # 更新雙詞組頻率
            if w not in book2:
                book2[w] = dict()
                book2[w][nnw] = 1
            else:
                if nnw not in book2[w]:
                    book2[w][nnw] = 1
                else:
                    book2[w][nnw] += 1

            # 更新三詞組頻率
            if w not in book3:
                book3[w] = dict()
                book3[w][nnnw] = 1
            else:
                if nnnw not in book3[w]:
                    book3[w][nnnw] = 1
                else:
                    book3[w][nnnw] += 1

            # 更新反向頻率字典 ibook (next_word -> previous_word)
            if nw not in ibook:
                ibook[nw] = dict()
                ibook[nw][w] = 1
            else:
                if w not in ibook[nw]:
                    ibook[nw][w] = 1
                else:
                    ibook[nw][w] += 1

            # 更新反向頻率字典 ibook2 (next_next_word -> previous_word)
            if nnw not in ibook2:
                ibook2[nnw] = dict()
                ibook2[nnw][w] = 1
            else:
                if w not in ibook2[nnw]:
                    ibook2[nnw][w] = 1
                else:
                    ibook2[nnw][w] += 1

            # 更新反向頻率字典 ibook3 (next_next_next_word -> previous_word)
            if nnnw not in ibook3:
                ibook3[nnnw] = dict()
                ibook3[nnnw][w] = 1
            else:
                if w not in ibook3[nnnw]:
                    ibook3[nnnw][w] = 1
                else:
                    ibook3[nnnw][w] += 1
                    
                    
                    
                    

    def response(self, prompt, maxLength=100):


        book = self.getBook()
        ibook=self.__ibook
        ibook2=self.__ibook2
        ibook3=self.__ibook3
        
        # 初始化回應，使用輸入的 prompt 的副本
        ans = prompt.copy()

        # 生成指定長度的回應
        for i in range(maxLength):
            # 獲取當前回應的最後3個詞
            words = ans[-1]
            bwords = ans[-2]
            bbwords = ans[-3]
            # 如果最後一個詞在模型的詞頻字典中
            if words in book:
                ############### 獲取下一個詞的頻率列表和詞彙列表#############
                
                
                wordList = list(book[words].keys()) # 有幾種不同的字

                
                #--------------------------------------------------------
                probs=[]
                for j in range(len(wordList)):
                    action=wordList[j]
                    
                    ############## 計算 P(action | words) ################
                    
                    pa = np.sum(list(book[action].values())) / 10000 # 字數

                    

                    #--------------------------------------------------------
                    ############# # 計算 P(words | action) ################

                    iCount = list(ibook[action].values())
                    p_d1a = ibook[action][words] / np.sum(iCount)


                    #--------------------------------------------------------         
                    ################### 計算 P(bwords | action)   ############
                    
                    iCount2 = list(ibook2[action].values())
                    
                    if bwords not in ibook2[action]:
                        p_d2a = 1e5
                    else:
                        p_d2a = ibook2[action][bwords] / np.sum(iCount2)
                    
                    
                    #--------------------------------------------------------

                    ################### 計算 P(bbwords | action) ###########
                    
                    iCount3 = list(ibook3[action].values())

                    if bbwords not in ibook3[action]:
                        p_d3a = 1e5
                    else:
                        p_d3a = ibook3[action][bbwords] / np.sum(iCount3)
 
                        
                    #--------------------------------------------------------
                    probs.append(pa*p_d1a*p_d2a*p_d3a)
                    
                    
                
                ####################### 計算下一個詞的機率分佈 (PMF)#########

                pmf = np.array(probs) / np.sum(probs)

                
                #--------------------------------------------------------
                ########################### 計算累積機率分佈 (CDF)#######

                cdf = np.cumsum(pmf)

                
                #--------------------------------------------------------
                
                ######################## 生成一個 0 到 1 之間的隨機數######

                select = np.random.uniform(0, 1)
                
                
                #--------------------------------------------------------
               ####################### # 根據 CDF 選擇下一個詞########

                pick = np.where(select < cdf)[0][0]
                
                
                
                #--------------------------------------------------------             
                action = wordList[pick]   
                ans += action
                

                # 如果下一個詞是停止標記，則停止生成
                if action in self.__stop:
                    break
            else:
                break

        return ''.join(ans)

    def save(self, path='./LM'):
        super().save(path)
        np.save(path + 'b3.npy', self.__book3)
        np.save(path + 'ibook.npy', self.__ibook)
        np.save(path + 'ibook2.npy', self.__ibook2)
        np.save(path + 'ibook3.npy', self.__ibook3)

    def load(self, path='./LM'):
        super().load(path)
        self.__book3 = np.load(path + 'b3.npy', allow_pickle=True).item()
        self.__ibook = np.load(path + 'ibook.npy', allow_pickle=True).item()
        self.__ibook2 = np.load(path + 'ibook2.npy', allow_pickle=True).item()
        self.__ibook3 = np.load(path + 'ibook3.npy', allow_pickle=True).item()