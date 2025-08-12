# Soft Prompt VLM Fine-tuning 教學

[English](README_EN.md) | 中文

這個項目展示了如何使用 **Soft Prompt** 技術來 fine-tune 視覺語言模型 (VLM)，這是一種參數高效的 fine-tuning 方法。

## 🎯 什麼是 Soft Prompt？

**Soft Prompt** 是一種可學習的連續向量，用來引導語言模型的行為。與傳統的文本 prompt（硬 prompt）不同，soft prompt 不是人類可讀的文字，而是通過訓練學習得到的數值向量。

### 🔍 傳統 Prompt vs Soft Prompt

#### 傳統 Prompt（硬 Prompt）
```
輸入: "請描述這張圖片："
圖像: [貓咪圖片]
輸出: "這是一隻可愛的貓咪..."
```

#### Soft Prompt
```
輸入: [可學習的向量] + 圖像特徵 + 文本
Soft Prompt: [0.23, -0.45, 0.67, ...] (10個可學習的數值)
圖像: [視覺特徵向量]
文本: "這是一隻..."
輸出: "這是一隻可愛的貓咪..."
```

### 🧠 核心概念

#### 1. **可學習的向量**
```python
# 在我們的代碼中
self.prompt_embeddings = nn.Parameter(
    torch.randn(prompt_length, hidden_size) * 0.02
)
```
- 這些向量在訓練過程中會不斷更新
- 不是固定的文字，而是數值參數
- 能夠捕捉到更複雜的模式

#### 2. **參數效率**
```python
# 只訓練這些參數：
# - Soft prompt embeddings: 10 × 768 = 7,680 參數
# - 視覺投影層: 768 × 768 = 589,824 參數
# 總計: ~600K 參數

# 而不是整個模型：
# - GPT-2: 124M 參數
# - CLIP: 150M 參數
```

#### 3. **凍結原始模型**
```python
def _freeze_models(self):
    """凍結原始模型的參數"""
    for param in self.vision_model.parameters():
        param.requires_grad = False  # 不更新
    for param in self.language_model.parameters():
        param.requires_grad = False  # 不更新
```

### 📊 工作原理

#### 步驟 1：創建 Soft Prompt
```python
# 初始化隨機向量
soft_prompt = [0.1, -0.2, 0.3, ...]  # 10個數值
```

#### 步驟 2：組合輸入
```python
# 將 soft prompt 與其他輸入組合
combined_input = [
    soft_prompt,      # 可學習的引導向量
    vision_features,  # 圖像特徵
    text_embeddings   # 文本特徵
]
```

#### 步驟 3：訓練更新
```python
# 在訓練過程中，soft prompt 會根據損失函數更新
loss = calculate_loss(output, target)
loss.backward()  # 計算梯度
optimizer.step() # 更新 soft prompt 參數
```

### 🎯 為什麼使用 Soft Prompt？

#### 1. **效率高**
- 只訓練少量參數（幾千到幾萬個）
- 訓練速度快，記憶體需求少
- 適合在有限資源下進行 fine-tuning

#### 2. **靈活性強**
- 可以學習到人類難以表達的複雜模式
- 能夠捕捉到數據中的細微差別
- 適應性強，可以針對不同任務優化

#### 3. **可解釋性**
- 雖然是數值向量，但可以分析其行為模式
- 可以視覺化 soft prompt 的影響
- 便於調試和優化

### 🎨 類比理解

想像 Soft Prompt 就像是：

1. **音樂指揮家**：指揮樂團（模型）如何演奏（生成）
2. **調味料**：為菜餚（模型輸出）添加特定風味
3. **濾鏡**：改變相機（模型）的拍攝風格

Soft Prompt 就是那個「無形的指導者」，通過學習到的數值來引導模型產生我們想要的行為，而不需要修改模型的核心結構。

### 主要優勢：
- **參數效率高**：只訓練少量參數（soft prompt + 投影層）
- **計算資源少**：不需要修改原始模型參數
- **訓練速度快**：大大減少了訓練時間
- **記憶體友好**：適合在有限資源下進行 fine-tuning

## 📁 項目結構

```
soft_prompt_VLM/
├── soft_prompt_finetune.py  # 主要訓練腳本
├── requirements.txt         # 依賴項
└── README.md               # 說明文件
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 準備數據

創建一個 JSON 格式的數據文件，包含圖像-文本對：

```json
[
    {
        "image_path": "path/to/image1.jpg",
        "text": "這是一張圖片的描述"
    },
    {
        "image_path": "path/to/image2.jpg", 
        "text": "另一張圖片的描述"
    }
]
```

### 3. 運行訓練

```bash
python soft_prompt_finetune.py
```

## 🔧 核心組件說明

### 1. SoftPrompt 類別

```python
class SoftPrompt(nn.Module):
    def __init__(self, prompt_length=10, hidden_size=768, vocab_size=50257):
        # 創建可學習的 prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, hidden_size) * 0.02)
```

- `prompt_length`：soft prompt tokens 的數量
- `hidden_size`：隱藏層維度
- `vocab_size`：詞彙表大小

### 2. VLMWithSoftPrompt 類別

整合了 CLIP 視覺編碼器和語言模型，使用 soft prompt 來引導模型：

```python
class VLMWithSoftPrompt(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32", 
                 language_model_name="gpt2", prompt_length=10):
        # 載入預訓練模型
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        
        # 創建 soft prompt
        self.soft_prompt = SoftPrompt(prompt_length, hidden_size, vocab_size)
        
        # 凍結原始模型參數
        self._freeze_models()
```

### 3. 訓練流程

1. **凍結原始模型**：CLIP 和 GPT-2 的參數保持不變
2. **只訓練 soft prompt**：更新可學習的 prompt embeddings
3. **視覺投影**：將視覺特徵投影到語言模型空間
4. **組合輸入**：`[soft_prompt, vision_features, text_embeddings]`

## 📊 模型架構

```
輸入: [圖像] + [文本]
    ↓
[CLIP Vision Encoder] → [Vision Projection]
    ↓
[Soft Prompt] + [Vision Features] + [Text Embeddings]
    ↓
[Language Model (GPT-2)]
    ↓
輸出: 生成的文本
```

## ⚙️ 可調參數

### 模型參數
- `prompt_length`：soft prompt 長度（預設：10）
- `vision_model_name`：視覺模型（預設：openai/clip-vit-base-patch32）
- `language_model_name`：語言模型（預設：gpt2）

### 訓練參數
- `learning_rate`：學習率（預設：1e-4）
- `num_epochs`：訓練輪數（預設：3）
- `batch_size`：批次大小（預設：2）
- `max_length`：最大文本長度（預設：128）

## 💡 使用建議

### 1. 數據準備
- 確保圖像路徑正確
- 文本描述要清晰且相關
- 數據集大小建議 100-1000 個樣本

### 2. 參數調優
- 根據任務調整 `prompt_length`
- 根據數據集大小調整 `learning_rate`
- 根據 GPU 記憶體調整 `batch_size`

### 3. 訓練技巧
- 使用較小的學習率（1e-4 到 1e-5）
- 監控訓練損失，避免過擬合
- 定期保存檢查點

## 🔍 進階用法

### 自定義數據集

```python
# 創建自定義數據集
dataset = VLMDataset(
    data_path="your_data.json",
    tokenizer=model.tokenizer,
    max_length=128
)
```

### 載入預訓練的 Soft Prompt

```python
# 載入檢查點
checkpoint = torch.load('soft_prompt_vlm_checkpoint.pth')
model.soft_prompt.load_state_dict(checkpoint['soft_prompt_state_dict'])
```

### 推理使用

```python
# 設置為評估模式
model.eval()

# 進行推理
with torch.no_grad():
    outputs = model(images, text_ids)
```

## 🐛 常見問題

### Q: 訓練時出現記憶體不足錯誤
A: 減少 `batch_size` 或使用梯度累積

### Q: 模型收斂很慢
A: 調整學習率或增加 `prompt_length`

### Q: 生成的文本質量不好
A: 檢查數據質量，增加訓練數據，調整超參數

## 📚 參考資料

- [Soft Prompting](https://arxiv.org/abs/2104.08691)
- [CLIP: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request 來改進這個項目！

## 📄 授權

MIT License 