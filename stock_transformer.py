import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossStockAttention(nn.Module):
    """跨股票注意力機制 - 讓不同股票之間能夠學習關聯性"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, stock_mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and split into heads
        Q = self.q_linear(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if stock_mask is not None:
            scores = scores.masked_fill(stock_mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out(context), attention_weights

class StockTransformer(nn.Module):
    """
    多股票時間序列預測的Transformer模型
    - 支援跨股票學習
    - 預測未來多個時間點
    - 考慮股票間的相關性
    """
    def __init__(self, 
                 input_dim, 
                 num_stocks, 
                 d_model=256, 
                 nhead=8, 
                 num_layers=6, 
                 dropout=0.1,
                 prediction_horizon=336,  # 7天 * 48個30分鐘時間段
                 max_seq_len=1000):
        super(StockTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_stocks = num_stocks
        self.prediction_horizon = prediction_horizon
        
        # Feature projection
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # Stock embeddings - 讓每支股票有獨特的表示
        self.stock_embedding = nn.Embedding(num_stocks, d_model)
        self.stock_type_embedding = nn.Embedding(10, d_model)  # 股票類型嵌入 (ETF, 科技股等)
        
        # Positional encoding for time
        self.time_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 分鐘級別的週期性編碼
        self.minute_embedding = nn.Embedding(48, d_model)  # 一天48個30分鐘
        self.hour_embedding = nn.Embedding(24, d_model)
        self.day_embedding = nn.Embedding(7, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-stock attention
        self.cross_stock_attention = MultiHeadCrossStockAttention(d_model, nhead, dropout)
        
        # Stock relationship learning
        self.stock_relation_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Decoder for multi-step prediction
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, prediction_horizon)  # 預測未來336個時間點
        )
        
        # Price change predictor (輔助任務)
        self.price_change_predictor = nn.Linear(d_model, 3)  # 上漲、下跌、持平
        
        self.dropout = nn.Dropout(dropout)

    def create_time_features(self, batch_size, seq_len, device):
        """創建時間特徵"""
        # 假設這是30分鐘間隔的數據
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 分鐘級別 (0-47, 每天48個30分鐘區間)
        minute_positions = positions % 48
        
        # 小時級別 (0-23)
        hour_positions = (positions // 2) % 24
        
        # 星期級別 (0-6)
        day_positions = (positions // 48) % 7
        
        return minute_positions, hour_positions, day_positions

    def forward(self, src, stock_ids, attention_mask=None):
        """
        Args:
            src: [batch_size, seq_len, input_dim] - 輸入特徵
            stock_ids: [batch_size] - 股票ID
            attention_mask: [batch_size, seq_len] - 注意力遮罩
        """
        batch_size, seq_len, _ = src.shape
        device = src.device
        
        # 1. Feature projection
        x = self.feature_projection(src)  # [batch_size, seq_len, d_model]
        
        # 2. Add stock embeddings
        stock_emb = self.stock_embedding(stock_ids).unsqueeze(1)  # [batch_size, 1, d_model]
        stock_emb = stock_emb.expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
        x = x + stock_emb
        
        # 3. Add time embeddings
        minute_pos, hour_pos, day_pos = self.create_time_features(batch_size, seq_len, device)
        
        time_emb = self.time_embedding(torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1))
        minute_emb = self.minute_embedding(minute_pos)
        hour_emb = self.hour_embedding(hour_pos)
        day_emb = self.day_embedding(day_pos)
        
        x = x + time_emb + minute_emb + hour_emb + day_emb
        
        # 4. Layer normalization
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # 5. Transformer encoding
        if attention_mask is not None:
            # Convert attention mask to the format expected by transformer
            attention_mask = attention_mask.bool()
            src_key_padding_mask = ~attention_mask
        else:
            src_key_padding_mask = None
            
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 6. Cross-stock attention (讓不同股票學習彼此關聯)
        cross_attended, attention_weights = self.cross_stock_attention(encoded, encoded, encoded)
        
        # 7. Combine original and cross-attended features
        combined = encoded + cross_attended
        
        # 8. 提取最後時間步的特徵用於預測
        last_hidden = combined[:, -1, :]  # [batch_size, d_model]
        
        # 9. Multi-step prediction
        predictions = self.output_projection(last_hidden)  # [batch_size, prediction_horizon]
        
        # 10. 輔助任務：價格變化方向預測
        price_direction = self.price_change_predictor(last_hidden)  # [batch_size, 3]
        
        return {
            'predictions': predictions,  # 未來336個時間點的價格
            'price_direction': price_direction,  # 價格變化方向
            'attention_weights': attention_weights,
            'hidden_states': last_hidden
        }

    def predict_future_prices(self, src, stock_ids, current_price, steps=336):
        """
        預測未來價格序列
        
        Args:
            src: 歷史數據 [batch_size, seq_len, input_dim]
            stock_ids: 股票ID [batch_size]
            current_price: 當前價格 [batch_size]
            steps: 預測步數 (預設7天*48 = 336步)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(src, stock_ids)
            
            # 將預測的變化轉換為實際價格
            price_changes = outputs['predictions'][:, :steps]  # [batch_size, steps]
            
            # 累積價格變化
            current_price = current_price.unsqueeze(1)  # [batch_size, 1]
            future_prices = current_price + torch.cumsum(price_changes, dim=1)
            
            return {
                'future_prices': future_prices,
                'price_direction': torch.softmax(outputs['price_direction'], dim=-1),
                'attention_weights': outputs['attention_weights']
            }

class StockDataset(torch.utils.data.Dataset):
    """股票數據集類"""
    def __init__(self, features, targets, stock_ids, sequence_length=60, prediction_horizon=336):
        self.features = features
        self.targets = targets
        self.stock_ids = stock_ids
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # 輸入序列
        x = self.features[idx:idx + self.sequence_length]
        
        # 未來價格目標 (接下來336個時間點)
        y = self.targets[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon]
        
        # 股票ID
        stock_id = self.stock_ids[idx + self.sequence_length - 1]  # 使用序列最後一個時間點的股票ID
        
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.LongTensor([stock_id])

def create_stock_relationship_matrix(stock_symbols):
    """創建股票關係矩陣（簡化版本）"""
    num_stocks = len(stock_symbols)
    relationship_matrix = torch.eye(num_stocks)
    
    # 簡單的關係定義（實際應用中可以使用更複雜的方法）
    for i, symbol1 in enumerate(stock_symbols):
        for j, symbol2 in enumerate(stock_symbols):
            if i != j:
                # ETF 之間的關係
                if symbol1.lower() in ['spy', 'qqq', 'iwm'] and symbol2.lower() in ['spy', 'qqq', 'iwm']:
                    relationship_matrix[i, j] = 0.8
                # 科技股之間的關係
                elif symbol1.lower() in ['aapl', 'msft', 'googl'] and symbol2.lower() in ['aapl', 'msft', 'googl']:
                    relationship_matrix[i, j] = 0.7
                else:
                    relationship_matrix[i, j] = 0.1
                    
    return relationship_matrix