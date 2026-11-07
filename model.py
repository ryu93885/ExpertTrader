import torch
import torch.nn as nn
import torchvision.models as models

# 💡 B案: 銘柄埋め込み(symbol embedding)のための、銘柄名↔ID対応表。
# portfolio_trading_bot.py の TARGET_SYMBOLS・train_portfolio_rl.py の symbols 等、
# 既存のポートフォリオ関連スクリプトで使われている並び順と完全に一致させている。
# 学習側・ライブ側で万が一この対応がズレると、モデルが別の銘柄のつもりで
# 予測してしまう(気づきにくい形で結果が壊れる)ため、この辞書を唯一の正とし、
# 他のファイルは必ずここから SYMBOL_TO_ID をインポートして使う。
SYMBOL_LIST = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
SYMBOL_TO_ID = {sym: i for i, sym in enumerate(SYMBOL_LIST)}
NUM_SYMBOLS = len(SYMBOL_LIST)


class MultimodalFXmodel(nn.Module):

    def __init__(self,num_tabular_features = 32,num_classes = 3,num_symbols = NUM_SYMBOLS,symbol_embed_dim = 8):
        """
        Multimodal Multitask FX Forecasting Model

        Args:
            num_tabular_features (int): Number of numerical features input from a CSV file
            num_classes (int): Number of classes for classification tasks (assumes three classes: Long=1, Hold=0, Short=-1)
            num_symbols (int): 銘柄埋め込みの語彙数(対応する銘柄の総数)
            symbol_embed_dim (int): 銘柄埋め込みベクトルの次元数
        """
        super(MultimodalFXmodel,self).__init__()

        # 💡 B案: 銘柄埋め込み層を追加。単一の共有モデルでも、銘柄ごとの値動きの
        # 性質の違い(例: GOLDの高ボラティリティ・トレンド性 vs JPYクロスの
        # レンジ傾向)を学習で条件付けできるようにする。
        self.symbol_embedding = nn.Embedding(num_symbols, symbol_embed_dim)
        #=======================================
        #1.vision expert
        #=======================================
        
        self.cnn = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

        #delete the last classifier
        cnn_out_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.cnn_compressor = nn.Sequential(
            nn.Linear(cnn_out_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


        #=======================================
        #data expert
        #=======================================
        tabular_out_features = 96
        self.gru = nn.GRU(
            input_size = num_tabular_features,
            hidden_size = tabular_out_features,
            num_layers = 2,
            batch_first = True,
            dropout = 0.2
        )

        #attention機構の追加
        self.attention = TemporalAttention(tabular_out_features)
        #Integration and Output
        # 3.Integration and Output
        #Integration and Output

        

        # 💡 B案: 銘柄埋め込みベクトルを結合特徴量に追加
        fused_in_features  = 64 + tabular_out_features + symbol_embed_dim

        self.fusion = nn.Sequential(
            nn.Linear(fused_in_features,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256,128),
            nn.ReLU()
        )

        #output headA:trading signal(classifier:Long/hold/short)
        self.classification_head = nn.Linear(128,num_classes)

        #output headB:risk prediction
        self.regression_head = nn.Linear(128,1)


    def forward(self,image_x,tabular_x,symbol_id,return_attention = False):
        """
        Args:
            symbol_id (torch.LongTensor): shape (batch,)。SYMBOL_TO_ID による銘柄ID。
                学習側・ライブ側で必ず同じ対応表(SYMBOL_TO_ID)を使って生成すること。
        """

        #forward propagation
        #feed the image data through a CNN
        cnn_features = self.cnn(image_x)
        cnn_features = self.cnn_compressor(cnn_features)

        gru_out,hidden = self.gru(tabular_x)
        tabular_features,attn_weights = self.attention(gru_out)

        # 💡 B案: 銘柄埋め込みベクトルを取得
        symbol_features = self.symbol_embedding(symbol_id)

        #concatenate the two features horizontally
        combined = torch.cat((cnn_features,tabular_features,symbol_features),dim = 1)

        #through the bonding layer
        fused_features = self.fusion(combined)

        #output two answers
        class_out = self.classification_head(fused_features)
        risk_out  = self.regression_head(fused_features)

        if return_attention:
            return class_out,risk_out,attn_weights

        return class_out,risk_out
    

class TemporalAttention(nn.Module):
    def __init__(self,hidden_dim):
        super(TemporalAttention,self).__init__()
        self.attention = nn.Linear(hidden_dim , 1)

    def forward(self,gru_outputs):
        #各ステップの重要度を計算
        attn_scores  = self.attention(gru_outputs)
        #Softmax関数をつかい、確率へ返還
        attn_weights = torch.softmax(attn_scores,dim=1)
        #重みをGRUの出力にかけ合わせ、過去のデータの加重平均を計算
        context_vector = torch.sum(gru_outputs * attn_weights,dim = 1)

        return context_vector,attn_weights

if __name__ == "__main__":
    dummy_num_features = 32
    seq_length = 40
    model = MultimodalFXmodel(num_tabular_features = dummy_num_features)

    #dummy data
    dummy_images = torch.randn(32,3,224,224)
    dummy_tabular = torch.randn(32,seq_length,dummy_num_features)
    dummy_symbol_id = torch.randint(0, NUM_SYMBOLS, (32,))

    #feed the data into the model
    out_class,out_risk = model(dummy_images,dummy_tabular,dummy_symbol_id)

    print("===model structure the successful")
    print(f"classifier output size :{out_class.shape}")
    print(f"regressor output size:{out_risk.shape}")
