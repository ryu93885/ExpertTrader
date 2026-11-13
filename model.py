import torch
import torch.nn as nn
import torchvision.models as models

# 💡 追加(2026-11-13の改修): ResNet50バックボーンを丸ごとfine-tuningしており、
# 学習時間・GPUメモリの両面でコストが大きい。転移学習の一般的な知見では、CNNの
# 浅い層(conv1・layer1・layer2)はエッジ・テクスチャ等の汎用的な特徴を学習して
# おり、画像ドメインが変わっても比較的そのまま転用しやすい。一方、深い層
# (layer3・layer4)はImageNet(自然画像)に特化した高次の特徴を学習しているため、
# ローソク足チャート画像のような全く異なる画像ドメインでは、ある程度の再学習
# (fine-tuning)が必要になりやすい、というのが標準的な転移学習の考え方。
# この知見に基づき、複数の「どこまで凍結するか」の選択肢を用意し、学習時間と
# 精度のトレードオフを実際の学習で比較検証できるようにする。
# ⚠️ 本環境ではGPUでの実データ学習・比較ができないため、実際の精度への影響は
# 確証が無い(一般的な転移学習の知見に基づく推奨にとどまる)。"none"(従来通り)
# との比較を、実際の学習で行うことを推奨する。
RESNET_FREEZE_PRESETS = {
    "none": [],  # 従来通り全層をfine-tuning(基準点。学習コストは最大だが、
                 # 凍結による精度低下リスクも無い)
    "stem": ["conv1", "bn1"],  # 最初の畳み込み層のみ凍結。影響は最小限と予想される
    "early": ["conv1", "bn1", "layer1", "layer2"],  # 推奨の出発点。浅い層(汎用的な
                                                      # 特徴)を凍結し、ドメイン適応が
                                                      # 必要になりやすい深い層
                                                      # (layer3・layer4)のみ学習する
    "early_plus_layer3": ["conv1", "bn1", "layer1", "layer2", "layer3"],  # より積極的な
        # 凍結。学習コストはさらに下がるが、layer3もドメイン適応できなくなるため、
        # "early"より精度低下のリスクが高いと考えられる
}


def freeze_resnet_layers(resnet: nn.Module, preset: str) -> int:
    """
    resnetの指定したpreset名に対応する子モジュール(例: layer1・layer2)を
    requires_grad=Falseにする。戻り値は凍結したパラメータ数(ログ用)。
    """
    if preset not in RESNET_FREEZE_PRESETS:
        raise ValueError(f"未知のfreeze preset: '{preset}'. 選択肢: {list(RESNET_FREEZE_PRESETS.keys())}")
    frozen_names = RESNET_FREEZE_PRESETS[preset]
    frozen_param_count = 0
    for name, module in resnet.named_children():
        if name in frozen_names:
            for param in module.parameters():
                param.requires_grad = False
                frozen_param_count += param.numel()
    return frozen_param_count

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

    def __init__(self,num_tabular_features = 32,num_classes = 3,num_symbols = NUM_SYMBOLS,symbol_embed_dim = 8,
                 freeze_backbone = "none"):
        """
        Multimodal Multitask FX Forecasting Model

        Args:
            num_tabular_features (int): Number of numerical features input from a CSV file
            num_classes (int): Number of classes for classification tasks (assumes three classes: Long=1, Hold=0, Short=-1)
            num_symbols (int): 銘柄埋め込みの語彙数(対応する銘柄の総数)
            symbol_embed_dim (int): 銘柄埋め込みベクトルの次元数
            freeze_backbone (str): RESNET_FREEZE_PRESETSのいずれか。ResNet50バックボーンの
                どこまでを凍結するか("none"なら従来通り全層をfine-tuning)。
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

        # 💡 追加(2026-11-13の改修): freeze_backboneで指定した範囲のResNet50の層を
        # 凍結する("none"の場合は従来通り何も凍結しない)。凍結したパラメータ数を
        # 呼び出し元(train.py)がログへ残せるよう、インスタンス属性として保持する。
        self.frozen_backbone_param_count = freeze_resnet_layers(self.cnn, freeze_backbone)

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
