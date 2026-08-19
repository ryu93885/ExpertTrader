import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalFXmodel(nn.Module):
    
    def __init__(self,num_tabular_features = 32,num_classes = 3):
        """
        Multimodal Multitask FX Forecasting Model
        
        Args:
            num_tabular_features (int): Number of numerical features input from a CSV file
            num_classes (int): Number of classes for classification tasks (assumes three classes: Long=1, Hold=0, Short=-1)
        """
        super(MultimodalFXmodel,self).__init__()
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

        

        fused_in_features  = 64 + tabular_out_features

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


    def forward(self,image_x,tabular_x,return_attention = False):

        #forward propagation
        #feed the image data through a CNN
        cnn_features = self.cnn(image_x)
        cnn_features = self.cnn_compressor(cnn_features)

        gru_out,hidden = self.gru(tabular_x)
        tabular_features,attn_weights = self.attention(gru_out)



        #concatenate the two features horizontally
        combined = torch.cat((cnn_features,tabular_features),dim = 1)

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

    #feed the data into the model
    out_class,out_risk = model(dummy_images,dummy_tabular)

    print("===model structure the successful")
    print(f"classifier output size :{out_class.shape}")
    print(f"regressor output size:{out_risk.shape}")
