import torch
from torch.utils.data import DataLoader,dataloader
import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm

from portfolio_dataset import PortfolioFXDataset
from model import MultimodalFXmodel

logging.basicConfig(level=logging.INFO,format = "%(asctime)s [%(levelname)s] %(message)s")
def collate_fn_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return dataloader.default_collate(batch)

def main():
    logging.info("precompute_portfolio.py ver208.2")
    symbols = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
    mode = "short"
    merged_csv = "portfolio_merged_data_train.csv"
    img_dir = "images"
    model_path = f"saved_model/final_best_model_{mode}.pth"
    output_csv = "portfolio_merged_data_train_with_preds.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"デバイス:{device}")

    #1.preparing dataset and dataloader
    dataset = PortfolioFXDataset(merged_csv_path=merged_csv,base_img_dir = img_dir,symbols = symbols,mode = mode)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, collate_fn=collate_fn_drop_none)

    #2.loading model
    input_dim = len(dataset.feature_cols[symbols[0]])
    model = MultimodalFXmodel(num_tabular_features=input_dim).to(device)
    model.load_state_dict(torch.load(model_path,map_location = device,weights_only=False)["model_state_dict"])
    model.eval()

    #3.listing result of prediction
    all_probs = {sym:[] for sym in symbols}
    all_risks = {sym:[] for sym in symbols}

    logging.info("全データの事前推論を開始(バッチ処理)")

    for _ in range(dataset.seq_length - 1):
        for sym in symbols:
            all_probs[sym].append([0.0,0.0,0.0])
            all_risks[sym].append(0.0)


    with torch.no_grad():
        for batch in tqdm(dataloader,desc = "Pre-computing"):
            if batch is None:continue
            imgs_batch,tabs_batch = batch

            for i,sym in enumerate(symbols):
                sym_imgs = imgs_batch[:,i,:,:,:].to(device)
                sym_tabs = tabs_batch[:,i,:,:].to(device)

                class_out,risk_out = model(sym_imgs,sym_tabs)
                probs = torch.softmax(class_out,dim = 1).cpu().numpy()
                risks = risk_out.cpu().numpy()

                all_probs[sym].extend(probs.tolist())
                all_risks[sym].extend(risks.flatten().tolist())

    #4.saving CSV 
    df = pd.read_csv(merged_csv)
    for sym in symbols:
        df[f"{sym}_prob_0"] = [p[0] for p in all_probs[sym]]
        df[f"{sym}_prob_1"] = [p[1] for p in all_probs[sym]]
        df[f"{sym}_prob_2"] = [p[2] for p in all_probs[sym]]
        df[f"{sym}_risk_val"] = all_risks[sym]
    df.to_csv(output_csv, index=False)
    logging.info(f"✨ 事前推論完了！ {output_csv} に保存しました。")

if __name__ == "__main__":
    main()