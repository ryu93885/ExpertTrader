import pandas as pd
import glob

# 学習用データと推論用データのパス（実際のファイル名に合わせて変更してください）
train_file = glob.glob("labeled_data/*_train_*_labeled.csv")[0]
test_file = glob.glob("labeled_data/*_test_*_labeled.csv")[0] # もしくは processed_data の test ファイル

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

train_cols = set(df_train.columns)
test_cols = set(df_test.columns)

print("=== データ列の比較診断 ===")
print(f"学習データの列数: {len(train_cols)}")
print(f"テストデータの列数: {len(test_cols)}\n")

missing_in_test = train_cols - test_cols
print("❌ テストデータに足りない（抜け落ちている）列:")
for col in missing_in_test:
    print(f"  - {col}")