以下我會請你修改或創建python腳本，請你依照以下規則運行python file:
1. 務必要用wsl運行，在powershell輸入wsl進入
2. 使用uv run <python file> 運行腳本
3. 使用uv add <library> 或 uv add -r requirement.txt下載庫，但請確定requirement.txt內容是對的
4. 請不要把wsl跟uv的命令寫在同一次命令，因為wsl運行後才能使用uv

目前專案架構是，train_transformer.py進行訓練，然後data_normalizer.py是負責將data/資料夾裡的資料normalied到normalized_data資料夾裏頭。
然後用stock_data_fetcher.py下載資料