#!/bin/bash

echo "activating conda env..."
source /anaconda3/etc/profile.d/conda.sh

conda activate rl
echo "activated rl..."
echo "start to pull data and preprocessing..."
python main.py

echo "start model training..."

 echo "start model training for sector 10..."
 python3 fundamental_run_model.py   -sector_name sector10   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector10.csv    -first_trade_index 83
 echo "start model training for sector 15..."
 python3 fundamental_run_model.py   -sector_name sector15   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector15.csv    -first_trade_index 83
 echo "start model training for sector 20..."
 python3 fundamental_run_model.py   -sector_name sector20   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector20.csv    -first_trade_index 83
 echo "start model training for sector 25..."
 python3 fundamental_run_model.py   -sector_name sector25   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector25.csv    -first_trade_index 83
 echo "start model training for sector 30..."
 python3 fundamental_run_model.py   -sector_name sector30   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector30.csv    -first_trade_index 83
 echo "start model training for sector 35..."
 python3 fundamental_run_model.py   -sector_name sector35   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector35.csv    -first_trade_index 83
 echo "start model training for sector 40..."
 python3 fundamental_run_model.py   -sector_name sector40   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector40.csv    -first_trade_index 83
 echo "start model training for sector 45..."
 python3 fundamental_run_model.py   -sector_name sector45   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector45.csv    -first_trade_index 83
 echo "start model training for sector 50..."
 python3 fundamental_run_model.py   -sector_name sector50   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector50.csv    -first_trade_index 83
 echo "start model training for sector 55..."
 python3 fundamental_run_model.py   -sector_name sector55   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector55.csv    -first_trade_index 83
 echo "start model training for sector 60..."
 python3 fundamental_run_model.py   -sector_name sector60   -fundamental ress3k_fundamental_final.csv   -sector Data/1-focasting_data/sector60.csv    -first_trade_index 83

echo "select top20 stocks..."

python3 select_top20.py












