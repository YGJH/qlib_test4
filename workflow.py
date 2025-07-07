from colors import *
while True:
    try:
        import subprocess
        subprocess.run(["uv", "run", "stock_data_fetcher.py"])
        print_green("Stock data fetched successfully.")
        subprocess.run(["uv", "run", "check_data.py"])
        print_green("Robust normalization completed successfully.")
        subprocess.run(["uv", "run", "robust_normalizer.py"])
        print_green("Robust normalization check successfully.")
        subprocess.run(["uv", "run", "main.py"])
        print_green("Optimized training completed successfully.")
        break
    except Exception as e:
        print(f"Error occurred: {e}")
        continue