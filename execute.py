import subprocess

# run the initialization script to generate the configuation and data
subprocess.run(["python3", "src/init.py"])

# run the backtester
subprocess.run(["cargo", "run", "--release"], cwd="./src/speedy", check=True)

# run the order processor
subprocess.run(["python3", "src/process.py"])
