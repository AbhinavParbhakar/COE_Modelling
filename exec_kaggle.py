import os
import time
import json

os.system("kaggle kernels push")

kaggle_meta = json.load(open('./kernel-metadata.json'))
kernel_id = kaggle_meta['id']


while True:
    status = os.popen(f"kaggle kernels status {kernel_id}").read()
    if "complete" in status:
        print("Execution complete!")
        break
    elif "error" in status:
        print("Kernel encountered an error.")
        break
    time.sleep(30)

os.system(f"kaggle kernels output {kernel_id} ./data/output")
