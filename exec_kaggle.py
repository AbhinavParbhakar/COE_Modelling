import os
import time
import json

metadata = json.load(open('./kernel-metadata.json'))
id = metadata['id']

os.system('kaggle kernels push')

while True:
    result = os.popen(f'kaggle kernels status {id}').read().encode("utf-8", "ignore").decode("utf-8")
    if "complete" in result:
        print("complete")
        break
    elif "error" in result:
        print("Error")
        break
    else:
        time.sleep(5)
    
os.system(f'kaggle kernels output {id} -p ./data/output')
    

