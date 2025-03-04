import os
import time
import json

def run_model(model_name="GNN"):
    """
    Run the model based on the model name
    
    model_name = "GNN" or "NN"
    """
    metadata = json.load(open(f'./{model_name}/kernel-metadata.json'))
    print(metadata)
    id = metadata['id']

    os.system(f'kaggle kernels push -p ./{model_name}')

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
        
    os.system(f'kaggle kernels output {id} -p ./data/output/{model_name}')
    
if __name__ == "__main__":
    run_model(model_name="CNN")
    

