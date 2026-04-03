import ast
from faasr_ai.utils.faasr_function_converter import convert_to_faasr_function
import textwrap

def main():
    task_2_code = textwrap.dedent("""
    import os
    import pandas as pd

    def task_2(output_folder="data/", input_1="dataset1.csv", input_2="dataset2.csv", output_1="summed_dataset.csv"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        input_path_1 = os.path.join(output_folder, input_1)
        input_path_2 = os.path.join(output_folder, input_2)
        output_path_1 = os.path.join(output_folder, output_1)
        
        dataset1 = pd.read_csv(input_path_1)
        dataset2 = pd.read_csv(input_path_2)
        
        summed_dataset = dataset1 + dataset2
        summed_dataset.to_csv(output_path_1, index=False)

    task_2()
    """)
    task_1_code = textwrap.dedent("""
    import os
    import pandas as pd
    import numpy as np

    def task_1(output_folder="data/", output_1="dataset1.csv", output_2="dataset2.csv"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        length = 100
        dataset1 = np.random.rand(length)
        dataset2 = np.random.rand(length)
        
        df1 = pd.DataFrame(dataset1, columns=["Value"])
        df2 = pd.DataFrame(dataset2, columns=["Value"])
        
        df1.to_csv(os.path.join(output_folder, output_1), index=False)
        df2.to_csv(os.path.join(output_folder, output_2), index=False)

    task_1()
    """)
    
    task_1 = {
                "task_id": "1",
                "dependent_task_ids": [],
                "instruction": "Generate two sample numerical data sets of equal length",
                "task_type": "other",
                "inputs": [],
                "outputs": [
                "dataset1.csv",
                "dataset2.csv"
                ]
            }
    
    task_2 = {
                "task_id": "2",
                "dependent_task_ids": [
                "1"
                ],
                "instruction": "Sum corresponding elements of the two data sets to create a combined data set",
                "task_type": "transform-only",
                "inputs": [
                "dataset1.csv",
                "dataset2.csv"
                ],
                "outputs": [
                "summed_dataset.csv"
                ]
            }
    
    code = convert_to_faasr_function(task_2_code, task_2)
    
    print(code)

if __name__ == "__main__":
    main()