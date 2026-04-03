import asyncio
import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_notebook
from datetime import datetime

async def execute_one_cell():
    """Execute a single code cell and print the output"""
    
    # Step 1: Create a notebook
    nb = new_notebook()
    
    # Step 2: Add code to execute
    code = """
import random
numbers = [random.randint(1, 100) for _ in range(5)]
print(f"Generated numbers: {numbers}")
print(f"Sum: {sum(numbers)}")
print(f"Average: {sum(numbers) / len(numbers):.2f}")
"""
    nb.cells.append(new_code_cell(source=code))
    
    # Step 3: Create notebook client
    client = NotebookClient(nb, timeout=60)

    # Step 4: Start kernel and execute
    client.create_kernel_manager()
    client.start_new_kernel()
    client.start_new_kernel_client()
    
    await client.async_execute_cell(nb.cells[0], cell_index=0)

    # Step 5: Print output
    for output in nb.cells[0].outputs:
        if output.output_type == "stream":
            print(output.text, end="")

    # ✅ Step 6: EXPORT NOTEBOOK
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/output_notebook_{timestamp}.ipynb"
    nbformat.write(nb, output_path)
    print(f"\n📓 Notebook saved to: {output_path}")

    # Step 7: Clean up
    await client.km.shutdown_kernel()


if __name__ == "__main__":
    asyncio.run(execute_one_cell())
