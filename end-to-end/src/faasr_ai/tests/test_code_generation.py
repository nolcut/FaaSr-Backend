from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI
import asyncio
from typing import Optional

from faasr_ai.agents.coding import coding_node, make_initial_coding_state

import anthropic

load_dotenv()

if "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]:
    os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"].strip()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_TEMPERATURE = float(os.getenv("ANTHROPIC_TEMPERATURE", "0"))

client = anthropic.Anthropic()

def llm_call(prompt: str, system: Optional[str] = None) -> str:
    """LLM adapter used by all nodes."""
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        temperature=ANTHROPIC_TEMPERATURE,
        max_tokens=4096,
        system= system or "Follow instructions precisely. Output valid JSON when asked.",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.content[0].text or "").strip()

async def main():
    tasks = [
  {
    "task_id": "1",
    "dependent_task_ids": [],
    "instruction": "Generate two CSV files, each containing a single column named 'value' with a header row, and 15 randomly generated integers between 1 and 100. Save the first file as 'integers_a.csv' and the second file as 'integers_b.csv'. Both files must have exactly 15 rows of data (excluding the header) to ensure they can be paired for element-wise addition.",
    "inputs": [],
    "outputs": [
      "integers_a.csv",
      "integers_b.csv"
    ]
  },
  {
    "task_id": "2",
    "dependent_task_ids": [
      "1"
    ],
    "instruction": "Read both 'integers_a.csv' and 'integers_b.csv', each of which contains a header row with column name 'value' followed by 15 integer rows. First, verify that both files have the same number of data rows; if they do not match, raise an error. Then perform element-wise addition by pairing each integer from 'integers_a.csv' with the corresponding integer at the same row position in 'integers_b.csv'. Write the resulting sums to 'summed_results.csv' with a single column named 'value' and a header row, preserving row order from top to bottom, so that row 1 of the output contains the sum of row 1 from each input file, row 2 contains the sum of row 2, and so on.",
    "inputs": [
      "integers_a.csv",
      "integers_b.csv"
    ],
    "outputs": [
      "summed_results.csv"
    ]
  }
]
    
    user_request = """Generate two csv files with a single column of randomly generated integers. Perform an element-wise addition of the corresponding elements from both 
        files. The resulting sums should be written into a single output csv file containing one column with the summed values."""
    
    coding_agent = make_initial_coding_state(tasks, user_request, "data", "test_functions")
    result = await coding_node(coding_agent, llm_call, max_attempts=5)
    
    print("Notebook path:", result["notebook_path"])
    print("Generated functions:", list(result["generated_functions"].keys()))
    print("Errors:", result["coding_errors"])


if __name__ == "__main__":
    asyncio.run(main())

    
    
    
