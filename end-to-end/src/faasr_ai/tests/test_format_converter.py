import json
from faasr_ai.utils.faasr_workflow_converter import tasks_to_faasr_workflow

def main():
    tasks = [
  {
    "task_id": 1,
    "dependent_task_ids": [],
    "instruction": "generate the first sample data set",
    "task_type": "other",
    "inputs": [],
    "outputs": [
      "sample_data_set_1.csv"
    ]
  },
  {
    "task_id": 2,
    "dependent_task_ids": [],
    "instruction": "generate the second sample data set",
    "task_type": "other",
    "inputs": [],
    "outputs": [
      "sample_data_set_2.csv"
    ]
  },
  {
    "task_id": 3,
    "dependent_task_ids": [
      1,
      2
    ],
    "instruction": "sum the two sample data sets",
    "task_type": "other",
    "inputs": [
      "sample_data_set_1.csv",
      "sample_data_set_2.csv"
    ],
    "outputs": [
      "sum_of_sample_data_sets.csv"
    ]
  }
]

    wf = tasks_to_faasr_workflow(
        tasks,
        folder="tutorial",
        github_username="YOUR_USERNAME",
        action_repo_name="FaaSr-workflow",
        branch_name="main",
        invocation_id="tutorial",
        workflow_name="tutorial",
    )

    print(json.dumps(wf, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
