#!/bin/bash
set -e

WORKFLOW_FILE=""

HELP_MESSAGE="
Usage: $0 --workflow-file <workflow-file>

Options:
  -f|--workflow-file <workflow-file>  Path to the workflow JSON file
  -h|--help            Show this help message and exit

Example:
  $0 --workflow-file workflows/IntegrationTestWorkflow.json
"

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--workflow-file)
            # Exit if workflow file is not a file
            if [ ! -f "$2" ]; then
                echo "Error: Workflow file $2 is not a file"
                exit 1
            fi
            WORKFLOW_FILE=$2
            shift 2
            ;;
        -h|--help)
            echo "$HELP_MESSAGE"
            exit 0
            ;;
        -*|--*)
            echo "Invalid argument: $1"
            exit 1
    esac
done

if [ -z "$WORKFLOW_FILE" ]; then
    echo "Error: Workflow file is required"
    echo "$HELP_MESSAGE"
    exit 1
fi

export $(cat .env | xargs)
python -m framework.workflow_runner --workflow-file $WORKFLOW_FILE
