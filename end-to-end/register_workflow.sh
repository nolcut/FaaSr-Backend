#!/bin/bash
set -e

WORKFLOW_FILE=""

HELP_MESSAGE="
Usage: $0 --f <workflow-file> [-c]

Options:
  -f|--workflow-file <workflow-file>    Path to the workflow JSON file
  -c|--custom-container                 Allow custom containers
  -h|--help                             Show this help message

Example:
  $0 --workflow-file workflows/IntegrationTestWorkflow.json
  $0 --workflow-file workflows/IntegrationTestWorkflow.json -c
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
        -c|--custom-container)
            export CUSTOM_CONTAINER=true
            shift 1
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
cd faasr_workflow
python scripts/register_workflow.py --workflow-file "../$WORKFLOW_FILE"
git pull
