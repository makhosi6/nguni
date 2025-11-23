#!/bin/bash

# Set script to exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Starting dataset organization process...${NC}"

# Create Python virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv "$PROJECT_ROOT/venv"
fi

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install -q pathlib typing_extensions tqdm pyyaml

# Run the organization script
echo -e "${YELLOW}Organizing datasets...${NC}"
python "$SCRIPT_DIR/organize_datasets.py"

# Run validation
echo -e "${YELLOW}Validating dataset organization...${NC}"
python "$SCRIPT_DIR/validate_datasets.py"

# Check if both scripts completed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Dataset organization and validation completed successfully!${NC}"
    echo -e "${GREEN}Output data is available in the OUTPUT directory${NC}"
else
    echo -e "${RED}An error occurred during processing${NC}"
    exit 1
fi

# Deactivate virtual environment
deactivate
