#!/bin/bash


SOURCE_DIR="/Users/makhosi/Downloads/SPEECH_DATA_RAW/"
DEST_DIR="/Users/makhosi/Downloads/SPEECH_AUDIO_DATA/"
LOG_FILE="unzip_errors.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Initialize counters
processed=0
skipped=0
errors=0

echo -e "\n${YELLOW}=== Starting processing from $SOURCE_DIR to $DEST_DIR ===${NC}\n"

# Process each file in source directory
for file in "$SOURCE_DIR"/*; do
    # Skip if not a regular file
    [ ! -f "$file" ] && continue
    
    # Check if file is a zip file (case insensitive)
    if [[ "$file" =~ \.(zip|ZIP)$ ]]; then
        filename=$(basename "$file")
        base_filename="${filename%.*}"
        echo -e "\n${YELLOW}Processing: $filename...${NC}"
        
        # Create destination subdirectory based on zip filename
        zip_dest_dir="$DEST_DIR/$base_filename"
        mkdir -p "$zip_dest_dir"
        
        # Try to unzip while preserving folder structure
        if unzip -q "$file" -d "$zip_dest_dir" 2>/dev/null; then
            # Fix permissions (sometimes zip files have weird permissions)
            chmod -R u+rw "$zip_dest_dir"
            
            # Report success
            ((processed++))
            echo -e "${GREEN}✓ Successfully processed: $filename${NC}"
            echo "  Extracted to: $zip_dest_dir"
        else
            ((errors++))
            error_msg="Error unzipping $filename"
            echo -e "\n${RED}✗ ERROR: $error_msg${NC}\n" >&2
            echo "[$(date)] ERROR: $error_msg" >> "$LOG_FILE"
            unzip -l "$file" 2>&1 | sed 's/^/    /' >> "$LOG_FILE"
            echo "" >> "$LOG_FILE"
            
            # Remove failed extraction directory if empty
            rmdir --ignore-fail-on-non-empty "$zip_dest_dir"
        fi
    else
        ((skipped++))
        echo -e "${YELLOW}- Skipping non-zip file: $(basename "$file")${NC}"
    fi
done

echo -e "\n${YELLOW}=== Processing complete ===${NC}\n"
echo -e "${GREEN}Successfully processed: $processed files${NC}"
echo -e "${YELLOW}Skipped non-zip files: $skipped files${NC}"
echo -e "${RED}Errors encountered: $errors files${NC}"
[ $errors -gt 0 ] && echo -e "\n${RED}Error details logged to: $LOG_FILE${NC}"
echo ""