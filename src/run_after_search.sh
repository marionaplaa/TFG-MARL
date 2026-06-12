#!/bin/bash

# Wait until epymarl grid-search is finished
while pgrep -f "search.py run" > /dev/null; do
    echo "Grid search is still running... checking again in 600 seconds."
    sleep 600
done

echo "Grid search finished. Starting the next run..."
echo y | python3 search.py run --config=sssearch2.yaml --seeds 5 locally
