#!/bin/bash
# Pull latest course updates from the template repo

# Add upstream if not already set
if ! git remote | grep -q upstream; then
    echo "Adding upstream remote..."
    git remote add upstream https://github.com/UA-classroom/pia25-ml_1_course-ua_ml_1.git
fi

echo "Fetching updates from upstream..."
git fetch upstream

echo "Merging upstream/main..."
git merge upstream/main --no-edit

echo "Done! Your repo is up to date."
