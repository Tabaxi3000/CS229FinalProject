#!/bin/bash

# This script helps you push your code to GitHub
# Replace GITHUB_USERNAME with your GitHub username
# Replace REPO_NAME with your repository name

# Usage: ./push_to_github.sh GITHUB_USERNAME REPO_NAME

if [ $# -ne 2 ]; then
    echo "Usage: $0 GITHUB_USERNAME REPO_NAME"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME=$2

# Set the correct remote URL
git remote set-url origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# Verify the remote URL
echo "Remote URL set to: $(git remote get-url origin)"

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin cs229-milestone

echo "Done! Your code has been pushed to GitHub."
echo "Visit https://github.com/$GITHUB_USERNAME/$REPO_NAME to see your repository." 