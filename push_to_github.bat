@echo off
cd /d "%~dp0"
echo ========================================================
echo Pushing Iris Classification Benchmark to GitHub
echo ========================================================
echo.

echo [1/5] Initializing Git repository...
git init

echo.
echo [2/5] Staging files...
git add .

echo.
echo [3/5] Committing project files...
git commit -m "feat: add Iris supervised classification benchmark (Logistic Regression vs Decision Tree)"

echo.
echo [4/5] Setting main branch and linking GitHub remote...
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/pandureddypandureddy798-collab/iris-classification-benchmark910.git

echo.
echo [5/5] Pushing to GitHub...
git push -u origin main

echo.
echo ========================================================
echo Finished! Check the output above.
echo ========================================================
pause
