@echo off
pushd C:\quant\snailx
git pull
git add .
git commit -m "auto update data"
git push

