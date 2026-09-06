@echo off

for /d %%D in (faster_capture\output\*) do (
    python .\faster_capture\archive\bmp2npy.py "%%D" "%%D.npy"
)
