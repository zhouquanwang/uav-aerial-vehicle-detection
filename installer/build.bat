@echo off
REM ============================================================
REM  UAV 检测系统 - Windows 安装包一键构建
REM  运行此脚本将在 dist\UAV检测系统\ 下生成可执行程序
REM  预计耗时: 5-15 分钟（取决于机器性能）
REM ============================================================
cd /d "%~dp0\.."
echo.
echo   UAV 检测系统 — 安装包构建
echo   %date% %time%
echo.
echo   清理旧构建产物...
if exist "build" rmdir /s /q "build"
if exist "dist"  rmdir /s /q "dist"
echo.

echo   [1/2] 使用 PyInstaller 打包...
call .venv\Scripts\python.exe -m PyInstaller ^
    --distpath=dist ^
    --workpath=build ^
    --noconfirm --clean ^
    installer\uav_detection.spec

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   ✗ 打包失败！请检查错误信息
    pause
    exit /b 1
)

echo.
echo   [2/2] 整理输出目录...
if not exist "dist\UAV检测系统\outputs" mkdir "dist\UAV检测系统\outputs"
echo @echo off> "dist\启动UAV检测系统.bat"
echo echo 启动无人机航拍视频检测系统...>> "dist\启动UAV检测系统.bat"
echo start "" "UAV检测系统\UAV检测系统.exe">> "dist\启动UAV检测系统.bat"

echo.
echo   ============================================================
echo     ✓ 打包完成!
echo     输出目录: dist\UAV检测系统\
echo     启动方式: 双击 dist\启动UAV检测系统.bat
echo   ============================================================
echo.
pause
