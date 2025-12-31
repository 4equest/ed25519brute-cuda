@echo off
setlocal

rem Target selection
set TARGET=%1
if "%TARGET%"=="" set TARGET=main

if not exist build mkdir build

if "%TARGET%"=="main" (
    set SRC=src\main.cu
    set OUT=build\ed25519brute_cuda.exe
) else if "%TARGET%"=="test" (
    set SRC=src\test_kernels.cu
    set OUT=build\test_kernels.exe
) else (
    echo Unknown target: %TARGET%
    echo Usage: build.bat [main^|test]
    exit /b 1
)


rem Check if cl.exe (MSVC compiler) is already in path
where cl.exe >nul 2>nul
if %errorlevel% equ 0 goto :build

rem Find vcvars64.bat
echo Searching for vcvars64.bat...
set VCVARS=""
for /f "usebackq tokens=*" %%i in (`dir /s /b "C:\Program Files\Microsoft Visual Studio\*vcvars64.bat" 2^>nul`) do (
    set VCVARS="%%i"
    goto :found_vcvars
)

if %VCVARS%=="" (
    for /f "usebackq tokens=*" %%i in (`dir /s /b "C:\Program Files (x86)\Microsoft Visual Studio\*vcvars64.bat" 2^>nul`) do (
        set VCVARS="%%i"
        goto :found_vcvars
    )
)

if %VCVARS%=="" (
    echo Error: vcvars64.bat not found. Please install Visual Studio C++ build tools.
    exit /b 1
)

:found_vcvars
echo Found vcvars64.bat at %VCVARS%
call %VCVARS% >nul

:build
echo Building %TARGET% (%SRC% -^> %OUT%)...
nvcc -O3 --use_fast_math --extra-device-vectorization -arch=sm_89 -rdc=true -lcudadevrt -t 0 --ptxas-options=-v -o %OUT% %SRC% -I src > build_log.txt 2>&1

if %errorlevel% equ 0 (
    echo Build successful: %OUT%
) else (
    echo Build failed
    exit /b 1
)
