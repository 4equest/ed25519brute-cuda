@echo off
setlocal enabledelayedexpansion

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
    echo Usage: build.bat [main^|test] [--prefix ^<pattern^>] [--suffix ^<pattern^>]
    exit /b 1
)

rem Parse additional arguments for fixed search mode
set FIXED_FLAGS=
set PREFIX_ARGS=
set SUFFIX_ARGS=
set HAS_PREFIX=0
set HAS_SUFFIX=0

rem Skip first argument (target)
set ARGS_STARTED=0
for %%a in (%*) do (
    if !ARGS_STARTED!==0 (
        set ARGS_STARTED=1
    ) else (
        if "%%a"=="--prefix" (
            set NEXT_IS_PREFIX=1
        ) else if "%%a"=="--suffix" (
            set NEXT_IS_SUFFIX=1
        ) else if defined NEXT_IS_PREFIX (
            set PREFIX_ARGS=!PREFIX_ARGS! --prefix %%a
            set HAS_PREFIX=1
            set NEXT_IS_PREFIX=
        ) else if defined NEXT_IS_SUFFIX (
            set SUFFIX_ARGS=!SUFFIX_ARGS! --suffix %%a
            set HAS_SUFFIX=1
            set NEXT_IS_SUFFIX=
        )
    )
)

rem Generate fixed conditions if prefix or suffix specified
if !HAS_PREFIX!==1 set FIXED_FLAGS=-DFIXED_SEARCH -DFIXED_PREFIX
if !HAS_SUFFIX!==1 (
    if defined FIXED_FLAGS (
        set FIXED_FLAGS=!FIXED_FLAGS! -DFIXED_SUFFIX
    ) else (
        set FIXED_FLAGS=-DFIXED_SEARCH -DFIXED_SUFFIX
    )
)

if defined FIXED_FLAGS (
    echo Generating fixed conditions...
    python src\gen_conditions.py !PREFIX_ARGS! !SUFFIX_ARGS! -o src\fixed_conditions.cuh
    if errorlevel 1 (
        echo Error: Failed to generate fixed conditions
        exit /b 1
    )
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
if defined FIXED_FLAGS (
    echo Fixed search flags: !FIXED_FLAGS!
)
nvcc -O3 --use_fast_math --extra-device-vectorization -arch=sm_89 -rdc=true -lcudadevrt -t 0 --ptxas-options=-v !FIXED_FLAGS! -o %OUT% %SRC% -I src > build_log.txt 2>&1

if %errorlevel% equ 0 (
    echo Build successful: %OUT%
) else (
    echo Build failed
    type build_log.txt
    exit /b 1
)
