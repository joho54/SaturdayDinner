#!/bin/bash

# conda 초기화
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh

# signlang 환경 활성화
conda activate signlang

# 현재 디렉토리에서 실행
for i in {9..13}; do
    python3 main.py specs/spec$i.json
done