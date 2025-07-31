
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
if ! command -v conda &> /dev/null; then
    echo "[INFO] conda 명령어를 찾을 수 없어 Miniconda를 설치합니다..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    rm -f miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# Conda 환셩 생성 및 활성화
## TODO
eval "$(conda shell.bash hook)"
ENV_NAME="myenv"
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "[INFO] Conda 환경 '${ENV_NAME}'가 없어 생성합니다..."
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
conda install -y mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    name=$(basename "$file" .py)
    input_file="../input/${name}_input"
    output_file="../output/${name}_output"
    if [[ -f "$input_file" ]]; then
        echo "[RUN] python \"$file\" < \"$input_file\" > \"$output_file\""
        python "$file" < "$input_file" > "$output_file"
    else
        echo "[WARN] 입력 파일이 없습니다: $input_file (스킵)"
    fi
done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
mypy *.py > ../mypy_log.txt 2>&1
echo "[INFO] mypy 로그 저장: ../mypy_log.txt"

# conda.yml 파일 생성
## TODO
conda env export > ../conda.yml
echo "[INFO] conda.yml 저장 완료: ../conda.yml"

# 가상환경 비활성화
## TODO
conda deactivate