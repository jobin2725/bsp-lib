# CentOS 7 공용 서버에서 관리자 없이 QEMU / BSP 빌드 환경 구축 가이드


## 1. Python 환경 정리 (필수)

### 1.1 pyenv 설치

시스템 Python 2.7을 피하기 위해 pyenv를 사용한다.

```bash
curl https://pyenv.run | bash
```

`~/.bashrc`에 추가:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

적용:

```bash
source ~/.bashrc
```

---

### 1.2 Python 3.12 설치 (주의 사항 포함)

CentOS 7 기본 OpenSSL이 너무 오래되어 pyenv 빌드가 실패할 수 있다.  
이를 위해 **유저 홈에 OpenSSL을 먼저 설치**한다.

```bash
# OpenSSL 1.1.1
cd ~
wget https://www.openssl.org/source/openssl-1.1.1w.tar.gz
tar xf openssl-1.1.1w.tar.gz
cd openssl-1.1.1w
./config --prefix=$HOME/openssl
make -j
make install
```

환경 변수 설정:

```bash
export CPPFLAGS="-I$HOME/openssl/include"
export LDFLAGS="-L$HOME/openssl/lib"
export PKG_CONFIG_PATH="$HOME/openssl/lib/pkgconfig"
```

Python 설치:

```bash
pyenv install 3.12.12
pyenv local 3.12.12
```

확인:

```bash
python --version
```

---

### 1.3 venv 생성

```bash
python -m venv venv
source venv/bin/activate
```

기본 패키지 업데이트:

```bash
pip install --upgrade pip setuptools wheel
```

> ⚠️ Python 3.12에서는 `distutils`가 제거되었으므로 **setuptools 설치는 필수**다.

---

## 2. 빌드 툴체인 (로컬 설치)

### 2.1 Meson / Ninja

Meson과 Ninja는 Python 패키지로 venv에 설치한다.

```bash
pip install meson ninja
```

확인:

```bash
meson --version
ninja --version
```

---

### 2.2 CMake (시스템 버전 사용 불가)

CentOS 7 기본 CMake는 2.8.x로 너무 낮다.

공식 바이너리 사용:

```bash
cd ~
wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.tar.gz
tar xf cmake-3.27.9-linux-x86_64.tar.gz
```

PATH 추가:

```bash
export PATH=$HOME/cmake-3.27.9-linux-x86_64/bin:$PATH
```

확인:

```bash
cmake --version
```

---

## 3. glib 2.72 빌드 (핵심 파트)

### 3.1 왜 로컬 glib이 필요한가

- QEMU 최신은 **glib >= 2.66**을 요구
- CentOS 7 기본 glib (2.56)으로는 불가
- glib 2.66+는 **Meson 전용 빌드 시스템**

---

### 3.2 Clock skew 문제

Meson은 아래 조건에서 **하드 실패**한다.

```text
file mtime > system clock
```

본 서버에서는:
- system clock이 실제보다 미래
- Meson은 이를 무시할 옵션이 없음

해결책:
- **빌드를 로컬 파일시스템(/local 또는 /tmp)에서 수행**

> ⚠️ 아래 *부록 A*에 **비공식 우회(hack)** 방법을 기록한다.

---

### 3.3 glib 소스 이동 (중요)

먼저 glib 소스를 **홈 디렉토리** 에서 다운로드한다.

```bash
cd ~
wget https://download.gnome.org/sources/glib/2.72/glib-2.72.4.tar.xz
tar xf glib-2.72.4.tar.xz
```

---

### 3.4 glib 빌드

```bash
cd ~/glib-2.72.4
mkdir build && cd build

meson setup .. \
  --prefix=$HOME/local \
  -Dlibmount=disabled \
  -Dselinux=disabled \
  -Dman=false \
  -Dtests=false

ninja
ninja install
```

확인:

```bash
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$HOME/local/lib64/pkgconfig
pkg-config --modversion glib-2.0
```

---

## 4. QEMU 빌드


```bash
cd qemu-custom

export LD_LIBRARY_PATH=$HOME/local/lib:$HOME/local/lib64

meson setup build \
  --prefix=$HOME/local \
  -Dglib_subproject=disabled

ninja -C build
ninja -C build install
```

glib 링크 확인:

```bash
ldd build/qemu-system-x86_64 | grep glib
```

---

## 5. RISC-V BSP / Toolchain 빌드

여기서부터는 bsp-lib의 README.md를 따라하면 됩니다.

### 5.1 CMake 에러 해결

에러:

```text
CMake 3.20 or higher is required
```

→ 로컬 CMake 사용 (2.2 참고)

---

### 5.2 riscv64-unknown-elf-gcc 에러

에러:

```text
is not a full path to an existing compiler tool
```

원인:
- 툴체인이 아직 빌드되지 않았거나
- PATH에 없음

해결:

```bash
export RISCV=$HOME/project/bsp-lib/lib/riscv-newlib
export PATH=$RISCV/bin:$PATH
export CC=$RISCV/bin/riscv64-unknown-elf-gcc
export ASM=$RISCV/bin/riscv64-unknown-elf-gcc
```

---

## 9. 참고

- glib ≥ 2.66: Meson only
- Python 3.12: distutils 제거됨
- Meson: clock skew 무시 옵션 없음

---

## 부록 A. Meson Clock Skew 비공식 우회 방법 (⚠️ 비권장 / 실험용)

> **주의**: 이 방법은 Meson의 안전장치를 직접 수정하는 **비공식(hack)** 방식이다.
> 재현성/정확성을 보장하지 않으며, **책임 하에 사용**해야 한다.

### 배경

Meson은 빌드 중 다음 로직으로 clock skew를 검사한다.

```python
# venv/lib/python3.12/site-packages/mesonbuild/backend/backends.py

def check_clock_skew(self, file_list):
    import time
    now = time.time()
    for f in file_list:
        absf = os.path.join(self.environment.get_build_dir(), f)
        ftime = os.path.getmtime(absf)
        delta = ftime - now
        if delta > 1990.001:
            raise MesonException(
                f'Clock skew detected. File {absf} has a time stamp {delta:.4f}s in the future.'
            )
```

시스템 시간이 미래로 설정된 환경에서는 `delta`가 항상 양수가 되어 빌드가 중단된다.

### 적용한 우회 방법

- 위 코드의 **임계값(`1990.001`)을 더 큰 값으로 조정**하여
  현실적인 skew 범위를 무시하도록 변경

예시:
```python
# BEFORE
if delta > 1990.001:

# AFTER (예: 하루 이상 차이만 에러)
if delta > 86400 * 7:
```

이 변경으로 **수 분~수 시간 수준의 skew**는 통과하게 된다.

### 영향 및 리스크

- 증분 빌드 무결성 보장 ❌
- 재현성 ❌
- upstream Meson 동작과 불일치 ❌

### 언제 사용해도 되는가

- 공용 서버
- 관리자 권한 없음
- system clock 수정 불가
- **빌드를 반드시 진행해야 하는 실험 환경**

> 정식 해결책은 **system clock 수정** 또는 **정상 시간의 빌드 서버 사용**이다.

---

**이 문서는 실험 결과 기반으로 작성되었으며, 동일 서버에서 재현 가능함을 전제로 한다.**

