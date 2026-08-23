#!/usr/bin/env bash
set -e

# Colors for terminal formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   🤖 Installing Agent CLI & Environment          ${NC}"
echo -e "${BLUE}==================================================${NC}\n"

# 1. OS & Distribution Detection
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM="Linux";;
    Darwin*)    PLATFORM="macOS";;
    *)          echo -e "${RED}Unsupported platform: ${OS}${NC}"; exit 1;;
esac

echo -e "Detected OS: ${GREEN}${PLATFORM}${NC}"

# 2. Package Manager & Dependency Verification (Linux)
if [ "${PLATFORM}" = "Linux" ]; then
    if command -v apt-get &> /dev/null; then
        PM="apt"
    elif command -v dnf &> /dev/null; then
        PM="dnf"
    elif command -v pacman &> /dev/null; then
        PM="pacman"
    elif command -v zypper &> /dev/null; then
        PM="zypper"
    elif command -v apk &> /dev/null; then
        PM="apk"
    fi
fi

# Ensure Python 3 and Git are installed
if ! command -v python3 &> /dev/null || ! command -v git &> /dev/null; then
    echo -e "${YELLOW}Missing base dependencies (Python3 / Git). Installing...${NC}"
    if [ "${PLATFORM}" = "macOS" ]; then
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}Installing Homebrew...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python git
    elif [ "${PLATFORM}" = "Linux" ]; then
        case "${PM}" in
            apt)    sudo apt-get update && sudo apt-get install -y python3 python3-pip git curl;;
            dnf)    sudo dnf install -y python3 python3-pip git curl;;
            pacman) sudo pacman -Sy --noconfirm python python-pip git curl;;
            zypper) sudo zypper install -y python3 python3-pip git curl;;
            apk)    sudo apk add python3 py3-pip git curl;;
            *)      echo -e "${RED}Unable to auto-install dependencies. Please install python3 and git manually.${NC}"; exit 1;;
        esac
    fi
fi

# 3. Ensure 'uv' package manager is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}'uv' not found. Installing Astral uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

# 4. Install agent-cli using uv tool
echo -e "${GREEN}Installing agent-cli from GitHub repository...${NC}"
uv tool install git+https://github.com/CodeCentury22/agent-cli.git --force

# 5. Environment PATH Setup
SHELL_PROFILE=""
case "$SHELL" in
    */zsh)  SHELL_PROFILE="$HOME/.zshrc";;
    */bash) SHELL_PROFILE="$HOME/.bashrc";;
    *)      SHELL_PROFILE="$HOME/.profile";;
esac

UV_BIN_PATH="$HOME/.local/bin"
if [[ ":$PATH:" != *":$UV_BIN_PATH:"* ]]; then
    echo "export PATH=\"$UV_BIN_PATH:\$PATH\"" >> "$SHELL_PROFILE"
    echo -e "${YELLOW}Added $UV_BIN_PATH to $SHELL_PROFILE${NC}"
fi

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}🎉 agent-cli installed successfully!             ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "Run ${BLUE}source $SHELL_PROFILE${NC} or open a new terminal."
echo -e "Then launch the interactive agent anytime by typing:\n"
echo -e "    ${GREEN}agent-cli${NC}\n"