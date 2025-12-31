#!/bin/bash
# =============================================================================
# Lux Lattice Library Installer
# =============================================================================
#
# Downloads and installs pre-built lux-lattice library from GitHub releases.
#
# Usage:
#   ./install.sh [version] [prefix]
#
# Examples:
#   ./install.sh                    # Latest version to /usr/local
#   ./install.sh 1.0.0              # Specific version to /usr/local
#   ./install.sh 1.0.0 ~/.local     # Specific version and prefix
#
# Environment:
#   LATTICE_VERSION  Override version (default: latest)
#   LATTICE_PREFIX   Override install prefix (default: /usr/local)
#   LATTICE_REPO     Override GitHub repo (default: luxfi/lattice-cpp)
#
# =============================================================================

set -euo pipefail

# Configuration
REPO="${LATTICE_REPO:-luxfi/lattice-cpp}"
VERSION="${1:-${LATTICE_VERSION:-latest}}"
PREFIX="${2:-${LATTICE_PREFIX:-/usr/local}}"

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Normalize architecture names
case "$ARCH" in
    x86_64|amd64)
        ARCH="amd64"
        ;;
    arm64|aarch64)
        ARCH="arm64"
        ;;
    *)
        echo "Error: Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Map Darwin to macos
case "$OS" in
    darwin)
        OS="macos"
        # Get actual architecture on macOS
        ARCH=$(uname -m)
        ;;
    linux)
        OS="linux"
        ;;
    *)
        echo "Error: Unsupported OS: $OS"
        exit 1
        ;;
esac

echo "=== Lux Lattice Library Installer ==="
echo "Repository: $REPO"
echo "Version:    $VERSION"
echo "Platform:   $OS-$ARCH"
echo "Prefix:     $PREFIX"
echo ""

# Get latest version if not specified
if [ "$VERSION" = "latest" ]; then
    echo "Fetching latest version..."
    VERSION=$(curl -sSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name":' | sed -E 's/.*"v([^"]+)".*/\1/')
    if [ -z "$VERSION" ]; then
        echo "Error: Failed to get latest version"
        echo "The repository may not have any releases yet."
        echo ""
        echo "To build from source instead:"
        echo "  git clone https://github.com/$REPO"
        echo "  cd lattice-cpp && mkdir build && cd build"
        echo "  cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX"
        echo "  make -j\$(nproc) && sudo make install"
        exit 1
    fi
    echo "Latest version: $VERSION"
fi

# Download URL
FILENAME="lattice-${VERSION}-${OS}-${ARCH}.tar.gz"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${FILENAME}"

echo ""
echo "Downloading: $URL"

# Create temp directory
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Download
if ! curl -sSL -o "$TMPDIR/$FILENAME" "$URL"; then
    echo "Error: Failed to download $URL"
    echo ""
    echo "The release may not exist for this platform."
    echo "Available releases: https://github.com/$REPO/releases"
    exit 1
fi

# Extract
echo "Extracting to $PREFIX..."
cd "$TMPDIR"
tar -xzf "$FILENAME"

# Install (may need sudo)
if [ -w "$PREFIX" ]; then
    cp -R include/* "$PREFIX/include/" 2>/dev/null || mkdir -p "$PREFIX/include" && cp -R include/* "$PREFIX/include/"
    cp -R lib/* "$PREFIX/lib/" 2>/dev/null || mkdir -p "$PREFIX/lib" && cp -R lib/* "$PREFIX/lib/"
else
    echo "Installing to $PREFIX (requires sudo)..."
    sudo mkdir -p "$PREFIX/include" "$PREFIX/lib" "$PREFIX/lib/pkgconfig"
    sudo cp -R include/* "$PREFIX/include/"
    sudo cp -R lib/* "$PREFIX/lib/"
fi

# Verify installation
echo ""
echo "Verifying installation..."
if command -v pkg-config &> /dev/null; then
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    if pkg-config --exists lux-lattice; then
        echo "pkg-config: $(pkg-config --modversion lux-lattice)"
        echo "CFLAGS:     $(pkg-config --cflags lux-lattice)"
        echo "LIBS:       $(pkg-config --libs lux-lattice)"
    else
        echo "Warning: pkg-config cannot find lux-lattice"
        echo "Add $PREFIX/lib/pkgconfig to PKG_CONFIG_PATH"
    fi
else
    echo "Note: pkg-config not installed, skipping verification"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Library installed to: $PREFIX"
echo ""
echo "To use with CGO, ensure:"
echo "  export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH"
echo "  export LD_LIBRARY_PATH=$PREFIX/lib:\$LD_LIBRARY_PATH  # Linux"
echo "  export DYLD_LIBRARY_PATH=$PREFIX/lib:\$DYLD_LIBRARY_PATH  # macOS"
