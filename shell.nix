let
  pkgs = import (
    builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/23.11.tar.gz"
  ) {};
in
  pkgs.mkShell rec {
    packages = with pkgs; [
      python311
      nix-ld
      google-cloud-sdk-gce
      jq
    ];
    buildInputs = with pkgs; [
      libxkbcommon
      glui
      libGL
      gcc-unwrapped.lib
      pkg-config
      libz
      vulkan-loader
      xorg.libX11
      xorg.libXcursor
      xorg.libXrender
      xorg.libXi
      xorg.libXrandr
      xorg.libxcb
    ];
    LD_LIBRARY_PATH= pkgs.lib.makeLibraryPath buildInputs;

    NIX_LD_LIBRARY_PATH = LD_LIBRARY_PATH;
    NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
  }