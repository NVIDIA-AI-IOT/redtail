# Patch for Auvidea J-120 board + Jetson TX2
There is a known issue with J-120 board and Jetson TX2 which manifests itself as USB devices not recognized when plugged into the board USB ports.
The issue will be fixed in the next version of JetPack, meanwhile users need to apply the fix described below.

1. When installing JetPack, make sure to **not** remove the files after installation. Check that installation directory on the host contains `64_TX2` directory.
2. Copy .dtb files in the current directory to the following directories:
    ```
    64_TX2/Linux_for_Tegra_tx2/bootloader
    64_TX2/Linux_for_Tegra_tx2/kernel/dtb
    64_TX2/Linux_for_Tegra_tx2/rootfs/boot
    ```
3. Reflash Jetson with modified JetPack, make sure to select the same host installation directory as before.

**Note**: this fix is tested only for JetPack 3.1, it may not work with other JetPack versions.

