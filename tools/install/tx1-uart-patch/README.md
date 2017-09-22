# Patch for Jetson TX1 UART issue
There a known issue in JetPack 3.1 and Jetson TX1 which manifests itself as TX1 becoming unresponsive when connecting to `/dev/ttyTHS1` device. Such device is assigned to `UART 2` port on Auvidea J-120 board which makes it impossible to use any device connected to that UART port (e.g. Pixhawk).
The issue will be fixed in the next version of JetPack, meanwhile users need to apply the fix described below.

1. When installing JetPack, make sure to **not** remove the files after installation. Check that installation directory on the host contains `64_TX1` directory.
2. Copy .dtb file in the current directory to the following directories:
```
64_TX1/Linux_for_Tegra_64_tx1/bootloader
64_TX1/Linux_for_Tegra_64_tx1/kernel/dtb
64_TX1/Linux_for_Tegra_64_tx1/rootfs/boot​
```
3. Reflash Jetson with modified JetPack, make sure to select the same host installation directory as before.

**Note**: this fix is tested only for JetPack 3.1, it may not work with other JetPack versions.
