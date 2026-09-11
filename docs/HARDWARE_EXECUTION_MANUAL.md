# CoreEP4CE6 阵列控制执行手册

本手册面向 `16 x 16 = 256` 通道 SonicSurface 阵列与 Waveshare `CoreEP4CE6` 控制板。当前控制板为新空板，因此执行顺序是：先烧录 FPGA 固件，再确认 USB/UART 链路，验证通道映射与相位方向，最后下发 16 x 16 优化相位。

## 0. 安全边界

- 先连接 USB，再接通阵列发射电源；停止时先发送全关闭，再断开发射电源。
- 测试时保持阵列上方无遮挡，手、探头和易受声压影响物体不得靠近发射面。
- 保留物理电源开关或可快速断开的电源线。
- 不要让供应商程序和本项目服务同时打开同一个 `COM` 端口。
- `--live` 表示服务会实际写串口；非关闭图样还需要 profile 的 `mapping_verified: true`。

## 1. 烧录 FPGA 固件

Python 服务只发送 UART 相位帧，不能替代 FPGA 固件。空白 `CoreEP4CE6` 必须先通过 JTAG 下载 SonicSurface 的相位驱动工程。

### 1.1 已确认的第三方工程

仓库中的匹配工程为：

```text
third_party/SonicSurface/Firmware/FPGA primary/QuadrupleBuffer.qpf
```

该工程配置：

```text
器件：EP4CE6E22C8
系列：Cyclone IV E
顶层：QuadrupleBuffer
工具：Quartus II 13.0 SP1
```

选择 `FPGA primary`，不要选择 `FPGA secondary` 或 `FPGA tactile modulation`。供应商说明中，primary 固件使用命令 `192` 启用，并产生自己的时钟；这与本项目的单板 `fpga_256` UART 帧协议对应。

### 1.2 所需硬件与软件

- Intel/Altera USB-Blaster，或兼容 JTAG 下载器。
- CoreEP4CE6 的 JTAG 接口连接线。
- 与控制板匹配的稳定电源。
- Quartus II 13.0 SP1。该版本是第三方工程声明的原始版本；较新版本可尝试打开工程，但首次烧录优先使用 13.0 SP1。

USB-UART 线不能代替 USB-Blaster。USB-UART 用于运行时下发相位；USB-Blaster 用于把 FPGA 逻辑写入器件或配置 Flash。

### 1.3 首次临时下载到 FPGA

1. 断开阵列发射电源，连接 USB-Blaster 的 JTAG 信号与控制板，给控制板上电。
2. 打开 Quartus II，选择 `File -> Open Project`，打开：

```text
third_party/SonicSurface/Firmware/FPGA primary/QuadrupleBuffer.qpf
```

3. 确认器件显示为 `EP4CE6E22C8`，不修改 `.qsf` 中的引脚分配。
4. 执行 `Processing -> Start Compilation`，无错误后生成：

```text
third_party/SonicSurface/Firmware/FPGA primary/output_files/QuadrupleBuffer.sof
```

5. 打开 `Tools -> Programmer`，选择硬件 `USB-Blaster`，模式为 `JTAG`。
6. 添加 `QuadrupleBuffer.sof`，勾选 `Program/Configure`，点击 `Start`。
7. 成功后不要断电，直接进入第 2 节的 USB/UART 连接确认。

`.sof` 仅写入 FPGA 的易失配置 RAM。断电后会丢失，适用于首次验证。

### 1.4 写入配置 Flash

只有 JTAG 临时下载和后续声场测试均成功后，再烧录配置 Flash，避免在未验证的引脚/阵列装配上固化镜像。

1. 在 Quartus 里用 `File -> Convert Programming Files` 从已验证的 `.sof` 生成 `.jic`；配置器件型号必须以 CoreEP4CE6 板上实际 Flash 丝印和原理图为准，不能猜测。
2. 在 `Tools -> Programmer` 中选择 `Active Serial Programming`，添加生成的 `.jic`。
3. 勾选 `Program/Configure` 和 `Verify` 后开始写入。
4. 断电重启，在不连接 USB-Blaster 的情况下重新执行第 2 节。只有重启后仍能响应 UART 图样，才说明 Flash 自启动成功。

第三方仓库说明其预编译 `.jic` 可以通过 Blaster 上传，但当前本地副本未包含可直接使用的 `.jic`，故应由上述工程重新编译生成。该固件未包含阵元相位校准；校准仍在本项目 hardware profile 的 `phase_offsets_rad` 中完成。

## 2. 软件准备

在仓库根目录安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

硬件运行至少需要 `pyserial`。检查三个工具可用：

```powershell
python src/hardware_ports.py --help
python src/hardware_service.py --help
python src/hardware_smoke_test.py --help
```

## 3. 连接 USB 并识别串口

1. 关闭阵列发射电源。
2. 将 CoreEP4CE6 的 USB-UART 连接到电脑。
3. 确认控制板已完成 FPGA 配置，且控制板和阵列供电符合硬件装配要求。
4. 执行：

```powershell
python src/hardware_ports.py
```

记录新出现的 `COM` 端口，例如 `COM7`。如果没有新端口，先排查 USB 数据线、USB-UART 驱动、板卡供电和设备管理器。

单块 CoreEP4CE6 使用 `fpga_256` 协议和 `230400` 波特率。该 FPGA 串口协议通常不返回确认字节，因此“端口可以打开”只证明 USB/UART 链路存在，不能证明阵列通道顺序或声场正确。

## 4. 创建现场 profile

复制模板，不修改仓库内模板：

```powershell
Copy-Item `
  src/hardware/sonicsurface/profiles/example_fpga_256.yaml `
  outputs/hardware/coreep4ce6_lab.yaml
```

先填写串口：

```yaml
name: "coreep4ce6-lab"
board_model: "coreep4ce6"
protocol: "fpga_256"
solver_channels: 256
solver_to_device: "vendor_sonicsurface_16x16"
mapping_verified: false
phase_sign: 1
global_phase_offset_rad: 0.0
disabled_device_channels: []
serial:
  port: "COM7"
  baudrate: 230400
```

`vendor_sonicsurface_16x16` 会自动读取 `third_party/SonicSurface/ControlSoftware/Python/SonicSurface.py` 中供应商给出的 `EMITTERS_ORDER`，不需要手工抄写 256 项。它是供应商默认 16 x 16 PCB 的映射，不能代替实物验证；在验证前 `mapping_verified` 必须保持 `false`。实际板卡若与供应商 PCB 布线不同，再将该字段替换为显式 256 项排列。逐阵元 `phase_offsets_rad` 仍须由实测标定得到。

## 5. USB/UART 连接确认：只允许全关闭

此阶段尚未验证通道映射，因此只允许发送全关闭帧。

终端 A 启动服务：

```powershell
python src/hardware_service.py `
  --profile outputs/hardware/coreep4ce6_lab.yaml `
  --live --live-off-only `
  --log-file outputs/hardware/sonicsurface_service.log
```

终端 B 查询状态：

```powershell
python src/hardware_client.py --status
```

必须看到：

```text
live: true
armed: false
off_only: true
mapping_verified: false
```

发送全关闭：

```powershell
python src/hardware_smoke_test.py --test off
```

检查日志：

```powershell
Get-Content outputs/hardware/sonicsurface_service.log -Tail 20
```

日志应包含 `array_off`、`transmitted: true`、`frame_bytes: 258` 和 `board_model: coreep4ce6`。此步骤验证服务成功打开串口并写入一帧，但不证明 FPGA 已正确驱动 256 个换能器。

关闭服务后，再继续下面的映射验证。

## 6. 建立通道映射

`solver_to_device` 的含义是：

```text
solver_to_device[solver_index] = fpga_channel_index
```

求解器索引为 row-major：

```text
solver_index = row * 16 + column
```

建议按照下列顺序建立映射：

1. 使用供应商已知可工作的单焦点程序，确认阵列整体可发射。
2. 准备可观测方式：声压麦克风扫描、热敏/纸片响应或其他适合本实验室的空间声场观测手段。
3. 在临时校准工具中一次激励少量通道，记录对应物理阵元位置。
4. 以供应商 `EMITTERS_ORDER` 为初值，对照观测结果确认或修正 `solver_index -> fpga_channel_index`。
5. 若需要修正，将最终表写入 `solver_to_device`，确认它是 `0..255` 的完整无重复排列。

在没有可观测的单通道效果时，至少使用相位坡度和棋盘格的空间模式，与供应商程序输出的模式对照。不要因为 USB 写入成功就假设映射正确。

## 7. 验证相位方向、全局偏置与校准

映射尚未验证时，profile 必须继续保持：

```yaml
mapping_verified: false
phase_sign: 1
global_phase_offset_rad: 0.0
phase_offsets_rad: [0.0, ... 共 256 项]
```

启动短时诊断服务：

```powershell
python src/hardware_service.py `
  --profile outputs/hardware/coreep4ce6_lab.yaml `
  --live --live-calibration
```

`--live-calibration` 仅接受 `hardware_smoke_test.py` 标记为诊断用途的图样；常规优化相位客户端会被服务拒绝。每次诊断图样都必须显式确认，并会在指定时间后自动发送全关闭。

### 6.1 均匀相位

```powershell
python src/hardware_smoke_test.py `
  --test uniform --confirm-live --hold-seconds 1
```

用途：确认所有通道被同时更新。它不用于判断聚焦质量。

### 6.2 棋盘格

```powershell
python src/hardware_smoke_test.py `
  --test checkerboard --confirm-live --hold-seconds 1
```

用途：相邻阵元相差 π。若空间模式与预期不符，优先检查 `solver_to_device` 的行列方向、镜像和转置。

### 6.3 相位坡度

```powershell
python src/hardware_smoke_test.py `
  --test ramp_x --confirm-live --hold-seconds 1

python src/hardware_smoke_test.py `
  --test ramp_y --confirm-live --hold-seconds 1
```

用途：确认阵列两个轴的方向、行列顺序和可能的镜像。比较 `ramp_x` 与 `ramp_y` 的主瓣偏转方向，修正映射后重复测试。

### 6.4 中心几何聚焦

```powershell
python src/hardware_smoke_test.py `
  --test focus_center --focus-z-m 0.12 `
  --confirm-live --hold-seconds 1
```

用途：判断 `phase_sign` 与 `global_phase_offset_rad`。该脚本采用 10 mm pitch、阵列中心为原点、阵面位于 `z=0` 的自由场几何相位。若焦点方向相反或不能聚焦，先尝试将 `phase_sign` 由 `1` 改为 `-1`，再检查通道映射。

确认几何聚焦后，测量逐阵元或逐分块相位误差，填入 `phase_offsets_rad`。每次修改校准后都应重复中心聚焦测试。

## 8. 固化 profile

仅在以下项目均完成后，才保留：

```yaml
mapping_verified: true
```

完成标准：

1. 256 通道映射完整且无重复。
2. `ramp_x`、`ramp_y` 的物理方向与 solver 坐标一致。
3. `focus_center` 在预期位置形成可重复焦点。
4. 相位符号、全局偏置和逐阵元校准已写入 profile。
5. 已记录 CoreEP4CE6 固件/bitstream 版本、串口号和日志文件。

## 9. 生成并下发优化相位

先运行 16 x 16 优化：

```powershell
python src/main.py `
  --config src/examples/phase_oblique_reflecting_x_16x16.yaml
```

检查输出：

```text
outputs/phase_oblique_reflecting_x_16x16/
  result_best_phases_rad.npy
  result_final_phases_rad.npy
  result_phase_export.json
```

保持已验证的 live 服务运行，然后发送推荐相位：

```powershell
python src/hardware_client.py `
  --phase-file `
  outputs/phase_oblique_reflecting_x_16x16/result_best_phases_rad.npy
```

紧急关闭：

```powershell
python src/hardware_smoke_test.py --test off
```

每次实验保存服务日志、phase manifest、profile 副本、量化帧 JSON、测量数据和固件版本。比较几何相位、响应对齐与优化相位时，必须使用相同的测量位置、发射时间和声压指标。
