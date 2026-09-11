# 硬件接入

`hardware/sonicsurface/` 为 SonicSurface 提供本地服务边界。优化进程不直接访问串口，只向 loopback HTTP 服务提交连续相位；服务进程负责校验、校准、量化、组帧和可选串口传输。

空白 `CoreEP4CE6` 不能直接由 Python 控制：必须先通过 USB-Blaster 将 `third_party/SonicSurface/Firmware/FPGA primary/QuadrupleBuffer.qpf` 编译并下载到 FPGA。该工程目标器件为 `EP4CE6E22C8`；完整烧录和验证顺序见下方执行手册。

## 当前状态

- 已完成：相位量化、`fpga_256` / `esp32_512` 协议组帧、`CoreEP4CE6` 板型校验、配置校验、离线帧导出、本地服务和客户端。
- 已完成：未验证通道映射禁止 `--live`，默认 dry-run，只监听 `127.0.0.1`。
- 已完成：服务端轮转 JSON 审计日志，默认写入 `outputs/hardware/sonicsurface_service.log`。
- 未完成：真实阵列拓扑、通道顺序、相位符号、校准偏置、串口链路和实测声场验证。
- 未完成：对当前空白 CoreEP4CE6 的 FPGA 初始烧录及断电自启动验证。

因此，当前代码已经具备安全的软件接入路径，但尚不能宣称完成了真实硬件控制闭环。

## 服务架构

```text
求解结果 result_best_phases_rad.npy
  -> hardware_client.py
  -> http://127.0.0.1:8765/v1/pattern
  -> hardware_service.py
  -> 通道映射 + 校准 + 32 级量化
  -> UART 帧 -> SonicSurface
```

服务默认处于 dry-run 模式。只有显式传入 `--live`，且硬件 profile 中已设置 `mapping_verified: true`，服务才会打开串口并发送帧。

服务日志记录启动、相位请求、全关闭请求、实际发送、拒绝和异常；每条记录保存帧 SHA-256、字节数、来源、板型和 dry-run/live 状态，不保存完整相位向量。单个日志文件达到 5 MiB 后自动轮转，保留最近 5 个备份。

## 离线试运行

模板 profile 仅用于协议测试，其中 `mapping_verified: false`，无法启动 live 服务。

```powershell
python src/hardware_service.py `
  --profile src/hardware/sonicsurface/profiles/example_fpga_256.yaml `
  --log-file outputs/hardware/sonicsurface_service.log
```

在另一个进程中调用：

```powershell
python src/hardware_client.py --status
python src/hardware_client.py --off
python src/hardware_client.py `
  --phase-file outputs/<场景名>/result_best_phases_rad.npy
```

客户端会先验证相邻的 `*_phase_export.json`。服务拒绝通道数与 profile 不一致的相位向量；部署配置必须导出完整的 256 通道相位。

也可以生成可检查、完全不联网且不打开串口的帧文件：

```powershell
python src/hardware_export.py `
  --profile <已验证-profile.yaml> `
  --phase-file outputs/<场景名>/result_best_phases_rad.npy `
  --output outputs/<场景名>/device_frame.json
```

## 真实发送

完成通道顺序、相位符号、相位偏置、禁用阵元和串口参数测量后，创建私有 profile，设置 `mapping_verified: true`，再显式启动：

```powershell
python src/hardware_service.py `
  --profile <已验证-profile.yaml> --live
```

服务支持 SonicSurface 供应商定义的 `fpga_256` 和 `esp32_512` 组帧方式。任何模块在导入或自动化测试期间均不会打开串口。完整物理验证流程见 [`docs/HARDWARE_INTEGRATION_PLAN.md`](../../docs/HARDWARE_INTEGRATION_PLAN.md)。

对于供应商默认 16 x 16 PCB，profile 可使用 `solver_to_device: "vendor_sonicsurface_16x16"` 自动读取其 `EMITTERS_ORDER`，无需手填 256 项；该默认映射仍必须经过实物验证后才能设置 `mapping_verified: true`。

从 USB 连接、串口识别、全关闭链路确认到诊断图样与优化相位下发的完整命令，见 [`docs/HARDWARE_EXECUTION_MANUAL.md`](../../docs/HARDWARE_EXECUTION_MANUAL.md)。
