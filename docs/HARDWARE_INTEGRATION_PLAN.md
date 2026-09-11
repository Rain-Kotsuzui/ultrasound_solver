# SonicSurface 硬件接入方案

## 当前状态

已完成软件侧接入：连续相位导出、本地 loopback 服务、硬件 profile 校验、32 级相位量化、`fpga_256` / `esp32_512` UART 组帧、离线帧导出和 dry-run 测试。

尚未完成真实硬件闭环：尚未取得并验证实际阵列拓扑、阵元物理坐标、求解器到 FPGA 的通道映射、相位符号、全局偏置、逐阵元校准、串口链路及实测声场。因此当前不能向未验证的设备发送优化结果。

本方案的目标是把本仓库的 phase-only 优化结果安全地下发到 SonicSurface 阵列，并用实测声场验证聚焦效果；它不替代 Helmholtz 求解器。

## 供应商仓库依据

供应商代码位于 `third_party/SonicSurface/`。

- `ControlSoftware/Python/SonicSurface.py` 提供了现有上位机 API，可通过串口直接发送相位。
- 单块 FPGA 板接收的相位帧为：

  ```text
  254, phase[0:256], 253
  ```

  `254` 表示开始写入缓存，`253` 表示提交到 FPGA 双缓冲。
- ESP32 双板控制器向两块 FPGA 转发 512 通道相位帧：

  ```text
  254, 192 + board_id, phase[board_id * 256 : (board_id + 1) * 256], 253
  ```

- FPGA 在每个 40 kHz 周期中使用 32 个相位槽：`0..31` 为有效相位，`32` 表示关闭通道。
- 供应商 16 x 16 PCB 的 BOM 指定 Waveshare `CoreEP4CE6`，Quartus 主板和从板工程均设置 FPGA 器件为 `EP4CE6E22C8`。
- 供应商固件存在 duty-cycle 路径，但已公开的串口 API 只发送相位和关闭通道。当前求解器为 phase-only，这与现有协议匹配。

## 当前关键不匹配

部署配置使用 `16 x 16 = 256` 阵元，与单块 CoreEP4CE6 的 256 通道相位帧一一对应。求解器相位向量、通道映射和逐阵元校准必须保持完整的 256 通道对应关系。

真实发送前必须记录：

1. 安装的是单块 256 通道板，还是双板 512 通道系统。
2. 所有阵元在统一实验坐标系中的实际位置和阵面法向。
3. solver index 到 device index 的双射映射；供应商 `EMITTERS_ORDER` 必须用实物验证。
4. 相位符号：发送 `phi`、`-phi`，或附加全局相位偏置。
5. 每通道相位校准偏置和坏阵元/禁用阵元。
6. 串口链路：直连 FPGA，还是连接 ESP32 控制器。

## 本地服务架构

```text
*_best_phases_rad.npy + *_phase_export.json
  -> src/hardware_client.py
  -> http://127.0.0.1:8765/v1/pattern
  -> src/hardware_service.py
  -> 映射、校准、量化、组帧
  -> 串口 -> SonicSurface
```

服务代码位于：

```text
src/hardware/
  sonicsurface/
    profile.py           # CoreEP4CE6 profile 加载、映射和校准参数校验
    protocol.py          # 纯相位量化和 UART 组帧
    transport.py         # 延迟打开的串口传输
    service.py           # loopback HTTP 服务，也是唯一串口拥有者
    client.py            # 提交求解结果的客户端
    export_result.py     # 离线 .npy + manifest -> 帧 JSON
    profiles/            # 模板和私有硬件 profile
```

服务仅监听 `127.0.0.1`，优化、可视化和批量实验进程均不导入串口传输。服务默认 dry-run；只有在 profile 设置 `mapping_verified: true` 后，`--live` 才允许打开串口。

服务端默认把 JSON 审计日志写入 `outputs/hardware/sonicsurface_service.log`。日志记录服务启动/停止、图样接受、全关闭、拒绝和异常；每条记录包含 `CoreEP4CE6` 板型、profile、来源、帧哈希、帧长度、关闭通道数和 dry-run/live 状态，不记录完整相位数组。日志按 5 MiB 轮转，保留最近 5 个备份。

## 已实现的软件安全边界

- 必须提供与设备通道数完全一致的相位向量。
- `solver_to_device` 必须是完整的双射排列。
- 相位先应用符号、全局偏置和逐阵元偏置，再映射到 `[0, 2π)` 并量化到最近的 32 级相位槽。
- 关闭通道固定编码为 `32`。
- `--live` 会拒绝 `mapping_verified: false` 的 profile。
- 自动化测试和模块导入不会打开串口。
- `/v1/off` 始终生成全关闭帧；dry-run 中仅返回帧摘要，不进行传输。

## 部署与验证顺序

### 1. 固化硬件约定

- 检查板卡标签、USB 串口和供电拓扑。
- 获取或测量阵元坐标。
- 为实际阵列建立 profile：阵元数、通道映射、相位偏置、禁用通道、串口参数和板卡拓扑。
- 将求解器阵列配置保持为真实部署尺寸 `16 x 16 = 256`。

完成标准：每个物理通道都有经验证的映射。

### 2. 离线编码验证

使用 `src/hardware_export.py` 从 `*_best_phases_rad.npy` 和 manifest 生成 JSON/hex 帧。

完成标准：输出帧与供应商协议字节格式一致；量化、环绕、禁用通道和 256/512 分板测试均通过。

### 3. 安全冒烟测试

在不修改 FPGA/ESP32 固件的前提下：

1. 先发送全关闭帧。
2. 使用供应商程序发送一个已知几何聚焦。
3. 用本服务发送同一几何聚焦。
4. 发送相位坡度和棋盘格，验证通道顺序。
5. 保持可立即执行的关闭命令和可断电电源开关。

完成标准：供应商程序与本服务在同一测量条件下产生一致焦点。

### 4. 标定与优化相位传输

- 根据实测峰值方向确定相位符号和全局偏置。
- 测量逐阵元或逐分块相位偏置，写入 profile。
- 用实测几何、校准和量化重新评估仿真。
- 在同一测量条件下依次比较几何相位、响应对齐和优化相位。
- 记录相位文件哈希、profile、固件版本、量化帧、时间戳和实测指标。

完成标准：优化相位在预先定义的实测指标上优于几何基线。

### 5. 是否修改固件

静态 phase-only 图样不需要重新烧录 FPGA。只有在下列需求经验证后，才考虑修改 Quartus 或 ESP32 固件：

- 需要逐通道幅值/duty 控制；
- 需要超过 32 级相位；
- 需要校验、确认或自定义帧格式；
- 需要更高频率的实时图样更新；
- 实际板卡拓扑与供应商协议不一致。

任何固件修改都必须保留可回滚的版本，并重新完成 smoke test。

## 使用方式

详细命令见 [`src/hardware/README.md`](../src/hardware/README.md)。实际硬件实验应先使用 dry-run profile；只有测量完成后才创建私有 `mapping_verified: true` profile。
