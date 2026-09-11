# 本机硬件标定与部署

本目录说明如何在当前电脑上从零搭建 AcousticField 标定环境，完成 16 x 16 阵列的通道与相位标定，并将结果接入本项目的本地 S/C 服务。

标定结果尚不存在。完成本手册前，禁止使用 `--live` 向阵列发送优化相位。

## 目标与边界

```text
AcousticField
  -> 建立阵列几何、测量通道和相位校正
  -> 保存原始标定档案
  -> 导出 32 档相位校正

本项目
  -> 导入标定结果
  -> 复现实测板端 SimpleFPGA 协议
  -> 通过本地 HTTP S/C 服务下发 result_best_phases_rad.npy
```

板端固件和 AcousticField 协议来自 `third_party/Ultraino/`。本流程不修改主控制板固件。

## 1. 本机软件环境

当前工程的 NetBeans 配置指定：

```text
javac.source = 11
javac.target = 11
```

因此必须安装 **JDK 11 x64**。先完成安装并在新的 PowerShell 中验证：

```powershell
java -version
javac -version
```

两个命令都应报告版本 11。

然后安装支持 Java 11 项目的 Apache NetBeans，并打开：

```text
third_party/Ultraino/AcousticFieldSim
```

在 NetBeans 的项目属性中确认 Java Platform 为 JDK 11，再执行 `Clean and Build`。工程依赖已随源码放在：

```text
third_party/Ultraino/AcousticFieldSim/jars/
```

不要把 JDK 26 当作该旧工程的唯一编译环境；若使用较新的 NetBeans 启动器，仍应为该项目选择 JDK 11 平台。

## 2. 标定所需硬件

仅连接主阵列和电脑不能完成逐阵元相位标定。还需要：

1. 16 x 16 超声阵列及其现有主控制板。
2. 主控制板的 USB 串口连接。
3. 一个 Arduino Nano。
4. 相位检测电路与传感器，原理图位于：

```text
third_party/Ultraino/Arduino phase detector/schematics.jpg
```

5. 主控制板同步信号接 Arduino Nano `D2`。
6. 测量麦克风或检波电路输出接 Arduino Nano `A0`。
7. Arduino Nano 的 USB 串口连接。

将下列程序烧录到 **Arduino Nano**：

```text
third_party/Ultraino/Arduino phase detector/PhaseDetector/PhaseDetector.ino
```

该 Nano 以 `1000000` 波特率工作，采集每次触发后的 32 个 ADC 点。它不是阵列主控制板，不能替代主控制板固件。

主控制板和 Arduino Nano 必须出现在两个不同的 `COM` 端口。

## 3. 建立阵列几何

先在 AcousticField 中建立或导入真实 16 x 16 阵列：

1. 实测阵元中心间距、阵面方向、坐标原点和阵列外形。
2. 用 AcousticField 的阵元导入/创建功能生成 256 个阵元。
3. 为每个阵元设置对应的真实空间坐标。
4. 保存未标定基线：

```text
outputs/hardware/array_uncalibrated.xml.gz
```

不要仅按默认规则网格假设坐标。后续 Helmholtz 求解器的阵元位置必须与该阵列档案使用同一个坐标系和物理尺度。

## 4. 运行标定工具

本仓库中的标定窗口源码为：

```text
third_party/Ultraino/AcousticFieldSim/src/acousticfield3d/gui/AssignTransducers.java
```

该文件已做一个局部兼容修复：测试扫描阵元同时设置 `orderNumber=i`。原因是 SonicSurface 的 `SimpleFPGA` 按 `orderNumber` 寻址，原始窗口只设置 `driverPinNumber` 会导致扫描不完整。该修复不改变主控制板固件或 UART 协议。

在 AcousticField 中：

1. 在 `TransControlPanel` 选择 `SimpleFPGA`，连接主控制板串口。
2. 打开 `Transducer Assignment` 窗口。
3. 点击 `Connect`，选择 Arduino Nano 串口；`Speed` 设置为 `1000000`。
4. 将 `Max trans` 设置为 `256`。
5. 点击 `Res` 清空旧相位修正。
6. 勾选 `phaseCorr`，不要勾选 `OnlyPolarity`。
7. 保持 `autoassign` 勾选，点击 `Start`。
8. 对每个阵元点击 `Check`。窗口会扫描候选通道、读取 Nano 的 32 点波形，并记录幅值最高的通道及其相位。

完整相位模式下，工具写入：

```text
phaseCorrection = -measured_phase / pi
```

`OnlyPolarity` 仅产生 `0` 或 `1`，只适合排查极性反接，不可替代完整相位标定。

若出现 `timeout!!!`、幅值过低、重复通道或明显错误的相位，停止标定并检查同步线、A0 测量链路、串口和阈值。`MinAmp`、`MaxAmp` 默认值仅是原项目示例，必须根据当前测量链路重新设定。

## 5. 保存标定结果

标定结束后：

1. 点击 `Finish` 恢复阵元输出。
2. 通过 `File -> Save as` 保存完整档案：

```text
outputs/hardware/array_calibrated.xml.gz
```

该 `.xml.gz` 是唯一完整原始档案，包含阵元坐标、`orderNumber`、`driverPinNumber` 和 `phaseCorrection`。

3. 在标定窗口中将 `divs` 设为 `32`。
4. 点击 `export`。弹出的文本是 256 个逗号分隔的整数，范围为 `0..31`。
5. 将文本保存为：

```text
outputs/hardware/phase_corrections_bins.txt
```

`export` 不会自动写文件。`offsets` 输入框只有在已有 256 项附加档位补偿时才使用；首次标定时不要填入伪造值。

## 6. 导入到本项目

复制服务配置模板：

```powershell
Copy-Item src/hardware/ultraino_simple_fpga.example.yaml `
  outputs/hardware/ultraino_lab.yaml
```

将离散标定档位转换为服务使用的 `pi rad` 偏移：

```powershell
python src/hardware_import_calibration.py `
  --phase-bins outputs/hardware/phase_corrections_bins.txt `
  --config outputs/hardware/ultraino_lab.yaml `
  --solver-to-order identity
```

工具执行：

```text
phase_corrections_pi[i] = 2 * phase_bin[i] / 32
```

`identity` 仅在优化器的 16 x 16 行主序与 AcousticField 的 `orderNumber=0..255` 完全对应时成立。若阵列在任一侧存在旋转、镜像、行列交换或不同起点，创建 256 项映射文件：

```text
solver_index -> AcousticField orderNumber
```

然后改用：

```powershell
python src/hardware_import_calibration.py `
  --phase-bins outputs/hardware/phase_corrections_bins.txt `
  --solver-to-order outputs/hardware/solver_to_order.txt `
  --config outputs/hardware/ultraino_lab.yaml
```

导入不会自动将 `calibration_verified` 设置为 `true`。

## 7. 协议一致性验证

先不要发送优化结果。使用同一组中心几何聚焦相位分别通过 AcousticField 和本项目 dry-run/服务测试，确认：

1. 256 通道数量一致。
2. 相位档位均为 32。
3. 通道方向、行列顺序、镜像和坐标原点一致。
4. 声学焦点位置一致。

只有实测焦点一致后，才在 `outputs/hardware/ultraino_lab.yaml` 中填写实际 `serial.port` 并设置：

```yaml
calibration_verified: true
```

## 8. 实际下发优化相位

启动服务：

```powershell
python src/hardware_service.py `
  --config outputs/hardware/ultraino_lab.yaml --live
```

另开终端发送优化结果：

```powershell
python src/hardware_client.py `
  --phase-file outputs/<场景名>/result_best_phases_rad.npy
```

紧急关闭：

```powershell
python src/hardware_client.py --off
```

AcousticField 与 `hardware_service.py` 不能同时占用主控制板串口。
