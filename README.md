# usv-radio-control
控制双推进无人船。仿照之前红船控制代码。

# ArduRover 船控制脚本

适用于当前 ArduRover 4.5.6 双推进器船。参考原项目的通信、控制与遥测分层，使用 MAVLink。程序不会自动修改飞控参数。

## 安装与运行

在 VS Code 打开整个文件夹，使用 PowerShell 终端。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe boat.py ports
```

下面示例端口 COM3 来自最近一次连接。每次运行先列串口，USB 直连端口与数传端口分别选择。退出占用同一端口的 QGC 或其他脚本。当前安装依赖已在临时目录时，也可在当前终端先设置 PYTHONPATH 为 Join-Path $env:TEMP 'haihang-mavprobe'，再使用 python 运行。

## 状态与双向检查

```powershell
python boat.py status --port COM3 --duration 10
```

读取模式、解锁状态、接收机和失联参数，发送解锁前检查请求并显示飞控消息。收到心跳只证明船到电脑有数据。参数查询成功或收到对应命令回执，才能确认请求往返。

程序识别 type=11、autopilot=3 的 ArduRover 船。PX4 无人机连接会因不匹配而超时。

## 电机点动

先拆桨、固定船体并给电调供电。保持 Hold、已锁定。左电机为 Test Motor C，右电机为 D。

```powershell
python boat.py motor-test --port COM3 --motor left --percent 5 --duration 4
python boat.py motor-test --port COM3 --motor right --percent 5 --duration 4
```

每次只发送一次命令，避免无线重发造成反复启动或延长运行。默认 5%、4 秒，限制最多 10%、8 秒。由飞控执行有限时长测试，完成后采样输出是否恢复 1500。PWM 中位按这艘船当前配置为 1500。

ACK=0 表示接受。缺少回执时执行状态未知，不能立即认定未执行。PWM 变化证明飞控输出变化，实际电机运动仍需现场观察。磁场、罗盘不一致、遥控失联或其他解锁检查失败，应按消息处理原因。

## 键盘台架控制

```powershell
python boat.py keyboard --port COM3 --percent 5 --duration 30
```

该功能会请求正常解锁、进入 Manual 并发送 RC override。先确认左右推进器方向和 RC 通道映射。要求 FS_GCS_ENABLE=1、FS_ACTION=2，程序在运行时查询确认。当前恢复遥控配置后 FS_GCS_ENABLE=0，因此需要有意识地完成失联配置后才可运行此功能。RC override 必须被飞控允许，电脑 SYSID 使用 255，与 SYSID_MYGCS 对应。

终端取得焦点后，按住方向键 ↑/↓/←/→ 或 W/S/A/D，依靠系统按键重复前进、后退、左转、右转。

方向键控制的是固定幅度推力与转向输入，默认 5%。它尚未控制实际航速，持续按住不会自动增加推力。调整 --percent 可选择本次输入幅度，台架限制最高 10%。无按键重复超过 0.3 秒发送中位，空格回中，Esc 退出。初次按下到系统重复之间也可能短暂回中。30 秒后自动结束。方向仍需现场确认，SERVO 和 RC 反向配置会影响实际效果。

退出时尝试发送中位、Hold、释放 override 和正常锁定，并报告是否收到状态确认。链路中断时这些命令可能失败，需要飞控端失联保护兜底。该版本尚未验证实船键盘控制和断联停车，先完成拆桨台架验证。

## 遥测采集

```powershell
python boat.py record --port COM3 --duration 60
```

CSV 写到 data 目录，每条包含电脑单调时钟时间、消息名及原始字段，保留 ATTITUDE、GLOBAL_POSITION_INT、GPS_RAW_INT、SERVO_OUTPUT_RAW。时间戳为电脑接收时刻。ATTITUDE 角度和角速度采用 rad、rad/s，位置消息的坐标和速度保持 MAVLink 原始单位，分析时应明确换算及坐标系。

此版本采集当前已有消息流，缺少数据时需要另行设置消息频率。它尚未实现速度/角速度闭环，也没有完成水上系统辨识。

## 已知限制

- 串口号会随 USB 插口改变，使用 ports 确认。
- 数传近期存在请求超时，脚本显示反馈并保留失败，不通过自动电机命令重发掩盖问题。
- 仅靠电脑日志无法证明实物已停。有限时点动与失联停车必须分别验证。
- 原参考项目中值 4500、自定义 FF/FE 数据帧属于其下位机协议，本项目使用飞控 MAVLink 接口。
