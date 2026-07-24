# data Script CHOP — cook callback

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from td import scriptCHOP


def cook(scriptOp: scriptCHOP) -> None:
    ext    = parent.TDOAK.ext.OAKManagerExt
    status = ext.GetStatus()
    imu    = ext.GetImu()

    scriptOp.clear()
    scriptOp.appendChan('connected').vals  = [1.0 if status['connected'] else 0.0]
    scriptOp.appendChan('fps').vals        = [status['fps']]
    scriptOp.appendChan('latency_ms').vals = [status['latency_ms']]

    if imu:
        scriptOp.appendChan('imu_accel_x').vals = [imu['accel'][0]]
        scriptOp.appendChan('imu_accel_y').vals = [imu['accel'][1]]
        scriptOp.appendChan('imu_accel_z').vals = [imu['accel'][2]]
        scriptOp.appendChan('imu_gyro_x').vals  = [imu['gyro'][0]]
        scriptOp.appendChan('imu_gyro_y').vals  = [imu['gyro'][1]]
        scriptOp.appendChan('imu_gyro_z').vals  = [imu['gyro'][2]]
    else:
        for ch in ('imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                   'imu_gyro_x',  'imu_gyro_y',  'imu_gyro_z'):
            scriptOp.appendChan(ch).vals = [0.0]
