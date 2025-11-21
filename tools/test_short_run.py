import parameter

parameter.MAX_EPS_STEPS = 20

import datetime
import logging
import os

# 啟用詳細日誌以觀察 rebuild 訊息
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

from worker import Worker

if __name__ == "__main__":
    # 使用 3 台 robot，地圖索引 1，儲存影片
    worker = Worker(
        global_step=0,
        agent_num=3,
        map_index=1,
        plot=False,
        save_video=True,
        force_sync_debug=False,
    )
    success, steps = worker.run_episode(curr_episode=0)
    print(f"TEST_RUN finished: success={success}, steps={steps}")
