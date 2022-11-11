#!/bin/bash
kill -9 $(pgrep xt_main)
kill -9 $(pgrep xt_explorer)
kill -9 $(pgrep xt_broker)
kill -9 $(pgrep plasma_store_se)
rm /home/xys/bolt/test/ppo_cnn.tflite
rm /home/xys/bolt/test/ppo_cnn_f32.bolt


