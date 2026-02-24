#!/bin/bash
set -e

echo "Building..."
cargo build -p genesis-runtime
cargo build -p genesis-baker

ZONE=101
SOCK="/tmp/genesis_baker_${ZONE}.sock"
SHM="/dev/shm/genesis_shard_${ZONE}"

# override sim to trigger night fast
cat << INNER_EOF > tests/tmp/sim.toml
[world]
width_um = 3500
depth_um = 3500
height_um = 10250

[simulation]
seed = 42
master_seed = "123"
update_interval_ms = 1
tick_duration_us = 1000
total_ticks = 1000
sync_batch_ticks = 10
night_interval_ticks = 100
segment_length_voxels = 5
voxel_size_um = 10
global_density = 0.04
signal_speed_um_tick = 50

[homeostasis]
enabled = true
tau = 1000.0
target_rate = 5.0
max_adjust = 0.01

[stdp]
enabled = true
a_plus = 0.01
a_minus = 0.012
tau_plus = 20.0
tau_minus = 20.0
INNER_EOF

echo "Starting baker daemon in background..."
rm -f $SOCK
target/debug/genesis-baker-daemon -z $ZONE \
  --sim tests/tmp/sim.toml \
  --blueprints zones/V1/blueprints.toml \
  --shard-dir baked/ &
BAKER_PID=$!

sleep 1

cat << INNER_EOF > tests/tmp/shard_${ZONE}.toml
zone_id = "$ZONE"
network_port = 8888

[world_offset]
x = 0
y = 0
z = 0

[dimensions]
w = 500
d = 500
h = 2000

[neighbors]
INNER_EOF

echo "Running node for 2 seconds (should trigger Night Phase at tick 100)..."
timeout 2 target/debug/genesis-runtime \
  --config tests/tmp/shard_${ZONE}.toml \
  --simulation tests/tmp/sim.toml \
  --blueprints config/zones/V1/blueprints.toml \
  --baked-dir baked/ \
  --baker-socket $SOCK || true

echo "Killing baker..."
kill -9 $BAKER_PID
rm -f $SOCK $SHM

echo "Done."
