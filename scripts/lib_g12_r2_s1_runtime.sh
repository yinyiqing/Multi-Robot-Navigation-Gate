#!/usr/bin/env bash

g12_r2_s1_group_exists() {
  local pgid="$1"
  ps -eo pgid= | awk -v wanted="$pgid" '$1 == wanted { found = 1 } END { exit !found }'
}

g12_r2_s1_ports_are_free() {
  local ros_port="$1"
  local gazebo_port="$2"
  ! ss -ltnH | awk -v ros="$ros_port" -v gazebo="$gazebo_port" '
    {
      endpoint = $4
      sub(/^.*:/, "", endpoint)
      if (endpoint == ros || endpoint == gazebo) {
        found = 1
      }
    }
    END { exit !found }
  '
}

g12_r2_s1_stop_stage() {
  local stage_pid="$1"
  local ros_port="$2"
  local gazebo_port="$3"
  local attempt

  [[ "$stage_pid" =~ ^[0-9]+$ ]] || return 0

  if g12_r2_s1_group_exists "$stage_pid"; then
    kill -TERM -- "-$stage_pid" 2>/dev/null || true
  fi

  for attempt in $(seq 1 30); do
    if ! g12_r2_s1_group_exists "$stage_pid" \
        && g12_r2_s1_ports_are_free "$ros_port" "$gazebo_port"; then
      return 0
    fi
    sleep 1
  done

  echo "Stage process group $stage_pid did not stop after SIGTERM; escalating" >&2
  kill -KILL -- "-$stage_pid" 2>/dev/null || true
  sleep 2

  # ROS/Gazebo children can create new process groups. These two ports are
  # reserved exclusively for this serialized diagnostic.
  fuser -k -TERM "${ros_port}/tcp" "${gazebo_port}/tcp" >/dev/null 2>&1 || true
  sleep 2
  fuser -k -KILL "${ros_port}/tcp" "${gazebo_port}/tcp" >/dev/null 2>&1 || true

  for attempt in $(seq 1 10); do
    if ! g12_r2_s1_group_exists "$stage_pid" \
        && g12_r2_s1_ports_are_free "$ros_port" "$gazebo_port"; then
      return 0
    fi
    sleep 1
  done

  echo "Failed to release S1 ROS/Gazebo runtime for process group $stage_pid" >&2
  return 1
}
