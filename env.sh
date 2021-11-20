export CARLA_ROOT="./CARLA"
export PYTHONPATH="."
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla/dist/$(ls ${CARLA_ROOT}/PythonAPI/carla/dist | grep py3.)"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI/carla"