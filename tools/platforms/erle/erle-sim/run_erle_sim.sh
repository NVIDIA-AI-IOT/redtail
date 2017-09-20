ERLE_SIM_NAME=erle-sim
ERLE_SIM_ID=`docker ps -aqf "name=^/${ERLE_SIM_NAME}$"`
ERLE_SIM_HOST_DATA_DIR=/data/
if [ -z "${ERLE_SIM_ID}" ]; then
    echo "Creating new Erle Sim container."
    docker run -it --network=host -v ${ERLE_SIM_HOST_DATA_DIR}:/data/:rw -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix${DISPLAY} --name=${ERLE_SIM_NAME}  erle-sim:indigo bash
else
    echo "Found Erle Sim container: ${ERLE_SIM_ID}."
    # Check if the container is already running and start if necessary.
    if [ -z `docker ps -qf "name=^/${ERLE_SIM_NAME}$"` ]; then
        xhost +local:${ERLE_SIM_ID}
        echo "Starting and attaching to Erle Sim container..."
        docker start ${ERLE_SIM_ID}
        docker attach ${ERLE_SIM_ID}
    else
        echo "Found running Erle Sim container, attaching bash to it..."
        docker exec -it --privileged ${ERLE_SIM_ID} bash
    fi
fi

