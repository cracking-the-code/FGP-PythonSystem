#!/bin/sh
#_mydir="$(pwd)"
SERVICE_NAME=KFSystemService
PATH_TO_JAR=Initializer.py
PID_PATH_NAME=/tmp/KFSystemService-pid

# Colors :3
NC='\033[0m'       # Text Reset
# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

case $1 in
	start)
		echo "${Green}
        
				██╗  ██╗███████╗███████╗
				██║ ██╔╝██╔════╝██╔════╝
				█████╔╝ █████╗  ███████╗
				██╔═██╗ ██╔══╝  ╚════██║
				██║  ██╗██║     ███████║
				╚═╝  ╚═╝╚═╝     ╚══════╝${NC}

                               ${Yellow}Kimbo's Forecasting System${NC}
                                   ${Yellow} Version Beta 1.0${NC}

                           A software by: KIMBO TECHNOLOGIES \n\n"
		
		echo "Starting the Kimbo's Forecasting System..."
		if [ ! -f $PID_PATH_NAME ]; then
			echo $PATH_TO_JAR
			nohup python3 $PATH_TO_JAR &
			echo $! > $PID_PATH_NAME
			echo "$SERVICE_NAME is started..."
		else
			echo "$SERVICE_NAME is already running..."
		fi
	;;
	stop)
		if [ -f $PID_PATH_NAME ]; then
			PID=$(cat $PID_PATH_NAME);
			echo "Stoping the Kimbo's Forecasting System..."
			echo "$SERVICE_NAME Stopping..."
			kill $PID;
			echo "$SERVICE_NAME stopped..."
			echo "Come back soon!"
			rm $PID_PATH_NAME
		else
			echo "$SERVICE_NAME is not running..."
		fi
	;;
	restart)
		if [ -f $PID_PATH_NAME ]; then
			echo "Restarting the Kimbo's Forecasting System..."
			PID=$(cat $PID_PATH_NAME);
			echo "$SERVICE_NAME stopping..."
			kill $PID;
			echo "$SERVICE_NAME stopped..."
			rm $PID_PATH_NAME
			echo "$SERVICE_NAME starting..."
			nohup python3 $PATH_TO_JAR /tmp 2>> /dev/null >> /dev/null &
			echo $! > $PID_PATH_NAME
			echo "$SERVICE_NAME is started..."
		else
			echo "$SERVICE_NAME is not running..."
		fi
	;;
esac
