import setup_path
import airsim

# connect to the AirSim simulator
client = airsim.CarClient(ip = "192.168.2.1")
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
car_controls = airsim.CarControls()

client.setCarControls(car_controls)

print("reset")
client.reset()

client.setCarControls(car_controls)


