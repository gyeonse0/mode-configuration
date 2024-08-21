# Correcting and running the provided code to generate the plot as specified
import matplotlib.pyplot as plt

# Define the time and altitude data to reflect the gradual changes seen in the image
time = [0, 1, 6, 7, 8, 11, 12, 15, 16, 18, 19]
altitude = [0, 100, 100, 0, 100, 100, 0, 0, 100, 100, 0]

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(time, altitude)
plt.xlabel('Time (min)')
plt.ylabel('Altitude (m)')
plt.title('Drone Altitude over Time')
plt.xticks(range(0, 20))  # Setting x-axis ticks from 0 to 19 at 1 minute intervals
plt.grid(True)

# Annotating specific nodes
plt.annotate('Takeoff Node', xy=(0, 0), xytext=(0.5, 20), arrowprops=dict(arrowstyle='->', lw=1.5))
plt.annotate('Customer Node #1', xy=(7, 0), xytext=(4, 20), arrowprops=dict(arrowstyle='->', lw=1.5))
plt.annotate('Customer Node #2\n(Waiting for Landing Node)', xy=(12, 0), xytext=(8, 20), arrowprops=dict(arrowstyle='->', lw=1.5))
plt.annotate('Landing Node', xy=(19, 0), xytext=(17, 20), arrowprops=dict(arrowstyle='->', lw=1.5))

plt.show()