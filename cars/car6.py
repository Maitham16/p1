from kafka import KafkaProducer
import json
import time
import random
import datetime

# Constants
BASE_SPEED = 90  # km/h
CONSUMPTION_MODIFIER = 0.002  # arbitrary factor to simulate increased consumption at higher speeds
CHARGING_STATIONS = [150, 270, 460]  # Charging station locations
BATTERY_LIFE_DEGRADATION_RATE = 0.001 # % per km
DEGRADATION_FACTOR = 0.5    # arbitrary factor to simulate increased consumption at lower battery life

WEATHER_CONDITIONS = ['sunny', 'rainy', 'snowy']
TRAFFIC_CONDITIONS = ['light', 'moderate', 'heavy']
ROAD_GRADIENT = ['flat', 'uphill', 'downhill']
EMERGENCY_SITUATIONS = ['none', 'accident_ahead', 'road_closure']

SIMULATION_SPEED_FACTOR = 60  # 1 = real time, 60 = 1 hour per second, 3600 = 1 day per second 
real_start_time = datetime.datetime.now()
simulated_start_time = datetime.datetime.now()

def create_producer():
    return KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def get_simulated_timestamp(real_start_time, simulated_start_time, speed_factor):
    real_elapsed = datetime.datetime.now() - real_start_time
    simulated_elapsed = real_elapsed * speed_factor
    return simulated_start_time + simulated_elapsed

def calculate_distance_to_next_station(location):
    next_station = next((station for station in CHARGING_STATIONS if station > location), None)
    if next_station:
        return next_station - location
    return float('inf')  # No next station found

def update_car_data_while_charging(data):
    data['charge'] += 30  # 30% charge per 30 minutes
    if data['charge'] > 100:   # Ensure charging does not exceed 100%
        data['charge'] = 100
    if random.random() < 0.5:  # 50% chance to stop charging each time step
        data['charging'] = False

def update_car_data_based_on_weather(data, modified_consumption):
    weather = random.choice(WEATHER_CONDITIONS)
    if weather == 'rainy':
        modified_consumption *= 1.1
    elif weather == 'snowy':
        modified_consumption *= 1.2
    return modified_consumption

def update_car_data_based_on_traffic(data, modified_consumption):
    traffic = random.choice(TRAFFIC_CONDITIONS)
    if traffic == 'moderate':
        data['current_speed'] *= 0.8
        modified_consumption *= 1.1
    elif traffic == 'heavy':
        data['current_speed'] *= 0.6
        modified_consumption *= 1.2
    return modified_consumption

def update_car_data_based_on_gradient(data, modified_consumption):
    gradient = random.choice(ROAD_GRADIENT)
    if gradient == 'uphill':
        modified_consumption *= 1.2
    elif gradient == 'downhill':
        modified_consumption *= 0.8
    return modified_consumption

def handle_emergency_situations(data):
    weighted_emergency_list = ['none'] * 95 + ['accident_ahead', 'road_closure']
    emergency = random.choice(weighted_emergency_list)
    
    if emergency == 'accident_ahead':
        data['car_status'] = 'stopped'
        data['current_speed'] = 0
        data['emergency_duration'] = random.randint(5, 15)  # Random duration between 5 to 15 minutes
    elif emergency == 'road_closure':
        data['car_status'] = 'rerouting'
        data['current_speed'] *= 0.5
        data['emergency_duration'] = random.randint(5, 15)  # Random duration between 5 to 15 minutes

def update_car_data_while_moving(data, distance_step):
    # Updating location, distance covered, and current speed
    data['location'] += distance_step # km
    data['distance_covered'] += distance_step # km

    speed_factor = data['current_speed'] / BASE_SPEED 
    adjusted_consumption = data['consumption'] * speed_factor + (CONSUMPTION_MODIFIER * (speed_factor - 1))
    modified_consumption = adjusted_consumption * (1 + ((100 - data['battery_life']) / 100) * DEGRADATION_FACTOR)
    
    data['charge'] -= modified_consumption * distance_step / data['battery_capacity'] * 100
    data['battery_life'] -= BATTERY_LIFE_DEGRADATION_RATE

    # Calculate distance to next charging point and add to data
    data['distance_to_charging_point'] = calculate_distance_to_next_station(data['location'])

    modified_consumption = update_car_data_based_on_weather(data, modified_consumption)
    modified_consumption = update_car_data_based_on_traffic(data, modified_consumption)
    modified_consumption = update_car_data_based_on_gradient(data, modified_consumption)
    handle_emergency_situations(data)

    weather = random.choice(WEATHER_CONDITIONS)
    traffic = random.choice(TRAFFIC_CONDITIONS)
    gradient = random.choice(ROAD_GRADIENT)
    emergency = random.choice(EMERGENCY_SITUATIONS)

    # Check if the current location is approaching a charging station and if charge is low
    if data['charge'] <= 30:
        next_station = next((station for station in CHARGING_STATIONS if station > data['location']), None)
        if next_station:
            distance_to_next_station = next_station - data['location']
            charge_required_to_reach_station = distance_to_next_station / data['battery_capacity'] * data['consumption'] * 100


            if data['charge'] >= charge_required_to_reach_station:
                data['location'] = next_station
                data['charging'] = True
                data['car_status'] = "Charging"
            else:
                distance_possible = data['charge'] / data['consumption'] * data['battery_capacity'] / 100
                data['location'] += distance_possible
                data['charge'] = 0
                data['charging'] = False
                data['car_status'] = "Moving"

    # Randomly increase or decrease the current speed
    speed_variation = 5 # km/h
    min_speed = 70
    max_speed = 200
    current_speed = int(data['current_speed'])
    if current_speed <= min_speed:
        new_speed = random.randint(min_speed, min(max_speed, min_speed + speed_variation))
    elif current_speed >= max_speed:
        new_speed = random.randint(max(min_speed, max_speed - speed_variation), max_speed)
    else:
        new_speed = random.randint(max(min_speed, current_speed - speed_variation),
                                    min(max_speed, current_speed + speed_variation))
    data['current_speed'] = new_speed

    # Update the node based on the current location
    if data['location'] < 150:
        data['node'] = 'Node 1'
    elif data['location'] < 280:
        data['node'] = 'Node 2'
    elif data['location'] < 420:
        data['node'] = 'Node 3'
    elif data['location'] < 600:
        data['node'] = 'Node 4'
    else:
        print("Car has reached its destination.")

     # After the existing code, add logic to handle resuming speed or rerouting after emergency
    if data['emergency_duration'] > 0:
        time.sleep(data['emergency_duration'] * 60 / SIMULATION_SPEED_FACTOR)
        data['emergency_duration'] = 0
        if emergency == 'accident_ahead':
            data['car_status'] = 'moving'
            data['current_speed'] = random.randint(50, 100)
        elif emergency == 'road_closure':
            data['car_status'] = 'moving'
            data['current_speed'] = random.randint(50, 100)

    return modified_consumption, weather, traffic, gradient, emergency, data['emergency_duration']

def main():
    producer = create_producer()

    # Initial data
    data = {
        'car_id': 'car3',
        'model': 'Model F',
        'current_speed': 70,
        'battery_capacity': 55,
        'charge': 94,
        'consumption': 0.11,
        'location': 0,
        'node': 'Node 1',
        'car_status': "moving",
        'distance_covered': 0,
        'battery_life': 100,
        'distance_to_charging_point': calculate_distance_to_next_station(0),
        'charging': False,
        'emergency_duration': 0
    }

    time_step = 1 # minutes per time step
    distance_step = data['current_speed'] * (time_step/60) # km per time step
    counter = 0 # time step counter

    while data['charge'] > 0 and data['distance_covered'] < 600:
        counter += 1
        
        if data['charging']:
            update_car_data_while_charging(data)
        else:
            modified_consumption, weather, traffic, gradient, emergency, emergency_duration = update_car_data_while_moving(data, distance_step)

        # Add new fields to the data dictionary
        data['weather'] = weather
        data['traffic'] = traffic
        data['road_gradient'] = gradient
        data['emergency'] = emergency
        data['emergency_duration'] = emergency_duration
        
        # Recalculate the distance_step based on current speed
        distance_step = data['current_speed'] * (time_step/60)

        # Ensure charge does not go negative
        if data['charge'] < 0:
            data['charge'] = 0
            print("Car has run out of charge and has stopped.")
            break

        data['timestamp'] = get_simulated_timestamp(real_start_time, simulated_start_time, SIMULATION_SPEED_FACTOR).strftime('%Y-%m-%d %H:%M:%S')

        # Print current status
        print(f"Time step: {counter*time_step} minutes")
        print(f"Current speed: {data['current_speed']} km/h")
        print(f"Distance covered in this step: {distance_step} km")
        print(f"Charge consumed in this step: {modified_consumption * distance_step} %")
        print(f"Remaining charge: {data['charge']} %")
        print(f"Total distance covered: {data['distance_covered']} km")
        print(f"Battery life: {data['battery_life']} %")
        print(f"Weather: {weather}")
        print(f"Traffic: {traffic}")
        print(f"Road Gradient: {gradient}")
        print(f"Emergency Situation: {emergency}")
        print(f"Emergency Duration: {emergency_duration} minutes")
        print("--------------------------")
        
        # Send data to Kafka
        producer.send('electric_car_data', data)
        if data['node'] == 'Node 1':
            producer.send('node1_data', data)
        elif data['node'] == 'Node 2':
            producer.send('node2_data', data)
        elif data['node'] == 'Node 3':
            producer.send('node3_data', data)
        else:
            producer.send('node4_data', data)
        
        time.sleep(time_step * 60 / SIMULATION_SPEED_FACTOR)

if __name__ == "__main__":
    main()