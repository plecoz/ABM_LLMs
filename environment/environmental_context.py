"""
Environmental Context Example Usage

This file demonstrates how to use the environmental context features
that have been added to the FifteenMinuteCity model.

The environmental context includes:
- Temperature (in Celsius) with realistic stochastic modeling
- Time period (night, dawn, sunrise, daytime)
- Automatic time period updates based on simulation time
- Weather descriptions based on temperature

The temperature model includes:
- Sinusoidal daily cycle (peak at 2:30 PM, minimum at 6:30 AM)
- Base temperature as daily average
- Random daily variation (different each day)
- Small hourly noise for realistic fluctuations
"""

def demonstrate_environmental_context(model):
    """
    Demonstrate how to use environmental context features.
    
    Args:
        model: FifteenMinuteCity model instance
    """
    print("=== Environmental Context Demo ===")
    
    # 1. Set base temperature and model parameters
    print("\n1. Setting base temperature to 22°C with moderate variation")
    model.set_temperature(22.0)
    model.set_temperature_model_parameters(
        daily_amplitude=6.0,  # ±6°C daily variation
        daily_epsilon=2.0,    # ±2°C random daily variation
        hourly_noise=0.5      # ±0.5°C hourly noise
    )
    
    # 2. Show temperature info
    print("\n2. Temperature model information:")
    temp_info = model.get_temperature_info()
    print(f"   Base temperature: {temp_info['base_temperature']}°C")
    print(f"   Current temperature: {temp_info['current_temperature']:.1f}°C")
    print(f"   Daily amplitude: ±{temp_info['daily_amplitude']}°C")
    print(f"   Daily variation: ±{temp_info['daily_epsilon']}°C")
    print(f"   Hourly noise: ±{temp_info['hourly_noise_std']}°C")
    
    # 3. Show temperature predictions for different hours
    print("\n3. Expected temperature pattern (without noise):")
    key_hours = [0, 6, 9, 12, 15, 18, 21]
    for hour in key_hours:
        predicted_temp = model.predict_temperature_for_hour(hour)
        print(f"   {hour:02d}:00 -> {predicted_temp:.1f}°C")
    
    # 4. Demonstrate time period setting
    print("\n4. Setting time period to 'night'")
    model.set_time_period("night")
    
    # 5. Get environmental context
    print("\n5. Getting environmental context:")
    env_context = model.get_environmental_context()
    print(f"   Temperature: {env_context['temperature']:.1f}°C")
    print(f"   Time period: {env_context['time_period']}")
    print(f"   Weather: {env_context['weather_description']}")
    print(f"   Hour: {env_context['hour']}")
    
    print("\n=== Demo Complete ===")

def demonstrate_temperature_realism():
    """
    Demonstrate how the temperature model compares to real-world patterns.
    """
    print("""
=== Temperature Model Realism ===

The implemented temperature model uses established meteorological principles:

1. **Sinusoidal Daily Cycle**: 
   - Peak at 2:30 PM (solar heating maximum with thermal lag)
   - Minimum at 6:30 AM (maximum radiative cooling)
   - This matches real-world diurnal temperature patterns

2. **Stochastic Components**:
   - Daily random variation (weather fronts, cloud cover)
   - Hourly noise (micro-meteorological effects)
   - Base temperature (seasonal/climatic average)

3. **Temperature Amplitude**:
   - Default ±6°C matches typical urban temperature ranges
   - Coastal cities: ±4-6°C (ocean moderates temperature)
   - Continental cities: ±8-12°C (greater temperature swings)

4. **Compared to Real Meteorological Models**:
   - Simplified version of sinusoidal harmonic models
   - Similar to models used in building energy simulation
   - More realistic than constant temperature assumptions
   - Less complex than full numerical weather prediction

5. **Common Meteorological Temperature Models**:
   - Fourier series (multiple harmonics for seasonal + daily)
   - Autoregressive models (AR/ARMA for time series)
   - Empirical models (regression from historical data)
   - Physical models (energy balance equations)

Our model is appropriate for urban simulation studies where:
- Realistic daily temperature variation is needed
- Computational efficiency is important
- Detailed weather forecasting is not required
""")

def example_usage_in_simulation():
    """
    Example of how to use the stochastic temperature model in a simulation.
    """
    print("""
=== Example Usage in Simulation ===

# Setup different climate scenarios:

# Mediterranean climate (mild, moderate variation)
model.set_temperature(18.0)  # Base temperature
model.set_temperature_model_parameters(
    daily_amplitude=5.0,      # ±5°C daily variation
    daily_epsilon=1.5,        # ±1.5°C day-to-day variation
    hourly_noise=0.3          # ±0.3°C hourly noise
)

# Continental climate (more extreme)
model.set_temperature(15.0)  # Base temperature
model.set_temperature_model_parameters(
    daily_amplitude=8.0,      # ±8°C daily variation
    daily_epsilon=3.0,        # ±3°C day-to-day variation
    hourly_noise=0.5          # ±0.5°C hourly noise
)

# Tropical climate (warm, stable)
model.set_temperature(28.0)  # Base temperature
model.set_temperature_model_parameters(
    daily_amplitude=4.0,      # ±4°C daily variation
    daily_epsilon=1.0,        # ±1°C day-to-day variation
    hourly_noise=0.2          # ±0.2°C hourly noise
)

# During simulation:
# - Temperature updates automatically every hour
# - Concordia agents receive temperature context
# - Temperature affects path and movement decisions

# Example agent considerations:
# - Hot weather (>30°C): Prefer shaded routes, indoor activities
# - Cold weather (<10°C): Prefer shortest routes, warm destinations
# - Night time: Prefer well-lit, safer routes
# - Dawn/sunrise: May affect visibility and route preferences

# Monitor temperature:
temp_info = model.get_temperature_info()
print(f"Current: {temp_info['current_temperature']:.1f}°C")
print(f"Weather: {temp_info['weather_description']}")
""")

def simulate_temperature_day(model, show_plot=False):
    """
    Simulate a full day of temperature changes.
    
    Args:
        model: FifteenMinuteCity model instance
        show_plot: Whether to show a matplotlib plot of temperature
    """
    print("\n=== Simulating 24-Hour Temperature Pattern ===")
    
    # Store original values
    original_hour = model.hour_of_day
    original_temp = model.temperature
    
    # Simulate temperatures for 24 hours
    hours = list(range(24))
    temperatures = []
    
    for hour in hours:
        model.hour_of_day = hour
        model._update_temperature()
        temperatures.append(model.temperature)
        print(f"Hour {hour:02d}: {model.temperature:.1f}°C ({model._get_weather_description()})")
    
    # Restore original values
    model.hour_of_day = original_hour
    model.temperature = original_temp
    
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(hours, temperatures, 'b-', linewidth=2, label='Actual Temperature')
            
            # Plot expected pattern without noise
            expected_temps = [model.predict_temperature_for_hour(h) for h in hours]
            plt.plot(hours, expected_temps, 'r--', linewidth=1, label='Expected (no noise)')
            
            plt.xlabel('Hour of Day')
            plt.ylabel('Temperature (°C)')
            plt.title('24-Hour Temperature Pattern')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(0, 23)
            plt.xticks(range(0, 24, 3))
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available - skipping plot")
    
    print("=== End Temperature Simulation ===")

if __name__ == "__main__":
    print("This file demonstrates environmental context usage.")
    print("Import these functions and call them with your model instance.")
    demonstrate_temperature_realism()
    example_usage_in_simulation() 