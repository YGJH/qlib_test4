#!/usr/bin/env python3
"""
Example usage of future prediction system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_future import FuturePredictionGenerator
from colors import *
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def load_trained_model():
    """Load your trained model - modify this function to load your actual model"""
    try:
        # Example: Load from Keras/TensorFlow
        # from tensorflow.keras.models import load_model
        # model = load_model('your_model.h5')
        
        # For demonstration, we'll create a dummy model
        print_yellow("Warning: Using dummy model for demonstration")
        print_yellow("Replace this with your actual trained model")
        
        class DummyModel:
            def predict(self, x, verbose=0):
                # Generate some dummy predictions
                batch_size = x.shape[0]
                prediction_steps = 480  # Adjust based on your model
                
                # Generate realistic-looking random predictions
                np.random.seed(42)  # For reproducible results
                predictions = np.random.normal(0, 0.02, (batch_size, prediction_steps))
                
                # Add some trend
                for i in range(prediction_steps):
                    predictions[:, i] += np.sin(i * 0.1) * 0.01
                
                return predictions
        
        return DummyModel()
        
    except Exception as e:
        print_red(f"Error loading model: {e}")
        return None

def visualize_predictions(results, save_plot=True):
    """Visualize the predictions"""
    if results is None:
        return
        
    try:
        import matplotlib.pyplot as plt
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Predicted prices
        timestamps = results['future_timestamps'][:100]  # Show first 100 predictions
        prices = results['predicted_prices'][:100]
        
        ax1.plot(timestamps, prices, 'b-', linewidth=2, label='Predicted Prices')
        ax1.axhline(y=results['current_price'], color='r', linestyle='--', 
                   label=f'Current Price: ${results["current_price"]:.2f}')
        ax1.set_title(f'{results["stock_symbol"].upper()} - Future Price Predictions')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Price change percentage
        changes = results['prediction_changes'][:100] * 100
        colors = ['green' if x >= 0 else 'red' for x in changes]
        
        ax2.bar(range(len(changes)), changes, color=colors, alpha=0.7)
        ax2.set_title('Predicted Price Changes (%)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Change (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_plot_{results['stock_symbol']}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print_green(f"✓ Plot saved as {filename}")
        
        plt.show()
        
    except ImportError:
        print_yellow("Matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print_red(f"Error creating visualization: {e}")

def predict_multiple_stocks(predictor, model, stocks, target_datetime):
    """Predict for multiple stocks"""
    print_green("=" * 80)
    print_green("BATCH PREDICTION FOR MULTIPLE STOCKS")
    print_green("=" * 80)
    
    all_results = {}
    
    for stock in stocks:
        print_cyan(f"\nPredicting for {stock.upper()}...")
        
        results = predictor.predict_future_prices(model, stock, target_datetime)
        if results is not None:
            all_results[stock] = results
            
            # Show summary
            current_price = results['current_price']
            avg_predicted = np.mean(results['predicted_prices'])
            max_predicted = np.max(results['predicted_prices'])
            min_predicted = np.min(results['predicted_prices'])
            
            print_green(f"  Current: ${current_price:.2f}")
            print_cyan(f"  Average predicted: ${avg_predicted:.2f}")
            print_cyan(f"  Range: ${min_predicted:.2f} - ${max_predicted:.2f}")
        else:
            print_red(f"  Failed to predict for {stock}")
    
    return all_results

def main():
    """Main function demonstrating the prediction system"""
    print_green("=" * 80)
    print_green("STOCK FUTURE PREDICTION DEMO")
    print_green("=" * 80)
    
    # Initialize predictor
    predictor = FuturePredictionGenerator()
    
    # Load metadata
    if not predictor.load_metadata():
        print_red("Failed to load metadata. Make sure you have run megaData_normalizer.py first.")
        return
    
    # Load trained model
    model = load_trained_model()
    if model is None:
        print_red("Failed to load model.")
        return
    
    # Example 1: Single stock prediction
    print_green("\n1. Single Stock Prediction Example")
    stock_symbol = "AAPL"
    target_datetime = "2025-07-11 15:30:00"
    
    # Check if stock is available
    available_stocks = list(predictor.stock_to_id.keys())
    if stock_symbol.lower() not in available_stocks:
        print_yellow(f"Stock {stock_symbol} not found. Using first available stock.")
        stock_symbol = available_stocks[0]
    
    results = predictor.predict_future_prices(model, stock_symbol, target_datetime)
    
    if results is not None:
        predictor.display_predictions(results)
        predictor.save_predictions(results)
        
        # Visualize predictions
        visualize_predictions(results)
    
    # Example 2: Multiple stocks prediction
    print_green("\n2. Multiple Stocks Prediction Example")
    stocks_to_predict = available_stocks[:3]  # First 3 available stocks
    
    batch_results = predict_multiple_stocks(
        predictor, model, stocks_to_predict, target_datetime
    )
    
    # Summary comparison
    if batch_results:
        print_green("\n3. Comparison Summary")
        for stock, results in batch_results.items():
            current = results['current_price']
            predicted_avg = np.mean(results['predicted_prices'])
            change_pct = ((predicted_avg - current) / current) * 100
            
            color_func = print_green if change_pct >= 0 else print_red
            color_func(f"  {stock.upper()}: ${current:.2f} → ${predicted_avg:.2f} ({change_pct:+.2f}%)")
    
    # Example 3: Different time targets
    print_green("\n4. Different Time Targets Example")
    time_targets = [
        "2025-07-11 16:00:00",  # 30 minutes later
        "2025-07-11 18:00:00",  # 2 hours later
        "2025-07-12 09:30:00",  # Next day market open
    ]
    
    for target_time in time_targets:
        print_cyan(f"\nPredicting for {target_time}:")
        results = predictor.predict_future_prices(model, stock_symbol, target_time)
        
        if results is not None:
            predicted_price = results['predicted_prices'][0]  # First prediction
            change_pct = ((predicted_price - results['current_price']) / results['current_price']) * 100
            
            color_func = print_green if change_pct >= 0 else print_red
            color_func(f"  Predicted: ${predicted_price:.2f} ({change_pct:+.2f}%)")
    
    print_green("\n" + "=" * 80)
    print_green("PREDICTION DEMO COMPLETED!")
    print_green("=" * 80)
    print_cyan("To use this system with your trained model:")
    print_cyan("1. Replace the dummy model in load_trained_model() with your actual model")
    print_cyan("2. Call predictor.predict_future_prices(model, stock_symbol, target_datetime)")
    print_cyan("3. Use the results for trading decisions or further analysis")

if __name__ == "__main__":
    main()
