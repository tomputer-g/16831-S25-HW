import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_eval_average_returns(log_dir):
    # Create an EventAccumulator instance
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Check if 'Eval_AverageReturn' is in the scalar tags
    if 'Eval_AverageReturn' not in event_acc.Tags()['scalars']:
        print("Error: 'Eval_AverageReturn' not found in the log file.")
        return None

    # Get all scalar events for 'Eval_AverageReturn'
    eval_average_returns = event_acc.Scalars('Eval_AverageReturn')

    # Check if there are any events
    if not eval_average_returns:
        print("Error: No 'Eval_AverageReturn' events found in the log file.")
        return None

    # Extract values and store them in a numpy array
    return_values = np.array([event.value for event in eval_average_returns])

    return return_values

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <log_directory>")
        sys.exit(1)

    log_dir = sys.argv[1]
    eval_returns = get_eval_average_returns(log_dir)
    
    if eval_returns is not None:
        print("Eval_AverageReturn values:")
        print(eval_returns)
