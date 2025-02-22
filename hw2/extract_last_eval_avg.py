import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def print_last_eval_average_return(log_dir):
    # Create an EventAccumulator instance
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Check if 'Eval_AverageReturn' is in the scalar tags
    if 'Eval_AverageReturn' not in event_acc.Tags()['scalars']:
        print("Error: 'Eval_AverageReturn' not found in the log file.")
        return

    # Get all scalar events for 'Eval_AverageReturn'
    eval_average_returns = event_acc.Scalars('Eval_AverageReturn')

    # Check if there are any events
    if not eval_average_returns:
        print("Error: No 'Eval_AverageReturn' events found in the log file.")
        return

    # Get the last event
    last_event = eval_average_returns[-1]

    # Print the last value
    print(f"Last Eval_AverageReturn: {last_event.value}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <log_directory>")
        sys.exit(1)

    log_dir = sys.argv[1]
    print_last_eval_average_return(log_dir)
