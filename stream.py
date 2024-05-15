import numpy as np
import torch
import time
from threading import Thread
from queue import Queue

def group_and_flatten(arrays, group_size):
    grouped_arrays = []
    #print(len(arrays))
    for i in range(0, len(arrays), group_size):
        # Concatenate arrays in the group
        grouped_array = np.concatenate(arrays[i:i + group_size])
        # Flatten the concatenated array
        flattened_array = grouped_array.flatten()
        grouped_arrays.append(flattened_array)
    return grouped_arrays

def preprocess_signal(segment):
    # Stack the numpy arrays
    stack = np.stack(group_and_flatten(segment, 4))
    # Convert the stacked NumPy array to a PyTorch tensor

    ''' ADD FILTERS HERE '''

    tensor = torch.from_numpy(stack)
    tensor = tensor.unsqueeze(0)
    input = tensor.type(torch.float32)
    return input

def classify_signal(model, segment):
    output = model(segment)
    print("Model output:", output)


class DataStreamer(Thread):
    def __init__(self, sample_rate, buffer, condition, signal):
        super(DataStreamer, self).__init__()
        self.sample_rate = sample_rate
        self.buffer = buffer
        self.condition = condition
        self.running = True
        self.signal = signal

    def run(self):
        i = 0
        while self.running:
            # Simulate data acquisition
            new_data = self.signal[i,:]  # Generate a sampling point
            i = i+1
            with self.condition:
                self.buffer.append(new_data)
                if len(self.buffer)>40:
                  self.condition.notify_all()
            time.sleep(1/self.sample_rate)  # Simulate real-time delay
            
    def stop(self):
        self.running = False

class DataProcessor(Thread):
    def __init__(self, model, buffer, condition):
        super(DataProcessor, self).__init__()
        self.model = model
        self.buffer = buffer
        self.condition = condition
        self.running = True

    def run(self):
        while self.running:
            with self.condition:
                while len(self.buffer)<40:
                    self.condition.wait()  # Wait until some data is available
                data = []
                for _ in range(40):
                    data.append(self.buffer.pop(0))
            processed_data = preprocess_signal(data)
            classify_signal(self.model, processed_data)

    def stop(self):
        self.running = False
