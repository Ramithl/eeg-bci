import numpy as np
import torch
import time
from threading import Thread
from queue import Queue



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

class DataProcessor():
    def __init__(self, model):
        self.model = model
        self.q = Queue()
        self.running = True

    def run(self):
        while self.running:
            if not self.q.empty():
                if (self.q.qsize()>40):
                    data = []
                    for _ in range(40):
                        data.append(self.q.get())
                    processed_data = self.preprocess_signal(data)
                    output = self.classify_signal(self.model, processed_data)
                    return output

    def stop(self):
        self.running = False
    
    def classify_signal(self, model, segment):
        output = model(segment)
        return output

    def group_and_flatten(self, arrays, group_size):
        grouped_arrays = []
        #print(len(arrays))
        for i in range(0, len(arrays), group_size):
            # Concatenate arrays in the group
            grouped_array = np.concatenate(arrays[i:i + group_size])
            # Flatten the concatenated array
            flattened_array = grouped_array.flatten()
            grouped_arrays.append(flattened_array)
        return grouped_arrays
    
    def preprocess_signal(self, segment):
        # Stack the numpy arrays
        stack = np.stack(self.group_and_flatten(segment, 4))
        # Convert the stacked NumPy array to a PyTorch tensor

        ''' ADD FILTERS HERE '''

        tensor = torch.from_numpy(stack)
        tensor = tensor.unsqueeze(0)
        input = tensor.type(torch.float32)
        return input
