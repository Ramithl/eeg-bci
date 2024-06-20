import numpy as np
import torch
import time
from threading import Thread
from queue import Queue

class DataStreamer(Thread):
    def __init__(self, sample_rate, buffer, condition, signal, input_size=40):
        super(DataStreamer, self).__init__()
        self.sample_rate = sample_rate
        self.buffer = buffer
        self.condition = condition
        self.running = True
        self.signal = signal
        self.input_size = input_size

    def run(self):
        i = 0
        while self.running:
            # Simulate data acquisition
            new_data = self.signal[i,:]  # Generate a sampling point
            i = i+1
            with self.condition:
                self.buffer.append(new_data)
                if len(self.buffer)>self.input_size:
                  self.condition.notify_all()
            time.sleep(1/self.sample_rate)  # Simulate real-time delay
            
    def stop(self):
        self.running = False

class DataProcessor():
    def __init__(self, model, input_size=40):
        self.model = model
        self.q = Queue()
        self.running = True
        self.input_size = input_size

    def run(self):
        while self.running:
            if not self.q.empty():
                if (self.q.qsize()>self.input_size):
                    data = []
                    for _ in range(self.input_size):
                        data.append(self.q.get())
                    processed_data, label = self.preprocess_signal(data)
                    output = self.classify_signal(self.model, processed_data)
                    return output, label

    def stop(self):
        self.running = False
    
    def classify_signal(self, model, segment):
        output = model(segment)
        return output

    def group_and_flatten(self, arrays, group_size):
        grouped_arrays = []
        labels = np.array([arr[0] for arr in arrays])
        same_class = np.all(labels == labels[0])

        #print(len(arrays))
        for i in range(0, len(arrays), group_size):
            # Concatenate arrays in the group
            concatanated_arrays = np.stack(arrays[i:i + group_size])
            # Flatten the concatenated array
            flattened_array = concatanated_arrays[:, 1:].flatten()
            grouped_arrays.append(flattened_array)
        stack = np.stack(grouped_arrays)

        if same_class:
            return stack, labels[0]-1
        else:
            return stack, None
    
    def preprocess_signal(self, segment):
        # Stack the numpy arrays
        stack, label = self.group_and_flatten(segment, 4)
        # Convert the stacked NumPy array to a PyTorch tensor

        ''' ADD FILTERS HERE '''

        tensor = torch.from_numpy(stack)
        tensor = tensor.unsqueeze(0)
        input = tensor.type(torch.float32)
        return input, label
