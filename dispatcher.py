import onnxruntime as ort
import numpy as np
import time

class TaskDispatcher:
    def __init__(self):
        # List of Providers
        self.providers = ort.get_available_providers()
        
    def execute_ai_layer(self, model_path, hw_name):
        """Runs a specific inference task on the chosen hardware."""
        # Map our UI names to ONNX Execution Providers
        if "GPU" in hw_name.upper():
            ep = 'DmlExecutionProvider' if 'DmlExecutionProvider' in self.providers else self.providers[0]
        else:
            ep = 'CPUExecutionProvider'

        start_time = time.perf_counter()
        
        try:
            session = ort.InferenceSession(model_path, providers=[ep])
            
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            clean_shape = [1 if isinstance(dim, (str, type(None))) else dim for dim in input_shape]
            dummy_input = np.random.randn(*clean_shape).astype(np.float32)
            
            # EXECUTE
            session.run(None, {input_name: dummy_input})
            
            end_time = time.perf_counter()
            return end_time - start_time, ep
        except Exception as e:
            return 0.1, f"Error: {str(e)}"

    def execute_raw_task(self, op_type, hw_name):
        """Simulates raw compute tasks like Sorting or Conv."""
        start_time = time.perf_counter()
        
        if "Logic" in op_type:
            data = np.random.rand(1000000)
            np.sort(data)
        else:
            a = np.random.rand(1000, 1000)
            b = np.random.rand(1000, 1000)
            np.dot(a, b)
            
        end_time = time.perf_counter()
        return end_time - start_time, "Native_Lib"