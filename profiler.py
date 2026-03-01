import onnx
import os

class TaskProfiler:
    @staticmethod
    def profile_onnx(file_path):
        """Extracts layers and complexity from an ONNX AI model."""
        model = onnx.load(file_path)
        graph = model.graph
        tasks = []
        
        for node in graph.node:
            # Basic feature extraction
            task = {
                "name": node.name if node.name else node.op_type,
                "op_type": node.op_type,
                "complexity_score": 0, 
                "data_volume_mb": 0,    
                "parallelism": 0.8      
            }
            
            # Heuristic Complexity Scoring
            if node.op_type in ['Conv', 'Gemm', 'MatMul']:
                task["complexity_score"] = 100 
            elif node.op_type in ['Relu', 'Add', 'BatchNormalization']:
                task["complexity_score"] = 10  
            else:
                task["complexity_score"] = 5   
                
            tasks.append(task)
        return tasks

    @staticmethod
    def profile_raw(file_name, file_size):
        """Categorizes raw files into compute tasks."""
        ext = os.path.splitext(file_name)[1].lower()
        size_mb = file_size / (1024 * 1024)
        
        if ext == '.csv':
            return [{
                "name": "Data Sorting/Filter",
                "op_type": "Logic",
                "complexity_score": 40,
                "data_volume_mb": size_mb,
                "parallelism": 0.2 
            }]
        elif ext == '.mp4':
            return [{
                "name": "Frame Convolution",
                "op_type": "Signal Processing",
                "complexity_score": 90,
                "data_volume_mb": size_mb,
                "parallelism": 0.9 
            }]
        else:
            return [{"name": "Generic Processing", "op_type": "Unknown", "complexity_score": 50, "data_volume_mb": size_mb, "parallelism": 0.5}]