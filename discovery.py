import psutil
import pyopencl as cl
import platform

def get_system_specs():
    specs = {
        "os": platform.system(),
        "cpu": {
            "name": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A"
        },
        "accelerators": []
    }
    
    # OpenCL discovery for GPUs/FPUs
    try:
        platforms = cl.get_platforms()
        for p in platforms:
            devices = p.get_devices()
            for d in devices:
                device_type = cl.device_type.to_string(d.type)
                specs["accelerators"].append({
                    "name": d.name.strip(),
                    "vendor": d.vendor.strip(),
                    "type": device_type,
                    "max_compute_units": d.max_compute_units
                })
    except Exception as e:
        pass 
        
    return specs