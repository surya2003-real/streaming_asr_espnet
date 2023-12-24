import pyaudio as pa

def find_mics():
    p = pa.PyAudio()
    print('Available audio input devices:')
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)
            print(i, dev.get('name'))
    
    if len(input_devices):
        dev_idx = -2
        while dev_idx not in input_devices:
            print('Please type input device ID:')
            dev_idx = int(input())
    
    print(input_devices)
    return dev_idx,input_devices

# find_mics()
