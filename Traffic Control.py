import pyshark

capture = pyshark.LiveCapture(interface='Wi-Fi', bpf_filter='tcp port 65534')

for packet in capture.sniff_continuously(packet_count=1):
    if packet.tcp.dstport == '65534':
            print(packet.tcp.dstport)
            print(packet)
