from scapy.all import TCP, IP, Raw, send

protocol_type = 'tcp'
service = 'private'
flag = 'S0'
src_bytes = 0
dst_bytes = 0
count = 123
same_srv_rate = 0.05
dst_host_srv_count = 26
dst_host_same_srv_rate = 0.1
dst_host_same_src_port_rate = 0.0


ip = IP(src="192.168.0.1", dst="192.168.0.2")

tcp = TCP(sport=1234, dport=65534, flags="S",
          options=[('MSS', 1460), ('NOP', None), ('WScale', 2), ('NOP', None), ('NOP', None), ('SAckOK', '')])

payload = Raw(load=protocol_type + ',' + service + ',' + flag + ',' + str(src_bytes) + ',' + str(dst_bytes) + ',' + str(count) + ',' + str(same_srv_rate) + ',' + str(dst_host_srv_count) + ',' + str(dst_host_same_srv_rate) + ',' + str(dst_host_same_src_port_rate))

packet = ip / tcp / payload
packet[IP].len = len(packet)


packet.show()

# send packet
send(packet)
